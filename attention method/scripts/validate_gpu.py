"""Check CUDA inference, gradients and checkpoint recovery on synthetic inputs."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from attention_ecg.ecgdata import PROTOCOL, read_json, atomic_json, file_hash
from attention_ecg.model import CepstralSwin
from attention_ecg import ecgdata_training as training


def main():
    start = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this check')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = read_json(PROTOCOL)
    torch.manual_seed(0)
    cpu = CepstralSwin(**protocol['model']).eval()
    gpu = CepstralSwin(**protocol['model']).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    x = torch.from_numpy(np.random.default_rng(917).normal(size=(6, 13, 1, 16)).astype(np.float32))
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    expected, actual = cpu(x), gpu(x.cuda())
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-4)
    loss_cpu = torch.nn.functional.cross_entropy(expected, y)
    loss_gpu = torch.nn.functional.cross_entropy(actual, y.cuda())
    loss_cpu.backward()
    loss_gpu.backward()
    max_gradient_error = 0.0
    for p, q in zip(cpu.parameters(), gpu.parameters()):
        assert torch.isfinite(q.grad).all()
        torch.testing.assert_close(q.grad.cpu(), p.grad, atol=2e-5, rtol=2e-3)
        max_gradient_error = max(max_gradient_error, float((q.grad.cpu() - p.grad).abs().max()))

    # Exercise actual optimizer and resume code, without reading any ECG signals.
    data = dict(y=np.tile(np.repeat(np.arange(3), 2), 5),
                patients=np.repeat([f'synthetic_{i}' for i in range(15)], 2),
                fold=np.repeat(np.arange(5), 6),
                X=np.random.default_rng(918).normal(size=(30, 13, 1, 16)).astype(np.float32))
    protocol['training']['epochs'] = 2
    protocol['training']['batch_size'] = 8
    with tempfile.TemporaryDirectory(prefix='ecg-swin-cuda-validation-') as tmp:
        full, resumed = Path(tmp)/'full', Path(tmp)/'resumed'
        training.fit_swin(full, data, 0, 0, protocol, 'cuda')
        original_save = training.save_checkpoint

        def stop_after_epoch(folder, state):
            original_save(folder, state)
            if state['epoch'] == 1:
                raise InterruptedError('Deliberate synthetic recovery check')

        try:
            with patch.object(training, 'save_checkpoint', side_effect=stop_after_epoch):
                training.fit_swin(resumed, data, 0, 0, protocol, 'cuda')
        except InterruptedError:
            pass
        else:
            raise AssertionError('Interruption was not exercised')
        assert training.load_checkpoint(resumed)['epoch'] == 1
        training.fit_swin(resumed, data, 0, 0, protocol, 'cuda')
        a, b = training.load_checkpoint(full), training.load_checkpoint(resumed)
        assert a['epoch'] == b['epoch'] == 2
        for key in a['model']:
            torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
        for ah, bh in zip(a['history'], b['history']):
            assert ah['loss'] == bh['loss'] and ah['lr'] == bh['lr']
        with np.load(full/'predictions.npz') as fa, np.load(resumed/'predictions.npz') as fb:
            np.testing.assert_array_equal(fa['probabilities'], fb['probabilities'])
        assert training.verify_trial(full) and training.verify_trial(resumed)

    result = dict(status='passed', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  data='synthetic only; no ECG fitting', torch=str(torch.__version__),
                  cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                  capability=list(torch.cuda.get_device_capability()),
                  compiled_architectures=torch.cuda.get_arch_list(),
                  parameters=sum(p.numel() for p in cpu.parameters()),
                  max_forward_absolute_error=float((actual.detach().cpu()-expected.detach()).abs().max()),
                  max_gradient_absolute_error=max_gradient_error,
                  interrupted_resume_weights_and_predictions='bitwise equal to uninterrupted fit',
                  seconds=time.perf_counter()-start,
                  source_sha256=file_hash(__file__))
    atomic_json(ROOT/'reports/ecgdata_gpu_validation_2026-09-29.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
