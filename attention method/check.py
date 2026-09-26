"""Synthetic forward-only engineering check. Does not read ECG data or train."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_ecg import CepstralConfig, cepstral_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-only', action='store_true')
    args = parser.parse_args()
    cfg = CepstralConfig()
    rng = np.random.default_rng(0)
    t = np.arange(5000) / cfg.fs
    signal = np.sin(2 * np.pi * 1.2 * t)[:, None] * np.linspace(.5, 1.5, 12)
    signal += .05 * rng.standard_normal(signal.shape)
    features, meta = cepstral_features(signal, cfg)
    report = dict(status='synthetic_check_only', training_started=False,
                  feature_shape=list(features.shape), feature_config=cfg.to_dict(),
                  right_pad_samples=meta['right_pad_samples'])
    if not args.features_only:
        import torch
        from attention_ecg.model import CepstralSwin
        torch.manual_seed(0)
        torch.set_num_threads(1)
        model = CepstralSwin().eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.inference_mode():
            logits = model(torch.from_numpy(features[None]))
        assert torch.isfinite(logits).all()
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
        report.update(torch_version=torch.__version__, model_config=model.config,
                      parameters=sum(p.numel() for p in model.parameters()),
                      output_shape=list(logits.shape), weights_unchanged=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
