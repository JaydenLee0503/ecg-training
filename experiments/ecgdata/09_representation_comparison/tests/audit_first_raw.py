"""Read-only numerical spot check for all three completed fold-0 raw fits."""
import importlib.util
from pathlib import Path
import numpy as np
p=Path('experiments/ecgdata/09_representation_comparison/study.py').resolve();spec=importlib.util.spec_from_file_location('ecg09_raw_check',p);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
import torch
torch.set_num_threads(1);data=s.read_data();train,test=s.partition(data,0);assert not set(data['patients'][train])&set(data['patients'][test])
for seed in (0,1,2):
 folder=s.RUN/f'raw_cnn_seed{seed}_fold0';assert s.verify_trial(folder);fit=s.read_json(folder/'fit.json');assert fit['epochs']==40
 np.testing.assert_array_equal(fit['train_indices'],train);np.testing.assert_array_equal(fit['test_indices'],test)
 state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True);assert state['epoch']==40;model=s.RawCNN();model.load_state_dict(state['model']);model.eval()
 with np.load(folder/'predictions.npz',allow_pickle=False) as z:expected=z['probabilities'][:8].copy();np.testing.assert_array_equal(z['indices'],test)
 with torch.inference_mode():actual=model(torch.from_numpy(data['raw'][test[:8],None,:])).softmax(-1).numpy()
 np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=2e-6);np.testing.assert_array_equal(actual.argmax(1),expected.argmax(1));print('seed',seed,'CPU reload maximum probability error',float(abs(actual-expected).max()))
print('Three raw checkpoints, held-out patient boundaries and 40-epoch counts verified.')
