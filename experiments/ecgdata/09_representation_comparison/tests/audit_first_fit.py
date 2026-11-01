import importlib.util,sys
from pathlib import Path
import numpy as np
p=Path('experiments/ecgdata/09_representation_comparison/study.py').resolve();spec=importlib.util.spec_from_file_location('check9',p);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
from attention_ecg.model import CepstralSwin
from attention_ecg.ecgdata import fit_normalizer,normalize
import torch
x=s.read_data();f=s.RUN/'band_swin_seed0_fold0';assert s.verify_trial(f);tr,te=s.partition(x,0)
with np.load(f/'preprocessing.npz',allow_pickle=False) as z:
 m,t=fit_normalizer(x['bands'],tr);np.testing.assert_array_equal(z['mean'],m);np.testing.assert_array_equal(z['scale'],t)
with np.load(f/'predictions.npz',allow_pickle=False) as z:expected=z['probabilities'][:8].copy()
p=s.old.protocol()['model'];p['n_ceps']=32;model=CepstralSwin(**p);model.load_state_dict(torch.load(f/'checkpoint.pt',map_location='cpu',weights_only=True)['model']);model.eval();torch.set_num_threads(1)
with torch.inference_mode():actual=model(torch.from_numpy(normalize(x['bands'][te[:8]],m,t))).softmax(-1).numpy()
np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=2e-6)
print('First completed Swin checkpoint: verified normalizer and CPU probability reload, max error',abs(actual-expected).max())
