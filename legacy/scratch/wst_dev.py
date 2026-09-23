# %%
%load_ext autoreload
%autoreload 2
import numpy as np, matplotlib.pyplot as plt
import ecgvmd as E
# %%
import os, ecgvmd, importlib
print(ecgvmd.__path__)
print(sorted(os.listdir(ecgvmd.__path__[0])))
# %%  run ONCE, then comment out
"""
ds = E.load_ecgdata()
W, y, g = E.segment(ds, E.FAST)          # 6 windows/record -> 972 x 500, ~4 MB
np.savez("scratch/windows.npz", W=W, y=y, g=g)
"""

# %%  the cell you actually re-run
z = np.load("scratch/windows.npz", allow_pickle=True)
W, y, g = z["W"], z["y"], z["g"]

fs, N, Npad = 128.0, 500, 512
freqs = np.fft.rfftfreq(Npad, 1/fs)
idx = {c: np.where(y == c)[0][0] for c in E.CLASS_ORDER}
x = W[idx["NSR"]]
'''
print(ds)
print(W.shape, W.dtype)
print(np.unique(y, return_counts=True))
print(len(np.unique(g)))
print(W.mean(-1)[:3], W.std(-1)[:3])
'''
# %%
plt.plot(np.arange(500) / 128, W[0]); plt.xlabel("s")
# %%
importlib.invalidate_caches()
import ecgvmd.scatter as S_
S_.f()
# %%
%pip install kymatio

# %%
# %%
from kymatio.numpy import Scattering1D
S = Scattering1D(J=6, shape=500, Q=8)
Sx = S(W[:4])
m = S.meta()
print("out :", Sx.shape, Sx.dtype)
print("meta:", {k: np.shape(v) for k, v in m.items()})
print("order counts:", np.bincount(m["order"]))
print("xi[:3]  :", m["xi"][:3])
print("xi[-3:] :", m["xi"][-3:])