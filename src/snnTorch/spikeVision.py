import snntorch as snn
from snntorch.spikevision import spikedata
from pathlib import Path
from dvs_gesture_cache import prepare_cache

from torch.utils.data import DataLoader

# create datasets
train_ds = spikedata.DVSGesture("dataset/dvsgesture", train=True)
test_ds = spikedata.DVSGesture("dataset/dvs_gesture", train=False)

# create dataloaders
train_dl = DataLoader(train_ds, shuffle=True, batch_size=64)
test_dl = DataLoader(test_ds, shuffle=False, batch_size=64)


import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

# choose a random sample
n = 125

# index into a single sample and sum the on/off channels
events, target = train_ds[n]
a = events[:, 0] + events[:, 1]

#  Plot
fig, ax = plt.subplots()
anim = splt.animator(a, fig, ax, interval=10)
HTML(anim.to_html5_video())

anim.save('dvsgesture_animator.mp4', writer = 'ffmpeg', fps=50)