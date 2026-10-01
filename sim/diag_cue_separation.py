"""Can the mushroom body tell the five pads apart?

Each pad's odour is a sparse ORN subset. If the Kenyon-cell responses to the five
cues are not distinguishable, no amount of plasticity can teach the fly which
button is which - the whole request-pad experiment is impossible.

A note on the measurement: raw KC firing rates are dominated by the tonic
background every neuron receives, so comparing them directly is nearly
meaningless - two completely different odours both come out ~0.99 similar. What
the mushroom body actually reads is the EVOKED response, the change the odour
produces. This compares baseline-subtracted patterns, and prints the raw-rate
figure alongside to show how misleading it is.

  python sim/diag_cue_separation.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from sim.brain import Brain
from sim import world as W
from sim.world import World

brain = Brain(os.path.join("sim", "brain_malecns.npz"))
world = World(brain, rng=np.random.default_rng(2))
kc = brain.pop("kenyon")
print(f"{len(kc):,} Kenyon cells")
print(f"cue size: {len(world.pads[0].cue_idx):,} ORNs per pad, disjoint\n")


def present(pad, seed=51, settle=700, measure=800):
    """Mean KC firing rate while an odour is present; pad=None for no odour."""
    brain.rng = np.random.default_rng(seed)
    brain.v[:] = 0.0
    brain.refrac[:] = 0.0
    brain.rate[:] = 0.0
    brain.alive[:] = True
    acc = np.zeros(len(kc))
    n = 0
    for t in range(settle + measure):
        brain.clear_input()
        brain.inject("visual", W.LIGHT_DRIVE)
        brain.inject("mechano_proprio", 0.4)
        if pad is not None:
            brain.ext[pad.cue_L] += np.float32(W.CUE_DRIVE)
            brain.ext[pad.cue_R] += np.float32(W.CUE_DRIVE)
        brain.step()
        if t >= settle:
            acc += brain.rate[kc]
            n += 1
    return acc / n * 1000.0


baseline = present(None)
raw = {p.name: present(p) for p in world.pads}
evoked = {k: v - baseline for k, v in raw.items()}

print(f"  {'(no odour)':8} KC mean {baseline.mean():6.2f} Hz")
for name in raw:
    ev = evoked[name]
    driven = ev > 1.0
    print(f"  {name:8} raw {raw[name].mean():5.2f} Hz   evoked {ev.mean():+5.2f} Hz   "
          f"driven {driven.sum():4,} ({driven.mean() * 100:4.1f}% of KCs)")

names = list(raw)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


print("\nEVOKED pattern similarity (what the mushroom body can discriminate):")
print("          " + "".join(f"{n:>9}" for n in names))
for i, n in enumerate(names):
    row = f"{n:10}"
    for j, m in enumerate(names):
        row += f"{'-':>9}" if i == j else f"{cos(evoked[n], evoked[m]):>9.3f}"
    print(row)

off = [cos(evoked[names[i]], evoked[names[j]])
       for i in range(len(names)) for j in range(i + 1, len(names))]
raw_off = [cos(raw[names[i]], raw[names[j]])
           for i in range(len(names)) for j in range(i + 1, len(names))]

print(f"\nmean EVOKED similarity : {np.mean(off):+.3f}   <- the figure that matters")
print(f"mean RAW-RATE similarity: {np.mean(raw_off):+.3f}   <- dominated by tonic background")
print("verdict:", "KC code SEPARATES the cues - discrimination is learnable"
      if np.mean(off) < 0.75 else
      "KC patterns are entangled - the fly cannot tell the pads apart")
