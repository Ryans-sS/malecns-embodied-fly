"""Find every descending neuron that actually carries a lateralised odour signal.

DNa01/DNa02 are the textbook steering neurons, but MaleCNS contains exactly one of
each per side - four cells in total - and a firing-rate estimate from four cells is
mostly shot noise. Too noisy to steer with, which the approach assay confirmed.

So rather than trusting the textbook pair, measure all 1,314 descending neurons:
drive the left antenna, drive the right antenna, and keep the cells whose response
differs between the two. Those are the ones that know where the odour is.

Writes sim/steer_readout.npz with the selected cells and their weights.

  python sim/probe_steering.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from sim.brain import Brain
from sim.body import Body
from sim import world as W
from sim.world import World

SETTLE_MS = 800
MEASURE_MS = 1200
REPEATS = 3          # different noise seeds, so selection is not fitting noise
TOP_K = 60           # cells kept per side

brain = Brain(os.path.join("sim", "brain_malecns.npz"))
dn = brain.pop("descending")
side = np.load(os.path.join("sim", "brain_malecns.npz"), allow_pickle=True)["side"]
dn_side = np.array([str(side[i]) for i in dn])
print(f"{len(dn):,} descending neurons  "
      f"({int((dn_side=='L').sum())} left, {int((dn_side=='R').sum())} right)")

# Selection must use the stimulus the fly will actually receive. An earlier
# version drove a whole antennal lobe (931-1,699 ORNs at 0.88 mV) and picked 182
# beautifully lateralised cells - which then did nothing in the arena, because a
# pad cue only drives its own sparse subset. Here the probe reproduces the exact
# arena stimulus: one pad's cue, split across the two antennal lobes with the same
# bearing-dependent gains apply_senses() uses, at a representative distance.
body = Body(brain.N, rng=np.random.default_rng(5))
world = World(brain, rng=np.random.default_rng(2))
pad = world.pads[0]
DIST = 4.0
STRENGTH = W.CUE_DRIVE * (1.0 - DIST / W.CUE_R)
GL_HARD = 1.0 + W.CUE_ANTENNA_SHARPNESS
GR_HARD = 1.0 - W.CUE_ANTENNA_SHARPNESS
print(f"probe stimulus = one pad cue ({len(pad.cue_L)} L + {len(pad.cue_R)} R ORNs) "
      f"at {STRENGTH:.2f} mV, gains {GL_HARD:.2f}/{GR_HARD:.2f}")


def mean_rates(pad_on_left, seed):
    brain.rng = np.random.default_rng(seed)
    brain.v[:] = 0.0
    brain.refrac[:] = 0.0
    brain.rate[:] = 0.0
    brain.alive[:] = True
    gL, gR = (GL_HARD, GR_HARD) if pad_on_left else (GR_HARD, GL_HARD)
    acc = np.zeros(len(dn))
    n = 0
    for t in range(SETTLE_MS + MEASURE_MS):
        brain.clear_input()
        # same background senses the fly always has, so selection is not done in
        # an artificially quiet brain
        brain.inject("visual", W.LIGHT_DRIVE)
        brain.inject("mechano_proprio", 0.4)
        brain.ext[pad.cue_L] += np.float32(STRENGTH * gL)
        brain.ext[pad.cue_R] += np.float32(STRENGTH * gR)
        brain.step()
        if t >= SETTLE_MS:
            acc += brain.rate[dn]
            n += 1
    return acc / n * 1000.0


diffs = []
for rep in range(REPEATS):
    seed = 100 + rep * 17
    rL = mean_rates(True, seed)
    rR = mean_rates(False, seed)
    diffs.append(rL - rR)
    print(f"  repeat {rep+1}: mean |L-R| across DNs = {np.abs(rL-rR).mean():.3f} Hz, "
          f"max = {np.abs(rL-rR).max():.2f} Hz", flush=True)

D = np.array(diffs)
mean_d = D.mean(axis=0)
std_d = D.std(axis=0) + 1e-6
# keep cells whose lateralisation is consistent across noise seeds
consistency = np.abs(mean_d) / std_d
score = np.abs(mean_d) * np.clip(consistency, 0, 3)

print(f"\ncells with a consistent lateralised response (|mean| > 2x std): "
      f"{int((consistency > 2).sum())}")

sel = np.argsort(score)[::-1]
sel = [i for i in sel if consistency[i] > 1.5][:TOP_K * 2]
if not sel:
    print("no DN carries a consistent lateralised odour signal - cannot steer on this")
    sys.exit(1)

# A cell that fires MORE when the left antenna is driven votes "odour is left".
# Group by the sign of that preference rather than by soma side, because what
# matters for a readout is which way the cell reports, not where it lives.
left_pref = [i for i in sel if mean_d[i] > 0][:TOP_K]
right_pref = [i for i in sel if mean_d[i] < 0][:TOP_K]
print(f"kept {len(left_pref)} left-preferring and {len(right_pref)} right-preferring DNs")

ptype = np.load(os.path.join("sim", "brain_malecns.npz"), allow_pickle=True)["primary_type"]
print("\nstrongest left-preferring:")
for i in left_pref[:8]:
    print(f"    {str(ptype[dn[i]]):12} side={dn_side[i]:1}  "
          f"L-R {mean_d[i]:+7.2f} Hz  consistency {consistency[i]:5.1f}")
print("strongest right-preferring:")
for i in right_pref[:8]:
    print(f"    {str(ptype[dn[i]]):12} side={dn_side[i]:1}  "
          f"L-R {mean_d[i]:+7.2f} Hz  consistency {consistency[i]:5.1f}")

out = os.path.join("sim", "steer_readout.npz")
np.savez_compressed(
    out,
    left_idx=dn[np.array(left_pref, dtype=int)].astype(np.int32),
    right_idx=dn[np.array(right_pref, dtype=int)].astype(np.int32),
    left_w=np.abs(mean_d[np.array(left_pref, dtype=int)]).astype(np.float32),
    right_w=np.abs(mean_d[np.array(right_pref, dtype=int)]).astype(np.float32),
)
sep = mean_d[np.array(left_pref, dtype=int)].sum() - mean_d[np.array(right_pref, dtype=int)].sum()
print(f"\ntotal lateralised signal available: {sep:.1f} Hz "
      f"(was ~3.8 Hz from the DNa01/DNa02 pair alone)")
print(f"wrote {out}")
