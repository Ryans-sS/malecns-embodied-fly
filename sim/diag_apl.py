"""Is APL doing its job, and do different odours recruit different Kenyon cells?

APL is the giant GABAergic feedback neuron of the mushroom body: ~2,300 Kenyon
cells in, ~2,300 out, ~100,000 synapses each way per hemisphere. Its role is
global normalisation - it is what keeps the KC code sparse and decorrelated.

Two things to establish:
  1. Is APL actually firing, and is its inhibition reaching the Kenyon cells?
  2. Cosine similarity of evoked rates says the cues are 0.77 similar, but for a
     sparse code what matters is WHICH cells are in the active set. Compare the
     identity of the most-driven KCs per cue, not just the rate vectors.

  python sim/diag_apl.py
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

# locate APL by cell type
ptype = np.load(os.path.join("sim", "brain_malecns.npz"), allow_pickle=True)["primary_type"]
apl = np.array([i for i in range(brain.N) if str(ptype[i]) == "APL"], dtype=np.int64)
print(f"APL cells found: {len(apl)}   Kenyon cells: {len(kc):,}")

# how much drive does APL deliver to KCs, per spike?
kc_set = np.zeros(brain.N, dtype=bool)
kc_set[kc] = True
tot_inh = 0.0
for a in apl:
    sl = slice(brain.out_start[a], brain.out_end[a])
    post = brain.edge_post[sl]
    w = brain.edge_w[sl]
    m = kc_set[post]
    tot_inh += float(w[m].sum())
print(f"APL -> KC weight per APL spike: {tot_inh:.1f} mV total across KCs "
      f"({tot_inh/len(kc):.4f} mV per KC)\n")


def present(pad, seed=51, settle=700, measure=800):
    brain.rng = np.random.default_rng(seed)
    brain.v[:] = 0.0
    brain.refrac[:] = 0.0
    brain.rate[:] = 0.0
    brain.alive[:] = True
    acc_kc = np.zeros(len(kc))
    acc_apl = 0.0
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
            acc_kc += brain.rate[kc]
            acc_apl += float(brain.rate[apl].mean())
            n += 1
    return acc_kc / n * 1000.0, acc_apl / n * 1000.0


base_kc, base_apl = present(None)
print(f"{'condition':10}{'APL Hz':>9}{'KC mean':>10}{'KC>1Hz':>9}")
print(f"{'no odour':10}{base_apl:9.2f}{base_kc.mean():10.2f}{int((base_kc>1).sum()):9,}")

ev, tops = {}, {}
TOP = 200
for p in world.pads:
    r, a = present(p)
    ev[p.name] = r - base_kc
    tops[p.name] = set(np.argsort(ev[p.name])[::-1][:TOP].tolist())
    print(f"{p.name:10}{a:9.2f}{r.mean():10.2f}{int((r>1).sum()):9,}")

names = list(ev)
print(f"\nIdentity overlap of the top {TOP} most-driven Kenyon cells (Jaccard):")
print("          " + "".join(f"{n:>9}" for n in names))
for i, n in enumerate(names):
    row = f"{n:10}"
    for j, m in enumerate(names):
        if i == j:
            row += f"{'-':>9}"
        else:
            inter = len(tops[n] & tops[m])
            row += f"{inter / (2 * TOP - inter):>9.3f}"
    print(row)

off = [len(tops[names[i]] & tops[names[j]]) / (2 * TOP - len(tops[names[i]] & tops[names[j]]))
       for i in range(len(names)) for j in range(i + 1, len(names))]
print(f"\nmean top-{TOP} identity overlap: {np.mean(off):.3f}  "
      f"(0 = completely different cells, 1 = identical)")
print("verdict:", "the active SETS are distinct - a sparse code the MB can learn on"
      if np.mean(off) < 0.4 else
      "the same cells dominate for every odour - no usable identity code")
