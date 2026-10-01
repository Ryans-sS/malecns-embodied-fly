"""Find a cue strength that serves discrimination, feeding and steering at once.

Three things depend on how hard a pad's odour drives the receptor neurons, and
they were tuned against each other by accident:

  * the mushroom body needs the cue strong enough that Kenyon-cell responses are
    odour-specific rather than dominated by the network's own background;
  * the feeding pathway must not be swamped;
  * chemotaxis needs a left/right difference big enough to steer on.

Cue drive was dropped 1.80 -> 0.75 to stop it blocking feeding, but the real
cause of that turned out to be a spurious mechano_head injection, since removed.
So the constraint may no longer bind. This measures all three at once.

A note on the overlap metric. An earlier version compared the top 200 Kenyon
cells regardless of how many were actually active, so at high thresholds it was
comparing ~30 driven cells plus 170 cells of noise and reported nonsense. Here
the comparison is between ACTIVE SETS - cells whose evoked response clears a
threshold - which is density-matched by construction.

  python sim/diag_operating_point.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from sim.brain import Brain
from sim.body import Body
from sim import world as W
from sim.world import World

EVOKED_THRESH = 0.5     # Hz above baseline to count a Kenyon cell as driven
DRIVES = [0.75, 1.50, 2.50, 4.00]

brain = Brain(os.path.join("sim", "brain_malecns.npz"))
brain.measure_baseline(["mn_proboscis"])
kc = brain.pop("kenyon")
ptype = np.load(os.path.join("sim", "brain_malecns.npz"), allow_pickle=True)["primary_type"]
apl = np.array([i for i in range(brain.N) if str(ptype[i]) == "APL"], dtype=np.int64)


def kc_response(world, pad, seed=51, settle=700, measure=800):
    brain.rng = np.random.default_rng(seed)
    brain.v[:] = 0.0
    brain.refrac[:] = 0.0
    brain.rate[:] = 0.0
    brain.alive[:] = True
    acc = np.zeros(len(kc))
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
            acc += brain.rate[kc]
            acc_apl += float(brain.rate[apl].mean())
            n += 1
    return acc / n * 1000.0, acc_apl / n * 1000.0


def feeding(world, seed=61, settle=800, measure=900):
    """Proboscis response while standing on a dispensing sugar pad."""
    body = Body(brain.N, rng=np.random.default_rng(7))
    body.energy = 0.20
    pad = world.pads[0]
    pad.deliver_ms = 60000.0
    world.x, world.y, world.theta = pad.x, pad.y, 0.0
    brain.rng = np.random.default_rng(seed)
    brain.v[:] = 0.0
    brain.refrac[:] = 0.0
    brain.rate[:] = 0.0
    brain.alive[:] = True
    acc = 0.0
    n = 0
    for t in range(settle + measure):
        world.apply_senses(body)
        brain.step(excitability=body.excitability)
        world.x, world.y = pad.x, pad.y
        if t >= settle:
            acc += brain.delta_rate("mn_proboscis")
            n += 1
    pad.deliver_ms = 0.0
    return acc / n


def steer_dprime(world, seed=71, settle=700, measure=1200):
    """Separation of the steering imbalance, pad left vs pad right, over noise."""
    body = Body(brain.N, rng=np.random.default_rng(9))
    pad = world.pads[0]
    out = []
    for bearing_deg in (+90, -90):
        bearing = math.radians(bearing_deg)
        world.theta = 0.0
        world.x = pad.x - 5.0 * math.cos(bearing)
        world.y = pad.y - 5.0 * math.sin(bearing)
        brain.rng = np.random.default_rng(seed)
        brain.v[:] = 0.0
        brain.refrac[:] = 0.0
        brain.rate[:] = 0.0
        brain.alive[:] = True
        s = []
        for t in range(settle + measure):
            world.apply_senses(body)
            brain.step(excitability=1.0)
            world.x = pad.x - 5.0 * math.cos(bearing)
            world.y = pad.y - 5.0 * math.sin(bearing)
            if t >= settle:
                L = float(brain.rate[brain.steer_idxL] @ brain.steer_wL) * 1000.0
                R = float(brain.rate[brain.steer_idxR] @ brain.steer_wR) * 1000.0
                s.append(L - R)
        out.append(np.array(s))
    pooled = math.sqrt((out[0].var() + out[1].var()) / 2) + 1e-9
    return abs(out[0].mean() - out[1].mean()) / pooled


print(f"{'cue mV':>7}{'APL Hz':>8}{'KC act':>8}{'sparse%':>9}"
      f"{'overlap':>9}{'probosc':>9}{'eats':>6}{'steer d-prime':>15}")
for drive in DRIVES:
    W.CUE_DRIVE = drive
    world = World(brain, rng=np.random.default_rng(2))
    base, _ = kc_response(world, None)
    sets, dens = {}, []
    apl_hz = 0.0
    for p in world.pads:
        r, a = kc_response(world, p)
        ev = r - base
        sets[p.name] = set(np.flatnonzero(ev > EVOKED_THRESH).tolist())
        dens.append(len(sets[p.name]))
        apl_hz = a
    names = list(sets)
    ov = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_, b_ = sets[names[i]], sets[names[j]]
            u = len(a_ | b_)
            ov.append(len(a_ & b_) / u if u else 1.0)
    prob = feeding(world, )
    dp = steer_dprime(world)
    m = float(np.mean(ov)) if ov else 1.0
    print(f"{drive:7.2f}{apl_hz:8.1f}{int(np.mean(dens)):8,}"
          f"{np.mean(dens)/len(kc)*100:9.1f}{m:9.3f}{prob:+9.1f}"
          f"{'YES' if prob >= W.PROBOSCIS_FEED_DELTA else 'no':>6}{dp:15.2f}", flush=True)

print(f"\nactive set = Kenyon cells with evoked response > {EVOKED_THRESH} Hz")
print("overlap is Jaccard between those sets - density-matched, unlike top-N")
print(f"feeding needs proboscis >= +{W.PROBOSCIS_FEED_DELTA} Hz; steering needs d-prime > ~1")
