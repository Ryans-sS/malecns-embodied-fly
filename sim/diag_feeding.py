"""Why can't the fly swallow while standing on a dispensing sugar pad?

probe.py measured taste-peg drive at 4 mV producing +22.6 Hz in the proboscis
motor pool. On the pad it produces under +2 Hz. Something else in the arena's
sensory mix is cancelling it. Add the inputs one at a time and watch.
"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from sim.brain import Brain
from sim.body import Body
from sim import world as W
from sim.world import World

b = Brain(os.path.join("sim", "brain_malecns.npz"))
b.measure_baseline(["mn_proboscis"])
base = b.baseline["mn_proboscis"]
world = World(b, rng=np.random.default_rng(2))
pad = world.pads[0]
print(f"proboscis baseline {base:.2f} Hz;  threshold to eat is +{W.PROBOSCIS_FEED_DELTA} Hz\n")

def run(parts, amp_taste=W.TASTE_DRIVE, settle=900, meas=900, seed=41):
    b.rng = np.random.default_rng(seed)
    b.v[:]=0; b.refrac[:]=0; b.rate[:]=0; b.alive[:]=True
    acc=0.0; n=0; grn=0.0
    for t in range(settle+meas):
        b.clear_input()
        if "taste" in parts: b.inject("grn_tastepeg", amp_taste)
        if "cue" in parts:
            b.ext[pad.cue_L] += np.float32(W.CUE_DRIVE)
            b.ext[pad.cue_R] += np.float32(W.CUE_DRIVE)
        if "light" in parts: b.inject("visual", W.LIGHT_DRIVE)
        if "proprio" in parts: b.inject("mechano_proprio", 0.5)
        if "head" in parts: b.inject("mechano_head", 0.4)
        b.step()
        if t>=settle:
            acc += b.delta_rate("mn_proboscis"); grn += b.pop_rate("grn_tastepeg"); n+=1
    return acc/n, grn/n

print(f"{'inputs active':44}{'proboscis Δ':>13}{'tastepeg Hz':>13}  eats?")
for parts in (["taste"], ["taste","light"], ["taste","light","proprio"],
              ["taste","light","proprio","head"], ["taste","cue"],
              ["taste","cue","light","proprio","head"]):
    d,g = run(parts)
    print(f"{'+'.join(parts):44}{d:+13.2f}{g:13.1f}  {'YES' if d>=W.PROBOSCIS_FEED_DELTA else 'no'}")

print(f"\n{'taste amplitude sweep (taste only)':44}")
for amp in (1.0, 2.0, 3.0, 4.0, 6.0):
    d,g = run(["taste"], amp_taste=amp)
    print(f"{('  '+str(amp)+' mV'):44}{d:+13.2f}{g:13.1f}  {'YES' if d>=W.PROBOSCIS_FEED_DELTA else 'no'}")
