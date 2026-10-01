"""Characterise the sensorimotor transfer of the connectome.

Drive each real sensory population in turn and measure how every motor pool
responds relative to baseline. This is how the simulation finds out which taste
channel drives feeding, rather than being told: the answer comes out of the
wiring.

  python sim/probe.py
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from sim.brain import Brain

SETTLE_MS = 900      # discard while the network equilibrates
MEASURE_MS = 600     # average rates over this window
DRIVE_MV = 1.5

SENSORS = [
    "grn_labellar", "grn_leg", "grn_wing", "grn_tastepeg", "grn_pharyngeal",
    "hygro", "thermo", "olfactory", "visual", "mechano_head",
    "mechano_tactile", "mechano_proprio", "chemo_other",
]
MOTORS = [
    "mn_proboscis", "mn_leg_front_L", "mn_leg_front_R",
    "mn_leg_mid_L", "mn_leg_mid_R", "mn_leg_hind_L", "mn_leg_hind_R",
    "mn_wing", "mn_abdomen", "descending", "dan", "mbon", "kenyon",
]

b = Brain(os.path.join("sim", "brain_malecns.npz"))


def run(drive_pop=None, amp=DRIVE_MV, seed=7):
    b.rng = np.random.default_rng(seed)
    b.v[:] = 0.0
    b.refrac[:] = 0.0
    b.rate[:] = 0.0
    b.alive[:] = True
    acc = {m: 0.0 for m in MOTORS}
    n = 0
    for t in range(SETTLE_MS + MEASURE_MS):
        b.clear_input()
        if drive_pop:
            b.inject(drive_pop, amp)
        b.step(excitability=1.0)
        if t >= SETTLE_MS:
            for m in MOTORS:
                acc[m] += b.pop_rate(m)
            n += 1
    return {m: acc[m] / n for m in MOTORS}, float(b.rate.mean() * 1000.0)


t0 = time.time()
print("baseline (tonic drive only) ...")
base, base_global = run(None)
print(f"  global mean rate {base_global:.2f} Hz")
print("  " + "  ".join(f"{m}={base[m]:.1f}" for m in MOTORS[:7]))

rows = {}
for s in SENSORS:
    if len(b.pop(s)) == 0:
        continue
    r, g = run(s)
    rows[s] = (r, g)
    print(f"[{time.time()-t0:5.0f}s] probed {s:16} ({len(b.pop(s)):>5,} cells) "
          f"global {g:.2f} Hz", flush=True)

print("\n" + "=" * 100)
print("change in motor firing rate (Hz) when each sensory population is driven")
print("=" * 100)
hdr = f"{'sensor':16}" + "".join(f"{m.replace('mn_','').replace('_',''):>11}" for m in MOTORS[:9])
print(hdr)
for s, (r, g) in rows.items():
    line = f"{s:16}"
    for m in MOTORS[:9]:
        line += f"{r[m]-base[m]:>11.2f}"
    print(line)

print("\n" + "=" * 100)
print("learning-circuit and pathway response (Hz change)")
print("=" * 100)
print(f"{'sensor':16}" + "".join(f"{m:>13}" for m in ("descending", "dan", "mbon", "kenyon")))
for s, (r, g) in rows.items():
    print(f"{s:16}" + "".join(f"{r[m]-base[m]:>13.2f}" for m in ("descending", "dan", "mbon", "kenyon")))

print("\n>>> which sensor most excites the proboscis (feeding) motor neurons?")
rank = sorted(rows.items(), key=lambda kv: kv[1][0]["mn_proboscis"] - base["mn_proboscis"], reverse=True)
for s, (r, g) in rank:
    print(f"    {s:16} {r['mn_proboscis']-base['mn_proboscis']:+8.2f} Hz")
