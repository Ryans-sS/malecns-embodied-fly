"""Compile the MaleCNS connectome into a runnable spiking-network substrate.

Reads lab/malecns_v10.duckdb and writes sim/brain_malecns.npz containing:

  edges, sorted by presynaptic neuron so each cell's outgoing synapses are
  contiguous (this is what makes event-driven propagation cheap), each carrying a
  SIGNED weight from the presynaptic neuron's predicted neurotransmitter;

  population indices for every sensor, actuator and learning-circuit group that
  the simulation is allowed to touch. Every one of these is a real annotated
  population in the connectome - nothing here is invented.

  python sim/build_brain.py
"""
import os, sys, time
import numpy as np
import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

DB = os.path.join("lab", "malecns_v10.duckdb")
OUT = os.path.join("sim", "brain_malecns.npz")

# Sign of each predicted transmitter's effect on its targets.
#
# ACH excitatory, GABA and GLUT inhibitory follows the published whole-brain
# leaky-integrate-and-fire model of the FlyWire connectome (Shiu et al., Nature
# 2024) - glutamate is largely inhibitory in Drosophila via GluCl-alpha.
# Histamine is inhibitory (HisCl/ort) and is what photoreceptors release.
# DA/SER/OCT are neuromodulators rather than fast transmitters; giving them a
# small positive weight is a simplification, flagged as such.
NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
    "dopamine": +0.25,
    "serotonin": +0.25,
    "octopamine": +0.25,
    "unclear": +0.0,   # unknown sign contributes nothing rather than a guess
    None: +0.0,
}

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


con = duckdb.connect(DB, read_only=True)

# ---- neuron table -------------------------------------------------------
cells = con.execute("""
    SELECT bodyId, consensus_nt, superclass, "class", subclass, primary_type,
           somaSide, rootSide
    FROM cells ORDER BY bodyId
""").fetchnumpy()
ids = cells["bodyId"]
N = len(ids)
lut = np.full(int(ids.max()) + 1, -1, dtype=np.int32)
lut[ids] = np.arange(N, dtype=np.int32)
log(f"{N:,} neurons")


def col(name):
    """DuckDB gives masked arrays for nullable columns; flatten to object list."""
    c = cells[name]
    if hasattr(c, "filled"):
        c = c.filled(None)
    return [None if v is None or v is np.ma.masked else str(v) for v in c]


nt = col("consensus_nt")
superclass = col("superclass")
klass = col("class")
subclass = col("subclass")
ptype = col("primary_type")
soma_side = col("somaSide")
root_side = col("rootSide")
side = [s or r or "" for s, r in zip(soma_side, root_side)]

sign_per_neuron = np.array(
    [NT_SIGN.get((v or "").lower(), 0.0) for v in nt], dtype=np.float32
)

# ---- edges --------------------------------------------------------------
# Drop the weakest connections. At >=3 synapses this keeps 84% of all synaptic
# weight in 41% of the edges. Runtime is proportional to edges traversed, and
# connectome analyses routinely threshold at least this hard - Codex's own
# connectivity views use >=5. Raising dt was tried instead and does not help,
# because the cost scales with spikes rather than steps.
# >=3 was tried and gave no speedup over >=2 (0.46x vs 0.48x, within noise) -
# the cells that spike are hubs whose edges are mostly strong, so pruning weak
# ones barely reduces what is actually traversed. >=2 keeps more weight for free.
MIN_SYN_COUNT = 2
e = con.execute(
    f"SELECT pre_body, post_body, syn_count FROM connections "
    f"WHERE syn_count >= {MIN_SYN_COUNT}"
).fetchnumpy()
pre, post = lut[e["pre_body"]], lut[e["post_body"]]
w = e["syn_count"].astype(np.float32)
ok = (pre >= 0) & (post >= 0)
pre, post, w = pre[ok], post[ok], w[ok]
# signed by the PREsynaptic cell's transmitter
w = w * sign_per_neuron[pre]
order = np.argsort(pre, kind="stable")
pre, post, w = pre[order], post[order], w[order]
out_start = np.searchsorted(pre, np.arange(N), side="left").astype(np.int64)
out_end = np.searchsorted(pre, np.arange(N), side="right").astype(np.int64)
log(f"{len(pre):,} edges, {int((w > 0).sum()):,} excitatory / {int((w < 0).sum()):,} inhibitory")

# ---- populations --------------------------------------------------------
# Helper predicates over the real annotation columns.
def where(fn):
    return np.array([i for i in range(N) if fn(i)], dtype=np.int32)


def is_side(i, s):
    return side[i] == s


pops = {}

# --- SENSORS -------------------------------------------------------------
# Gustatory receptor neurons, grouped by the bristle class they belong to.
# MaleCNS types GRNs anatomically (LB* labellar, LgLG*/LgAG* leg, WG* wing,
# tpGRN taste peg) and does NOT record which tastant each responds to. So these
# are kept as separate, honestly-unlabelled taste channels; which one drives
# ingestion is something the simulation reveals, not something asserted here.
def grn_group(prefix):
    return where(
        lambda i: klass[i] == "gustatory"
        and (ptype[i] or "").startswith(prefix)
    )


pops["grn_labellar"] = grn_group("LB")
pops["grn_leg"] = np.concatenate([grn_group("LgLG"), grn_group("LgAG")])
pops["grn_wing"] = grn_group("WG")
pops["grn_tastepeg"] = where(
    lambda i: klass[i] == "gustatory" and "tpGRN" in (ptype[i] or "")
)
pops["grn_pharyngeal"] = where(
    lambda i: klass[i] == "gustatory" and subclass[i] == "pharyngeal sensillum"
)

# Hygrosensory receptor neurons - humidity. VP1d/VP4/VP5 are the real annotated
# hygrosensory types; again their dry/moist polarity is not asserted here.
pops["hygro"] = where(lambda i: klass[i] == "hygrosensory")
pops["thermo"] = where(lambda i: klass[i] == "thermosensory")
pops["olfactory"] = where(lambda i: klass[i] == "olfactory")

# Olfactory laterality. Sensory neurons have no somaSide in MaleCNS - their cell
# bodies sit in the antenna, outside the imaged CNS - so side is derived from
# where they actually project: an olfactory receptor neuron targets one antennal
# lobe. This gives the two "antennae" a fly needs to compare, which is what
# chemotaxis runs on. 2,630 of 2,639 ORNs lateralise cleanly this way.
_al = con.execute("""
    WITH orn AS (SELECT bodyId FROM cells WHERE "class" = 'olfactory')
    SELECT er.pre_body AS body,
           sum(CASE WHEN er.neuropil = 'AL(L)' THEN er.syn_count ELSE 0 END) AS l,
           sum(CASE WHEN er.neuropil = 'AL(R)' THEN er.syn_count ELSE 0 END) AS r
    FROM edge_rois er SEMI JOIN orn o ON o.bodyId = er.pre_body
    GROUP BY 1
""").fetchall()
_olf_L, _olf_R = [], []
for body, l, r in _al:
    i = pos = lut[int(body)]
    if i < 0:
        continue
    if l > r:
        _olf_L.append(i)
    elif r > l:
        _olf_R.append(i)
pops["olfactory_L"] = np.array(sorted(_olf_L), dtype=np.int32)
pops["olfactory_R"] = np.array(sorted(_olf_R), dtype=np.int32)
pops["visual"] = where(lambda i: klass[i] == "visual")
pops["mechano_tactile"] = where(
    lambda i: (klass[i] or "").startswith("mechanosensory_tactile")
)
pops["mechano_proprio"] = where(
    lambda i: (klass[i] or "").startswith("mechanosensory_proprioceptive")
)
pops["mechano_head"] = where(lambda i: klass[i] == "mechanosensory")
pops["chemo_other"] = where(lambda i: klass[i] == "chemosensory")

# --- ACTUATORS -----------------------------------------------------------
# Proboscis muscle motor neurons: the canonical readout of "the fly is eating".
pops["mn_proboscis"] = where(
    lambda i: superclass[i] == "cb_motor" and subclass[i] == "pm"
)
# Leg motor neurons by segment and side -> locomotion and turning.
for tag, sc in (("front", "fl"), ("mid", "ml"), ("hind", "hl")):
    for s in ("L", "R"):
        pops[f"mn_leg_{tag}_{s}"] = where(
            lambda i, sc=sc, s=s: superclass[i] == "vnc_motor"
            and subclass[i] == sc
            and is_side(i, s)
        )
pops["mn_wing"] = where(
    lambda i: superclass[i] == "vnc_motor" and subclass[i] == "wm"
)
pops["mn_abdomen"] = where(
    lambda i: superclass[i] == "vnc_motor" and subclass[i] == "ad"
)
pops["mn_neck"] = where(
    lambda i: superclass[i] in ("vnc_motor", "cb_motor") and subclass[i] == "nm"
)

# --- LEARNING CIRCUIT ----------------------------------------------------
# The mushroom body: Kenyon cells carry a sparse code of the current cue, DANs
# carry reinforcement, MBONs read out valence. KC->MBON synapses depressed by
# coincident DAN activity is the established plasticity rule in Drosophila.
pops["kenyon"] = where(lambda i: klass[i] == "Kenyon_Cell")
pops["mbon"] = where(lambda i: klass[i] == "MBON")
pops["dan"] = where(lambda i: klass[i] == "DAN")

# --- PATHWAY -------------------------------------------------------------
pops["descending"] = where(lambda i: superclass[i] == "descending_neuron")

# The steering command neurons. DNa01 and DNa02 are the best-characterised
# turn-control descending neurons in Drosophila: their activity drives IPSILATERAL
# turning, and DNa02 in particular both correlates with and causes turns. MaleCNS
# types one of each per side, so the fly has a real left/right steering command to
# be read out instead of an average over leg motor pools.
for _t in ("DNa01", "DNa02", "DNa03"):
    for _s in ("L", "R"):
        pops[f"dn_{_t}_{_s}"] = where(
            lambda i, t=_t, sd=_s: ptype[i] == t and side[i] == sd
        )
pops["dn_steer_L"] = np.concatenate([pops["dn_DNa01_L"], pops["dn_DNa02_L"]])
pops["dn_steer_R"] = np.concatenate([pops["dn_DNa01_R"], pops["dn_DNa02_R"]])
pops["ascending"] = where(lambda i: superclass[i] == "ascending_neuron")
pops["endocrine"] = where(lambda i: (superclass[i] or "").endswith("endocrine"))

for k, v in pops.items():
    log(f"  {k:22} {len(v):>7,}")

# ---- KC -> MBON synapse locations, for plasticity -----------------------
kc_mask = np.zeros(N, dtype=bool); kc_mask[pops["kenyon"]] = True
mbon_mask = np.zeros(N, dtype=bool); mbon_mask[pops["mbon"]] = True
kc_to_mbon = np.flatnonzero(kc_mask[pre] & mbon_mask[post]).astype(np.int64)
log(f"KC->MBON synapses available for plasticity: {len(kc_to_mbon):,}")

np.savez_compressed(
    OUT,
    ids=ids,
    out_start=out_start,
    out_end=out_end,
    edge_post=post.astype(np.int32),
    edge_w=w.astype(np.float32),
    edge_pre=pre.astype(np.int32),
    kc_to_mbon=kc_to_mbon,
    nt_sign=sign_per_neuron,
    superclass=np.array(superclass, dtype=object),
    primary_type=np.array(ptype, dtype=object),
    side=np.array(side, dtype=object),
    **{f"pop_{k}": v for k, v in pops.items()},
)
con.close()
log(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB)")
