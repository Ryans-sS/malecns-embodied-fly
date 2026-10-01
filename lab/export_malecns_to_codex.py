"""Export MaleCNS v1.0 from DuckDB into Codex's CSV schema.

Writes static/data/malecns/*.csv.gz so the Codex web app can serve MaleCNS as a
selectable dataset version alongside FAFB 783.

  python lab/export_malecns_to_codex.py [min_syn]     # default min_syn = 3

Requires lab/malecns_v10.duckdb built by, in order:
  lab/build_malecns_db.py, lab/build_malecns_rois.py, lab/build_malecns_matches.py
"""
import csv, gzip, os, sys, time
import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codex.data.malecns_roi_naming import to_codex_region  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Codex holds the whole connectome as nested Python dicts and sets, so the UI
# graph is thresholded. >=3 keeps 84% of all synapses in 10.5M edges; the
# complete 25.6M-edge graph stays in DuckDB. See LOCAL_SETUP.md.
MIN_SYN = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DB = os.path.join("lab", "malecns_v10.duckdb")
OUT = os.path.join("static", "data", "malecns")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# MaleCNS spells transmitters out; Codex uses short codes. HA and UNK are local
# additions to codex/data/neurotransmitters.py.
NT = {
    "acetylcholine": "ACH", "glutamate": "GLUT", "gaba": "GABA",
    "dopamine": "DA", "serotonin": "SER", "octopamine": "OCT",
    "histamine": "HA", "unclear": "UNK", None: "UNK", "": "UNK",
}
# Codex carries a per-transmitter average column for six of them only.
AVG_COL = {"DA": 0, "SER": 1, "GABA": 2, "GLUT": 3, "ACH": 4, "OCT": 5}

# MaleCNS soma coordinates are in 8nm voxels; Codex stores nanometres.
MALECNS_VOXEL_NM = 8


def nt_of(raw):
    return NT.get((raw or "").lower(), "UNK")


# MaleCNS superclass -> Codex flow, matching how FAFB 783 assigns it.
def flow_of(sc):
    if not sc:
        return ""
    if sc.endswith("_intrinsic"):
        return "intrinsic"
    if "sensory" in sc or sc == "ascending_neuron":
        return "afferent"
    if "motor" in sc or sc == "descending_neuron" or sc == "endocrine":
        return "efferent"
    if sc.startswith("visual_"):
        return "intrinsic"
    return ""


SIDE = {"L": "left", "R": "right", "M": "center"}

con = duckdb.connect(DB, read_only=True)


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def write_gz(name, header, rows):
    path = os.path.join(OUT, name)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(header)
        for r in rows:
            w.writerow(r)
            n += 1
    log(f"{name:34} {n:>10,} rows  ({os.path.getsize(path)/1e6:.1f} MB)")


def stream(sql, size=100000):
    q = con.execute(sql)
    while True:
        batch = q.fetchmany(size)
        if not batch:
            break
        for row in batch:
            yield row


# ---- neurons.csv --------------------------------------------------------
def neuron_rows():
    for body, nt_raw, conf in stream(
        "SELECT bodyId, consensus_nt, predicted_nt_confidence FROM cells ORDER BY bodyId"
    ):
        nt = nt_of(nt_raw)
        avgs = ["0"] * 6
        if nt in AVG_COL and conf is not None:
            avgs[AVG_COL[nt]] = f"{conf:.4f}"
        yield [body, "", nt, f"{conf:.4f}" if conf is not None else ""] + avgs


write_gz(
    "neurons.csv.gz",
    ["root_id", "group", "nt_type", "nt_type_score",
     "da_avg", "ser_avg", "gaba_avg", "glut_avg", "ach_avg", "oct_avg"],
    neuron_rows(),
)

# ---- classification.csv -------------------------------------------------
def class_rows():
    for body, sc, cls, sub, hl, ss, rs, nerve in stream("""
        SELECT bodyId, superclass, "class", subclass,
               hemilineage_itolee, somaSide, rootSide,
               coalesce(entryNerve, exitNerve)
        FROM cells ORDER BY bodyId
    """):
        side = SIDE.get(ss) or SIDE.get(rs) or ""
        yield [body, flow_of(sc), sc or "", cls or "", sub or "",
               hl or "", side, nerve or ""]


write_gz(
    "classification.csv.gz",
    ["root_id", "flow", "super_class", "class", "sub_class",
     "hemilineage", "side", "nerve"],
    class_rows(),
)

# ---- consolidated_cell_types.csv ---------------------------------------
write_gz(
    "consolidated_cell_types.csv.gz",
    ["root_id", "primary_type", "additional_type(s)"],
    (
        [b, t or "", ""]
        for b, t in stream("SELECT bodyId, primary_type FROM cells ORDER BY bodyId")
    ),
)

# ---- coordinates.csv ----------------------------------------------------
# Positions are nanometres, as in FAFB's file. Codex converts them per dataset
# (see cached_cell_details). MaleCNS has no supervoxel ids, so that column is 0.
write_gz(
    "coordinates.csv.gz",
    ["root_id", "position", "supervoxel_id"],
    (
        [b, f"[{x * MALECNS_VOXEL_NM} {y * MALECNS_VOXEL_NM} {z * MALECNS_VOXEL_NM}]", 0]
        for b, x, y, z in stream("""
            SELECT bodyId, soma_x, soma_y, soma_z FROM cells
            WHERE soma_x IS NOT NULL AND soma_y IS NOT NULL AND soma_z IS NOT NULL
            ORDER BY bodyId
        """)
    ),
)

# ---- nblast.csv ---------------------------------------------------------
# MaleCNS publishes no all-by-all top-N match table (the full matrix is 166k
# columns wide), but it does publish top-5 matches against the left/right
# flipped dataset. So Codex's "similar cells" shows contralateral homologs,
# which is the more useful comparison anyway. Codex wants integer scores 1-9.
def nblast_rows():
    cur, scores = None, []
    for body, match, score in stream("""
        SELECT id, match, score FROM contra_matches
        WHERE score > 0 ORDER BY id, rank
    """):
        s = min(9, max(1, int(round(score * 10))))
        if body != cur:
            if cur is not None and scores:
                yield [cur, ";".join(scores)]
            cur, scores = body, []
        scores.append(f"{match}:{s}")
    if cur is not None and scores:
        yield [cur, ";".join(scores)]


write_gz("nblast.csv.gz", ["root_id", "scores"], nblast_rows())

# ---- labels.csv ---------------------------------------------------------
# Codex treats labels as free-text search fodder. MaleCNS has no community
# annotations, but it does carry instance names, synonyms and matched type
# names in the other three connectomes - all worth being able to search.
def label_rows():
    lid = 0
    for body, instance, synonyms, fw, hb, manc in stream("""
        SELECT bodyId, instance, synonyms, flywireType, hemibrainType, mancType
        FROM cells ORDER BY bodyId
    """):
        parts = []
        if instance:
            parts.append(instance)
        if synonyms:
            parts.extend(p.strip() for p in synonyms.split(","))
        for name, src in ((fw, "FlyWire"), (hb, "hemibrain"), (manc, "MANC")):
            if name:
                parts.append(f"{src} type: {name}")
        seen = set()
        for p in parts:
            p = p.strip()
            if not p or p.lower() in seen:
                continue
            seen.add(p.lower())
            lid += 1
            yield [body, p, 0, "", 0, lid, "2026-06-08", "MaleCNS v1.0", "Janelia FlyEM"]


write_gz(
    "labels.csv.gz",
    ["root_id", "label", "user_id", "position", "supervoxel_id",
     "label_id", "date_created", "user_name", "user_affiliation"],
    label_rows(),
)

# ---- connections.csv ----------------------------------------------------
# One row per (pre, post, neuropil), same shape as FAFB's file. The threshold is
# applied to each edge's TOTAL weight, then that edge is split across regions --
# thresholding per-region would silently drop the tails of strong connections.
def conn_rows():
    for pre, post, roi, w, nt_raw in stream(f"""
        SELECT er.pre_body, er.post_body, er.neuropil, er.syn_count, n.consensus_nt
        FROM edge_rois er
        SEMI JOIN (
            SELECT pre_body, post_body FROM connections WHERE syn_count >= {MIN_SYN}
        ) c ON c.pre_body = er.pre_body AND c.post_body = er.post_body
        LEFT JOIN neurotransmitters n ON n.body = er.pre_body
        ORDER BY er.pre_body, er.post_body
    """, size=200000):
        yield [pre, post, to_codex_region(roi), w, nt_of(nt_raw)]


write_gz(
    "connections.csv.gz",
    ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"],
    conn_rows(),
)

# ---- no MaleCNS equivalent: header only --------------------------------
for name, header in (
    # MaleCNS publishes no per-neuron cable length / surface area / volume.
    ("cell_stats.csv.gz", ["root_id", "length_nm", "area_nm", "size_nm"]),
    ("connectivity_tags.csv.gz", ["root_id", "connectivity_tag"]),
):
    write_gz(name, header, iter(()))

con.close()
print(f"\nWrote {OUT} (min_syn={MIN_SYN})")
