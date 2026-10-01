"""Load the Janelia MaleCNS v1.0 flat-connectome Feather files into DuckDB.

Source files live in static/data/malecns_raw/ (see LOCAL_SETUP.md for URLs).
Output: lab/malecns_v10.duckdb

  python lab/build_malecns_db.py
"""
import os, sys, time
import duckdb, pyarrow as pa, pyarrow.feather as ft, pyarrow.ipc as ipc

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

RAW = os.path.join("static", "data", "malecns_raw")
OUT = os.path.join("lab", "malecns_v10.duckdb")
F_ANN = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
F_NT = "body-neurotransmitters-male-cns-v1.0.feather"
F_STATS = "body-stats-male-cns-v1.0-minconf-0.5.feather"
F_WT = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# What MaleCNS calls a neuron: a body that has been assigned a superclass. This
# is the project's own criterion -- the NBLAST README at
# gs://flyem-male-cns/v1.0/nblasts/ says the NBLASTs cover "all MaleCNS neurons
# with a `superclass`, regardless of status ... 166,700 neurons", which lands on
# the published 166,691 within 9 bodies. A status-based cut does not: Traced
# alone is 165,122 and Traced+Anchor+Assign is 167,565.
NEURON_PREDICATE = "superclass IS NOT NULL"

if os.path.exists(OUT):
    os.remove(OUT)
con = duckdb.connect(OUT)
con.execute("SET preserve_insertion_order = false")
con.execute("SET memory_limit = '12GB'")
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


# ---- annotations (small, read whole) -------------------------------------
ann = ft.read_table(os.path.join(RAW, F_ANN), memory_map=True)
# somaLocation/tosomaLocation are list<int64>; flatten to x/y/z for easy SQL.
keep = [n for n in ann.schema.names if n not in ("somaLocation", "tosomaLocation")]
ann_flat = ann.select(keep)
con.register("ann_arrow", ann_flat)
con.execute("CREATE TABLE annotations AS SELECT * FROM ann_arrow")
con.unregister("ann_arrow")
for src, pfx in (("somaLocation", "soma"), ("tosomaLocation", "tosoma")):
    col = ann[src].combine_chunks()
    tbl = pa.table(
        {
            "bodyId": ann["bodyId"],
            f"{pfx}_x": pa.array([v[0] if v else None for v in col.to_pylist()], pa.int64()),
            f"{pfx}_y": pa.array([v[1] if v else None for v in col.to_pylist()], pa.int64()),
            f"{pfx}_z": pa.array([v[2] if v and len(v) > 2 else None for v in col.to_pylist()], pa.int64()),
        }
    )
    con.register("loc_arrow", tbl)
    con.execute(f"CREATE TABLE {pfx}_locations AS SELECT * FROM loc_arrow")
    con.unregister("loc_arrow")
log(f"annotations {con.execute('SELECT count(*) FROM annotations').fetchone()[0]:,} rows")

# ---- neurotransmitters ---------------------------------------------------
nt = ft.read_table(os.path.join(RAW, F_NT), memory_map=True)
con.register("nt_arrow", nt)
con.execute("CREATE TABLE neurotransmitters AS SELECT * FROM nt_arrow")
con.unregister("nt_arrow")
log(f"neurotransmitters {con.execute('SELECT count(*) FROM neurotransmitters').fetchone()[0]:,} rows")

# ---- body stats: 88M segments in the file; keep the annotated ones ------
# The full table is mostly sub-micron debris. Ad-hoc access to all 88M rows is
# still possible straight off the Feather file with pyarrow.
with open(os.path.join(RAW, F_STATS), "rb") as fh:
    rdr = ipc.open_file(fh)
    sch = rdr.schema
    # status_fine is a dictionary column; cast to plain string for DuckDB.
    cast_to = pa.schema(
        [(n, pa.string() if pa.types.is_dictionary(t) else t) for n, t in zip(sch.names, sch.types)]
    )

    def batches():
        for i in range(rdr.num_record_batches):
            yield rdr.get_batch(i).cast(cast_to)

    reader = pa.RecordBatchReader.from_batches(cast_to, batches())
    con.register("stats_stream", reader)
    con.execute(
        "CREATE TABLE body_stats AS SELECT s.* FROM stats_stream s "
        "SEMI JOIN annotations a ON a.bodyId = s.body"
    )
    con.unregister("stats_stream")
log(f"body_stats {con.execute('SELECT count(*) FROM body_stats').fetchone()[0]:,} rows (of 88.4M segments)")

# ---- connectome weights: full 151.9M-edge graph, streamed ---------------
with open(os.path.join(RAW, F_WT), "rb") as fh:
    rdr = ipc.open_file(fh)
    sch = rdr.schema

    def wbatches():
        for i in range(rdr.num_record_batches):
            yield rdr.get_batch(i)

    reader = pa.RecordBatchReader.from_batches(sch, wbatches())
    con.register("wt_stream", reader)
    con.execute("CREATE TABLE weights AS SELECT * FROM wt_stream")
    con.unregister("wt_stream")
log(f"weights {con.execute('SELECT count(*) FROM weights').fetchone()[0]:,} edges")

# ---- neuron-only connectome --------------------------------------------
con.execute(f"""
CREATE TABLE neuron_ids AS
SELECT bodyId FROM annotations WHERE {NEURON_PREDICATE}
""")
con.execute("""
CREATE TABLE connections AS
SELECT w.body_pre AS pre_body, w.body_post AS post_body, w.weight AS syn_count
FROM weights w
SEMI JOIN neuron_ids p ON p.bodyId = w.body_pre
SEMI JOIN neuron_ids q ON q.bodyId = w.body_post
""")
log(f"connections {con.execute('SELECT count(*) FROM connections').fetchone()[0]:,} neuron-to-neuron edges")

for ddl in (
    "CREATE INDEX idx_mc_ann  ON annotations(bodyId)",
    "CREATE INDEX idx_mc_pre  ON connections(pre_body)",
    "CREATE INDEX idx_mc_post ON connections(post_body)",
    "CREATE INDEX idx_mc_st   ON body_stats(body)",
):
    try:
        con.execute(ddl)
    except Exception as e:
        print(f"  (index skipped: {e})")

# ---- views -------------------------------------------------------------
con.execute(f"""
CREATE OR REPLACE VIEW cells AS
SELECT a.bodyId,
       a.type AS primary_type, a.instance, a.superclass, a."class", a.subclass,
       a.somaSide, a.rootSide, a.somaNeuromere, a.entryNerve, a.exitNerve,
       a.itoleeHl AS hemilineage_itolee, a.trumanHl AS hemilineage_truman,
       a.status, a.statusLabel, a.dimorphism, a.receptorType, a.synonyms,
       a.flywireType, a.hemibrainType, a.mancType,
       n.consensus_nt, n.predicted_nt, n.predicted_nt_confidence,
       s.pre, s.post, s.downstream, s.synweight, s.rank AS synweight_rank,
       sl.soma_x, sl.soma_y, sl.soma_z
FROM annotations a
LEFT JOIN neurotransmitters n ON n.body = a.bodyId
LEFT JOIN body_stats        s ON s.body = a.bodyId
LEFT JOIN soma_locations   sl ON sl.bodyId = a.bodyId
WHERE a.{NEURON_PREDICATE}
""")

con.execute("""
CREATE OR REPLACE VIEW edges AS
SELECT c.pre_body, c.post_body, c.syn_count,
       pa.type AS pre_type,  pa.superclass AS pre_superclass,  pa.somaSide AS pre_side,
       qa.type AS post_type, qa.superclass AS post_superclass, qa.somaSide AS post_side,
       pn.consensus_nt AS pre_nt
FROM connections c
LEFT JOIN annotations pa ON pa.bodyId = c.pre_body
LEFT JOIN annotations qa ON qa.bodyId = c.post_body
LEFT JOIN neurotransmitters pn ON pn.body = c.pre_body
""")

n_cells = con.execute("SELECT count(*) FROM cells").fetchone()[0]
n_syn = con.execute("SELECT sum(syn_count) FROM connections").fetchone()[0]
log(f"views ready: cells={n_cells:,}  neuron-neuron synapses={n_syn:,}")
con.close()
print(f"\nWrote {OUT} ({os.path.getsize(OUT)/1e9:.1f} GB)")
