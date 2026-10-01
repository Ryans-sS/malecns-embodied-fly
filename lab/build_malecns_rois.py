"""Derive per-edge brain regions for MaleCNS from the synapse-partner table.

syn-partners carries primary_post: the neuropil the postsynaptic site sits in.
Aggregating it to (pre, post, roi) gives MaleCNS the per-edge region column the
flat connectome otherwise lacks. Adds to lab/malecns_v10.duckdb:

    edge_rois   pre_body, post_body, neuropil, syn_count
    rois        neuropil, n_synapses   (the observed ROI vocabulary)

  python lab/build_malecns_rois.py
"""
import os, sys, time
import duckdb, pyarrow as pa, pyarrow.ipc as ipc

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

RAW = os.path.join("static", "data", "malecns_raw")
SRC = os.path.join(RAW, "syn-partners-male-cns-v1.0-minconf-0.5.feather")
DB = os.path.join("lab", "malecns_v10.duckdb")
SCRATCH = os.environ.get("TEMP", ".")
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


con = duckdb.connect(DB)
con.execute("SET preserve_insertion_order = false")
con.execute("SET memory_limit = '14GB'")
con.execute(f"SET temp_directory = '{os.path.join(SCRATCH, 'duckdb_spill').replace(chr(92), '/')}'")

con.execute("DROP TABLE IF EXISTS edge_rois")
con.execute("DROP TABLE IF EXISTS rois")

with open(SRC, "rb") as fh:
    rdr = ipc.open_file(fh)
    sch = rdr.schema
    keep = ["body_pre", "body_post", "primary_post"]
    idx = [sch.names.index(k) for k in keep]
    # primary_post is dictionary-encoded; cast to plain string for DuckDB.
    out_sch = pa.schema(
        [
            (k, pa.string() if pa.types.is_dictionary(sch.types[i]) else sch.types[i])
            for k, i in zip(keep, idx)
        ]
    )

    def batches():
        for i in range(rdr.num_record_batches):
            b = rdr.get_batch(i)
            yield pa.RecordBatch.from_arrays(
                [
                    b.column(j).cast(t)
                    for j, t in zip(idx, out_sch.types)
                ],
                schema=out_sch,
            )

    log(f"streaming {rdr.num_record_batches} batches from syn-partners...")
    reader = pa.RecordBatchReader.from_batches(out_sch, batches())
    con.register("sp", reader)
    con.execute("""
        CREATE TABLE edge_rois AS
        SELECT s.body_pre AS pre_body, s.body_post AS post_body,
               coalesce(s.primary_post, 'UNASGD') AS neuropil,
               count(*)::BIGINT AS syn_count
        FROM sp s
        SEMI JOIN neuron_ids p ON p.bodyId = s.body_pre
        SEMI JOIN neuron_ids q ON q.bodyId = s.body_post
        GROUP BY 1, 2, 3
    """)
    con.unregister("sp")

n = con.execute("SELECT count(*) FROM edge_rois").fetchone()[0]
syn = con.execute("SELECT sum(syn_count) FROM edge_rois").fetchone()[0]
log(f"edge_rois {n:,} (pre,post,roi) rows covering {syn:,} synapses")

con.execute("""
CREATE TABLE rois AS
SELECT neuropil, sum(syn_count)::BIGINT AS n_synapses, count(*)::BIGINT AS n_edges
FROM edge_rois GROUP BY 1 ORDER BY n_synapses DESC
""")
log(f"rois {con.execute('SELECT count(*) FROM rois').fetchone()[0]:,} distinct regions")

try:
    con.execute("CREATE INDEX idx_mc_eroi ON edge_rois(pre_body)")
except Exception as e:
    print(f"  (index skipped: {e})")

print()
con.sql("SELECT * FROM rois LIMIT 25").show(max_rows=30)
con.close()
