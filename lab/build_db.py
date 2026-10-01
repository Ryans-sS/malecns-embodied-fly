"""Load every FlyWire Codex CSV for a version into one DuckDB file.

Usage:  python lab/build_db.py [version]      (default: 783)
Output: lab/flywire_<version>.duckdb
"""
import gzip, os, shutil, sys, tempfile
import duckdb

VERSION = sys.argv[1] if len(sys.argv) > 1 else "783"
SRC = os.path.join("static", "data", VERSION)
OUT = os.path.join("lab", f"flywire_{VERSION}.duckdb")

TABLES = {
    "neurons": "neurons.csv.gz",
    "classification": "classification.csv.gz",
    "cell_types": "consolidated_cell_types.csv.gz",
    "cell_stats": "cell_stats.csv.gz",
    "connections": "connections.csv.gz",
    "labels": "labels.csv.gz",
    "coordinates": "coordinates.csv.gz",
    "nblast": "nblast.csv.gz",
    "connectivity_tags": "connectivity_tags.csv.gz",
}

if os.path.exists(OUT):
    os.remove(OUT)
con = duckdb.connect(OUT)
tmp = tempfile.mkdtemp(prefix="flywire_csv_")
try:
    for table, fname in TABLES.items():
        gz = os.path.join(SRC, fname)
        if not os.path.exists(gz):
            print(f"  skip {table:18} (missing {fname})")
            continue
        # DuckDB reads .gz, but decompressing first is faster and avoids
        # sniffer hiccups on the very wide nblast file.
        plain = os.path.join(tmp, fname[:-3])
        with gzip.open(gz, "rb") as fi, open(plain, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 22)
        con.execute(
            f"CREATE TABLE {table} AS SELECT * FROM read_csv(?, header=true, "
            f"sample_size=-1, all_varchar=false, ignore_errors=false)",
            [plain.replace("\\", "/")],
        )
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table:18} {n:>10,} rows")
        os.remove(plain)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# Indexes for the joins you'll actually do
for ddl in (
    "CREATE INDEX idx_conn_pre  ON connections(pre_root_id)",
    "CREATE INDEX idx_conn_post ON connections(post_root_id)",
    "CREATE INDEX idx_neurons   ON neurons(root_id)",
    "CREATE INDEX idx_class     ON classification(root_id)",
):
    try:
        con.execute(ddl)
    except Exception as e:
        print(f"  (index skipped: {e})")

# A denormalized view: every edge with both endpoints' cell type / side / class
con.execute("""
CREATE OR REPLACE VIEW cells AS
SELECT n.root_id,
       n.group AS opticlobe_group, n.nt_type,
       c.flow, c.super_class, c."class", c.sub_class, c.hemilineage, c.side, c.nerve,
       t.primary_type, t."additional_type(s)" AS additional_types,
       s.length_nm, s.area_nm, s.size_nm,
       g.connectivity_tag
FROM neurons n
LEFT JOIN classification  c USING (root_id)
LEFT JOIN cell_types      t USING (root_id)
LEFT JOIN cell_stats      s USING (root_id)
LEFT JOIN connectivity_tags g USING (root_id)
""")

con.execute("""
CREATE OR REPLACE VIEW edges AS
SELECT c.pre_root_id, c.post_root_id, c.neuropil, c.syn_count, c.nt_type,
       pt.primary_type AS pre_type,  pk.side AS pre_side,  pk.super_class AS pre_super_class,
       qt.primary_type AS post_type, qk.side AS post_side, qk.super_class AS post_super_class
FROM connections c
LEFT JOIN cell_types     pt ON pt.root_id = c.pre_root_id
LEFT JOIN classification pk ON pk.root_id = c.pre_root_id
LEFT JOIN cell_types     qt ON qt.root_id = c.post_root_id
LEFT JOIN classification qk ON qk.root_id = c.post_root_id
""")

con.close()
print(f"\nWrote {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB)")
