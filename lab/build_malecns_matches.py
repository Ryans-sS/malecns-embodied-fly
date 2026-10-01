"""Load the MaleCNS NBLAST top-5 match tables into DuckDB.

Adds to lab/malecns_v10.duckdb, each as (id, match_n, score_n) long form:

    contra_matches      MaleCNS neuron -> its 5 best left/right-flipped matches
                        (match_1 is effectively the contralateral homolog)
    mcns_to_flywire     MaleCNS neuron -> 5 best FlyWire 783 root ids
    flywire_to_mcns     FlyWire 783 root id -> 5 best MaleCNS neurons

Together the last two are a morphology-based crosswalk between the two
connectomes. Source: gs://flyem-male-cns/v1.0/nblasts/ (see its README).

  python lab/build_malecns_matches.py
"""
import os, sys
import duckdb, pyarrow.feather as ft

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

RAW = os.path.join("static", "data", "malecns_raw")
DB = os.path.join("lab", "malecns_v10.duckdb")

FILES = {
    "contra_matches": "matches_mcns_v1.0_contra.feather",
    "mcns_to_flywire": "matches_mcns_v1.0_flywire783.feather",
    "flywire_to_mcns": "matches_flywire783_mcns_v1.0.feather",
}

con = duckdb.connect(DB)
con.execute("SET preserve_insertion_order = false")

for table, fname in FILES.items():
    path = os.path.join(RAW, fname)
    if not os.path.exists(path):
        print(f"  skip {table} (missing {fname})")
        continue
    con.register("m_arrow", ft.read_table(path, memory_map=True))
    con.execute(f"DROP TABLE IF EXISTS {table}")
    # wide (match_1..5 / score_1..5) -> long (rank, match, score)
    unions = " UNION ALL ".join(
        f"SELECT id, {i} AS rank, match_{i} AS match, score_{i} AS score FROM m_arrow"
        for i in range(1, 6)
    )
    con.execute(
        f"CREATE TABLE {table} AS SELECT * FROM ({unions}) "
        f"WHERE match IS NOT NULL AND score IS NOT NULL"
    )
    con.unregister("m_arrow")
    n, ids = con.execute(
        f"SELECT count(*), count(DISTINCT id) FROM {table}"
    ).fetchone()
    print(f"  {table:18} {n:>9,} rows  {ids:>9,} distinct source ids")
    try:
        con.execute(f"CREATE INDEX idx_{table} ON {table}(id)")
    except Exception as e:
        print(f"    (index skipped: {e})")

print()
con.sql("""
SELECT rank, count(*) n, round(min(score),3) min_score,
       round(avg(score),3) avg_score, round(max(score),3) max_score
FROM contra_matches GROUP BY 1 ORDER BY 1
""").show()
con.close()
