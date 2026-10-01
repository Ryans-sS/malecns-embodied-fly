"""Run SQL against the FlyWire DuckDB, or drop into an interactive shell.

  python lab/q.py "select super_class, count(*) from cells group by 1 order by 2 desc"
  python lab/q.py malecns "select superclass, count(*) from cells group by 1 order by 2 desc"
  python lab/q.py                 # interactive REPL (blank line to run, .quit to exit)
  python lab/q.py malecns         # REPL against MaleCNS instead of FAFB
  python lab/q.py -f query.sql
  python lab/q.py --csv "..." > out.csv
"""
import os, sys
import duckdb

# Windows consoles default to cp1252; DuckDB draws boxes in UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

HERE = os.path.dirname(__file__)
DATASETS = {
    "fafb": ("flywire_783.duckdb", "FlyWire FAFB v783 (female brain, 139,255 neurons)"),
    "malecns": ("malecns_v10.duckdb", "Janelia MaleCNS v1.0 (male brain+VNC, 167,565 neurons)"),
}

def main():
    args = sys.argv[1:]
    as_csv = "--csv" in args
    args = [a for a in args if a != "--csv"]

    ds = "fafb"
    if args and args[0] in DATASETS:
        ds = args.pop(0)
    elif args and args[0].startswith("--"):
        cand = args[0].lstrip("-")
        if cand in DATASETS:
            ds, args = cand, args[1:]
    fname, label = DATASETS[ds]
    DB = os.path.join(HERE, fname)
    if not os.path.exists(DB):
        sys.exit(f"{DB} not found - run lab/build_db.py or lab/build_malecns_db.py first")
    con = duckdb.connect(DB, read_only=True)

    # Attach the other connectome read-only so one query can span both, e.g.
    #   SELECT m.primary_type, f.primary_type
    #   FROM mcns_to_flywire x
    #   JOIN cells m ON m.bodyId = x.id
    #   JOIN fafb.cells f ON f.root_id = x.match
    for other, (ofile, _) in DATASETS.items():
        if other == ds:
            continue
        opath = os.path.join(HERE, ofile)
        if os.path.exists(opath):
            con.execute(
                f"ATTACH '{opath.replace(chr(92), '/')}' AS {other} (READ_ONLY)"
            )

    def run(sql):
        if not sql.strip():
            return
        try:
            rel = con.sql(sql)
            if as_csv:
                rel.write_csv("/dev/stdout")
            else:
                rel.show(max_rows=60, max_width=200)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)

    if args and args[0] == "-f":
        run(open(args[1], encoding="utf-8").read())
    elif args:
        run(" ".join(args))
    else:
        print(f"FlyWire v783 @ {DB}\nTables/views: " +
              ", ".join(r[0] for r in con.execute("SHOW TABLES").fetchall()))
        print("Blank line runs the buffer; .quit exits.\n")
        buf = []
        while True:
            try:
                line = input("sql> " if not buf else "...> ")
            except EOFError:
                break
            if line.strip() in (".quit", ".exit"):
                break
            if line.strip() == "" and buf:
                run("\n".join(buf)); buf = []
            elif line.strip():
                buf.append(line)
                if line.rstrip().endswith(";"):
                    run("\n".join(buf)); buf = []
    con.close()

main()
