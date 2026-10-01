"""Read an Arrow Feather file's schema over HTTP without downloading it."""
import sys, urllib.request, pyarrow as pa, pyarrow.ipc as ipc
sys.stdout.reconfigure(encoding="utf-8")
N = 24 * 1024 * 1024
for url in sys.argv[1:]:
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{N-1}"})
    head = urllib.request.urlopen(req).read()
    if head[:6] != b"ARROW1":
        print(f"{url}: not a feather file"); continue
    sch = pa.ipc.read_schema(ipc.read_message(pa.py_buffer(head[8:])))
    print(f"\n===== {url.rsplit('/',1)[-1]}")
    for n, t in zip(sch.names, sch.types):
        print(f"   {n:26} {t}")
