"""ig-dash reads run directories as the trainer writes them."""
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

from intgrad import dash


def _run(root, name, recs, final=None, samples=()):
    d = root / name
    (d / "samples").mkdir(parents=True)
    (d / "config.toml").write_text('name = "x"\n[train]\nsteps = 100\nlr = 0.001\n')
    (d / "meta.json").write_text(json.dumps({"git": "abc", "created": int(time.time())}))
    lines = [json.dumps(r) for r in recs]
    if final is not None:
        lines.append(json.dumps({"step": 100, "eval": {"bpb": final, "se": 0.01, "books": {"bpb": 1.5, "n": 3}},
                                 "final": True}))
    (d / "metrics.jsonl").write_text("\n".join(lines) + "\n" + '{"step": 10')   # torn last line
    for s in samples:
        (d / "samples" / f"{s:07d}.bin").write_bytes("héllo\x03\nwörld\x03\n".encode())
    return d


def test_summary_status_and_curves(tmp_path):
    train = [{"step": s, "lr": 1e-3, "loss": 2.0, "ce_bits": 2.5 - s / 100, "time": s * 0.5, "kB_per_s": 50}
             for s in range(0, 60, 10)]
    _run(tmp_path, "live", train + [{"step": 50, "val_quick": {"bpb": 1.9, "se": 0.1}}], samples=[50])
    _run(tmp_path, "done", train, final=1.6)
    runs = dash.Runs(tmp_path)
    by = {r["name"]: r for r in (runs.summary(d) for d in runs.dirs())}
    assert by["live"]["status"] == "running" and by["live"]["eta_s"] == 25
    assert by["live"]["quick"] == 1.9 and by["live"]["eval"] is None
    assert by["done"]["status"] == "done" and by["done"]["eval"] == 1.6
    assert by["done"]["sources"] == {"books": 1.5}
    assert by["done"]["config"]["train.lr"] == 0.001
    c = runs.curves(tmp_path / "live")
    assert len(c["train"]) == 6 and c["quick"] == [[50, 1.9]] and c["samples"] == [50]
    assert runs.samples(tmp_path / "live", "50") == ["héllo", "wörld"]
    assert runs.get("../live") is None and runs.get("nope") is None


def test_server_serves_page_and_api(tmp_path):
    _run(tmp_path, "r", [{"step": 0, "lr": 1e-3, "loss": 2.0, "ce_bits": 3.0, "time": 1.0, "kB_per_s": 1}])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), dash.handler(dash.Runs(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"intgrad runs" in urllib.request.urlopen(base + "/").read()
        assert json.loads(urllib.request.urlopen(base + "/api/runs").read())[0]["name"] == "r"
        assert json.loads(urllib.request.urlopen(base + "/api/run?name=r").read())["train"][0][1] == 3.0
        try:
            urllib.request.urlopen(base + "/api/run?name=..%2Fx")
            assert False
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        srv.shutdown()
