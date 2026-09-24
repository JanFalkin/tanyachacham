"""Embed every chunk with one model: resumable, logged, with a heat guard.

Long GPU runs on this machine are unattended, in a room without AC, on a box
that has hung once. So:

  - Resumable. Vectors are written in shards of `--shard` chunks, each
    atomically (write .tmp, then rename). Stop it any way you like -- Ctrl-C,
    power, a hang -- and the same command resumes; at most one shard is lost.
  - Heat guard. Every 30s a thread samples GPU and CPU temperature. Past
    --gpu-max or --cpu-max, work pauses between slices of 256 chunks and
    resumes once both are 8C below their limits.
  - Temperature log. Every sample goes to temps.csv beside the vectors, so a
    night's thermal history can be read in the morning.
  - Fingerprinted. meta.json records a hash of the chunks table. If chunks are
    rebuilt, old shards would silently pair vectors with the wrong text -- a
    wrong citation -- so the run refuses to resume against them.

  - Precision. --fp16 runs the model in half precision (Ada and later; not
    Pascal). Stored vectors are float32 either way. The dtype is in
    meta.json, so a run cannot resume with shards from the other precision.

Row i of the concatenated shards is chunks.id = i. Vectors are L2-normalized,
so cosine similarity is a dot product.

Usage:
    python -m embed.embed --model intfloat/multilingual-e5-base
    python -m embed.embed --model BAAI/bge-m3 --fp16
    python -m embed.embed --model ... --limit 600    # smoke test, separate dir
Watch:
    tail -f data/vectors/<model>/temps.csv
"""
from __future__ import annotations

import argparse, hashlib, json, sqlite3, subprocess, threading, time
from pathlib import Path

import numpy as np

from embed.bench import PREFIX, load

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "corpus.db"
OUT     = ROOT / "data" / "vectors"
SLICE   = 256       # chunks between heat checks: ~10s on e5-base, ~25s on bge-m3


def gpu_stats() -> tuple[float | None, float | None, float | None]:
    """-> (temp C, power W, utilization %) from nvidia-smi, each None if
    unreadable. Fields are parsed one by one: the 4060 in the dock reports
    power.draw as "[N/A]", and parsing all three together turned that into a
    missing temperature -- a heat guard that could never trip."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        fields = out.splitlines()[0].split(",")
    except Exception:
        return None, None, None
    def num(x):
        try:
            return float(x)
        except ValueError:
            return None
    t, w, u = (num(x) for x in (fields + [""] * 3)[:3])
    return t, w, u


def cpu_temp() -> float | None:
    """Ryzen Tctl from k10temp."""
    for h in Path("/sys/class/hwmon").iterdir():
        try:
            if (h / "name").read_text().strip() == "k10temp":
                return int((h / "temp1_input").read_text()) / 1000
        except OSError:
            pass
    return None


class HeatGuard(threading.Thread):
    """Samples temperatures into a CSV; `hot` is set while a limit is exceeded
    and cleared only once both readings are `margin` below their limits, so
    the run does not flap on and off at the threshold."""

    def __init__(self, log: Path, gpu_max: float, cpu_max: float,
                 every: float = 30, margin: float = 8):
        super().__init__(daemon=True)
        self.log, self.gpu_max, self.cpu_max = log, gpu_max, cpu_max
        self.every, self.margin = every, margin
        self.hot, self.stop = threading.Event(), threading.Event()
        self.ready = threading.Event()      # set after the first sample
        self.done, self.last = 0, (None, None, None, None)

    def run(self):
        new = not self.log.exists()
        with open(self.log, "a") as f:
            if new:
                f.write("time,gpu_c,gpu_w,gpu_util,cpu_c,chunks_done,paused\n")
            while not self.stop.is_set():
                g, w, u = gpu_stats(); c = cpu_temp()
                self.last = (g, w, u, c)
                over = ((g is not None and g >= self.gpu_max) or
                        (c is not None and c >= self.cpu_max))
                cool = ((g is None or g <= self.gpu_max - self.margin) and
                        (c is None or c <= self.cpu_max - self.margin))
                if over and not self.hot.is_set():
                    self.hot.set()
                    print(f"  HEAT PAUSE  gpu={g}C cpu={c}C "
                          f"(limits {self.gpu_max}/{self.cpu_max})", flush=True)
                elif cool and self.hot.is_set():
                    self.hot.clear()
                    print(f"  resumed     gpu={g}C cpu={c}C", flush=True)
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{g},{w},{u},{c},"
                        f"{self.done},{int(self.hot.is_set())}\n")
                f.flush()
                self.ready.set()
                self.stop.wait(self.every)


def fingerprint(db: sqlite3.Connection) -> tuple[int, str]:
    h, n = hashlib.sha1(), 0
    for i, ref, k in db.execute("SELECT id, first_ref, tokens FROM chunks ORDER BY id"):
        if i != n:
            raise SystemExit("chunk ids are not contiguous -- rebuild with embed.chunk")
        h.update(f"{i}|{ref}|{k}\n".encode()); n += 1
    return n, h.hexdigest()[:16]


def main(model_name: str, shard: int, gpu_max: float, cpu_max: float,
         limit: int, batch: int, dtype: str = "fp32") -> None:
    db = sqlite3.connect(DB_PATH)
    n, fp = fingerprint(db)
    if not n:
        raise SystemExit("chunks is empty -- run `python -m embed.chunk run` first")
    if limit:
        n = min(n, limit)
    out = OUT / (model_name.replace("/", "__") + ("-smoke" if limit else ""))
    out.mkdir(parents=True, exist_ok=True)

    meta = {"model": model_name, "chunks": n, "fingerprint": fp, "shard": shard,
            "dtype": dtype}
    mp = out / "meta.json"
    if mp.exists():
        old = json.loads(mp.read_text())
        old.setdefault("dtype", "fp32")     # written before --fp16 existed
        if {k: old.get(k) for k in meta} != meta:
            raise SystemExit(f"{out} holds vectors for different chunks or settings"
                             f" -- delete it to start over.\n  on disk: {old}\n  now:     {meta}")
    else:
        mp.write_text(json.dumps(meta, indent=1))

    shards = list(range((n + shard - 1) // shard))
    size = lambda s: min((s + 1) * shard, n) - s * shard
    todo = [s for s in shards if not (out / f"{s:05d}.npy").exists()]
    remaining = sum(size(s) for s in todo)
    print(f"{model_name} ({dtype}): {n:,} chunks in {len(shards)} shards, "
          f"{len(shards) - len(todo)} already done, {remaining:,} chunks to go", flush=True)
    if not todo:
        return

    model = load(model_name, dtype)
    prefix = PREFIX.get(model_name, "")

    guard = HeatGuard(out / "temps.csv", gpu_max, cpu_max)
    guard.done = n - remaining
    guard.start()
    # Work must not start before the first reading: in a hot room the first
    # slice would otherwise run unguarded (this happened in testing).
    if not guard.ready.wait(60) or guard.last[0] is None:
        guard.stop.set()
        raise SystemExit("no GPU temperature reading -- not starting blind")
    t0, done_now = time.time(), 0
    try:
        for s in todo:
            lo, hi = s * shard, s * shard + size(s)
            texts = [prefix + t for (t,) in db.execute(
                "SELECT text FROM chunks WHERE id >= ? AND id < ? ORDER BY id", (lo, hi))]
            parts = []
            for i in range(0, len(texts), SLICE):
                while guard.hot.is_set():
                    time.sleep(5)
                parts.append(model.encode(texts[i:i + SLICE], batch_size=batch,
                                          normalize_embeddings=True,
                                          convert_to_numpy=True))
            vecs = np.concatenate(parts).astype(np.float32)
            assert len(vecs) == hi - lo
            tmp = out / f"{s:05d}.tmp.npy"
            np.save(tmp, vecs)
            tmp.rename(out / f"{s:05d}.npy")

            done_now += hi - lo; guard.done += hi - lo
            rate = done_now / (time.time() - t0)
            g, w, u, c = guard.last
            print(f"  shard {s + 1:>4}/{len(shards)}  {guard.done:>9,}/{n:,}  "
                  f"{rate:5.1f} chunks/s  ETA {(remaining - done_now) / rate / 3600:4.1f}h  "
                  f"gpu {g}C {w}W  cpu {c}C", flush=True)
    except KeyboardInterrupt:
        print("\nstopped -- run the same command to resume", flush=True)
    finally:
        guard.stop.set(); guard.join(timeout=15)
    if done_now == remaining:
        print(f"done: {n:,} vectors in {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="embed.embed")
    ap.add_argument("--model", required=True)
    ap.add_argument("--shard", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gpu-max", type=float, default=80)
    ap.add_argument("--cpu-max", type=float, default=90)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N chunks")
    ap.add_argument("--fp16", action="store_true", help="half precision (Ada or later)")
    a = ap.parse_args()
    main(a.model, a.shard, a.gpu_max, a.cpu_max, a.limit, a.batch,
         "fp16" if a.fp16 else "fp32")
