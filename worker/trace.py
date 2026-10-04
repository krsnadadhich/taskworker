"""Run trace: a JSONL event log plus a human-readable report with evidence."""
import json
import shutil
import time
from pathlib import Path


class Trace:
    def __init__(self, run_dir: Path, echo: bool = True):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "evidence").mkdir(exist_ok=True)
        self._f = open(self.dir / "trace.jsonl", "a", encoding="utf-8")
        self.echo = echo
        self.t0 = time.monotonic()

    def event(self, kind: str, **data) -> None:
        rec = {"t": round(time.monotonic() - self.t0, 2), "kind": kind, **data}
        self._f.write(json.dumps(rec, default=str) + "\n")
        self._f.flush()

    def say(self, line: str) -> None:
        if self.echo:
            print(line, flush=True)

    def keep(self, src: Path) -> Path:
        dst = self.dir / "evidence" / src.name
        if src.resolve() != dst.resolve():
            shutil.copy(src, dst)
        return dst

    def close(self) -> None:
        self._f.close()
