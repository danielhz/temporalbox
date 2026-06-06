"""
Lightweight experiment logging to JSON Lines.

Each run appends one record per epoch (or per checkpoint) to a .jsonl file
in the results/ directory.  The run is identified by a short ID derived from
the config hash and a timestamp so that reruns with different seeds produce
separate files.

Usage:
    logger = ExperimentLogger.from_config(cfg, results_dir="results")
    for epoch in range(n_epochs):
        metrics = {"loss": 0.42, "hits1": 0.85}
        logger.log(epoch, metrics)
    logger.close()
"""

from __future__ import annotations
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunMeta:
    run_id: str
    config: dict[str, Any]
    host: str
    start_time: float  # unix timestamp


class ExperimentLogger:
    def __init__(self, path: Path, meta: RunMeta) -> None:
        self._path = path
        self._meta = meta
        self._file = path.open("a")
        # Write header record.
        self._write({"type": "meta", **asdict(meta)})

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        results_dir: str | Path = "results",
        run_id: str | None = None,
    ) -> ExperimentLogger:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        if run_id is None:
            cfg_hash = hashlib.md5(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest()[:8]
            ts = int(time.time())
            run_id = f"{ts}_{cfg_hash}"

        meta = RunMeta(
            run_id=run_id,
            config=config,
            host=os.uname().nodename,
            start_time=time.time(),
        )
        path = results_dir / f"{run_id}.jsonl"
        return cls(path, meta)

    def log(self, epoch: int, metrics: dict[str, Any]) -> None:
        self._write({"type": "metrics", "epoch": epoch, "t": time.time(), **metrics})

    def log_eval(self, split: str, metrics: dict[str, Any]) -> None:
        self._write({"type": "eval", "split": split, "t": time.time(), **metrics})

    def close(self) -> None:
        self._write({"type": "done", "t": time.time()})
        self._file.close()

    def _write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    @property
    def run_id(self) -> str:
        return self._meta.run_id

    @property
    def path(self) -> Path:
        return self._path
