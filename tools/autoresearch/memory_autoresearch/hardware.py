from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .cache import hardware_profile_path


@dataclass
class HardwareProfile:
    host_name: str
    physical_memory_gb: float
    usable_budget_gb: float
    typing_batch_size: int
    embedding_batch_size: int
    reranker_batch_size: int
    calibrated_at: str


def detect_memory_gb() -> float:
    try:
        raw = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(raw) / (1024**3)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 32.0


def calibrate_profile(max_memory_target_gb: float = 52.0) -> HardwareProfile:
    physical_memory_gb = detect_memory_gb()
    usable_budget_gb = min(max_memory_target_gb, physical_memory_gb * 0.82)
    if usable_budget_gb >= 52:
        typing_batch = 48
        embedding_batch = 32
        reranker_batch = 12
    elif usable_budget_gb >= 36:
        typing_batch = 32
        embedding_batch = 20
        reranker_batch = 8
    else:
        typing_batch = 16
        embedding_batch = 10
        reranker_batch = 4
    return HardwareProfile(
        host_name=platform.node() or "unknown-host",
        physical_memory_gb=round(physical_memory_gb, 1),
        usable_budget_gb=round(usable_budget_gb, 1),
        typing_batch_size=typing_batch,
        embedding_batch_size=embedding_batch,
        reranker_batch_size=reranker_batch,
        calibrated_at=datetime.now(timezone.utc).isoformat(),
    )


def load_or_create_profile(path: Path | None = None) -> HardwareProfile:
    path = path or hardware_profile_path()
    if path.exists():
        return HardwareProfile(**json.loads(path.read_text(encoding="utf-8")))
    profile = calibrate_profile()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    return profile
