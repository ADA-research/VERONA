from dataclasses import dataclass
from pathlib import Path

@dataclass
class VNNLibProperty:
        name: str
        content: str
        path: Path = None
