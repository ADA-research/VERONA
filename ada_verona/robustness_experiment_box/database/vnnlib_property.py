from dataclasses import dataclass
from pathlib import Path


@dataclass
class VNNLibProperty:
    """Dataclass for a VNNLib property."""

    name: str
    content: str
    path: Path = None
