"""pytest configuration for ratchet tests — ensure the package root is on sys.path."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
