import sys
from pathlib import Path

# Ensure package root is on sys.path for all tests
sys.path.append(str(Path(__file__).resolve().parents[1]))
