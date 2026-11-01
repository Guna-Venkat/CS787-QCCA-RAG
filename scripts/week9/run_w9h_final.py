#!/usr/bin/env python3
"""Manual W9-H entry point."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.week9.runner import main
if __name__ == '__main__':
    main('h')
