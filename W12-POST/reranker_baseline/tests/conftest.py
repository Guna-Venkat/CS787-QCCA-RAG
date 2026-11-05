import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'W12-POST'))
