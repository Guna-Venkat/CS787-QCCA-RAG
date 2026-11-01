"""Nested entry point avoids changing A–G global source-fingerprint inventory."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.week9.final_h05.producer import main
if __name__=='__main__':main()
