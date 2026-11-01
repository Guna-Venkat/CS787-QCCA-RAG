"""Isolated official compressor; one JSON request, CPU only, synthetic qualification."""
import json,sys,contextlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('NLTK_DATA',str(ROOT/'results/week12/tools/nltk_data'))
request=json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
 import torch
 torch.set_num_threads(4)
 from llmlingua import PromptCompressor
 compressor=PromptCompressor(model_name=request['snapshot'],use_llmlingua2=True,device_map='cpu')
 result=compressor.compress_prompt_llmlingua2(request['text'],rate=request['rate'],force_tokens=['\n'],chunk_end_tokens=['.','\n'],return_word_label=True)
json.dump(dict(compressed_prompt=result['compressed_prompt']),sys.stdout)
