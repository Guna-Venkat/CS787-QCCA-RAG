import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import datasets

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.data_utils import get_full_text_qasper, extract_answer_and_question_type_qasper, rank_documents, write_jsonl

def build_qasper_split(split: str, output_path: str, chunk_size: int = 256):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Loading allenai/qasper split '{split}'...")
    ds = datasets.load_dataset('allenai/qasper', split=split)
    rows = []
    
    for index_row, row in enumerate(tqdm(ds, desc=f"Processing QASPER {split}")):
        full_text = get_full_text_qasper(row)
        for index_question, question in enumerate(row['qas']["question"]):
            answers_list = row['qas']['answers'][index_question]['answer']
            if not answers_list:
                continue
            answer, question_type, answerable = extract_answer_and_question_type_qasper(answers_list[0])
            
            # Context ranking via BM25
            context = rank_documents(documents=full_text, question=question, chunk_size=chunk_size)
            
            if answerable and answer:
                # Format answer as list if string to match expected evaluation schemas
                answer_formatted = [answer] if isinstance(answer, str) else answer
                rows.append({
                    "id": f"{len(rows)}",
                    "example_id": f"{index_row}",
                    "question": question,
                    "context": context,
                    "answer": answer_formatted,
                    "answer_reformatted": answer_formatted,
                    "choices": None,
                    "question_type": question_type,
                })
                
    write_jsonl(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")

if __name__ == "__main__":
    build_qasper_split("train", "data/reformatted/QASPER_train.jsonl")
    build_qasper_split("test", "data/reformatted/QASPER_test.jsonl")
