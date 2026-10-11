"""Script to parse all outputs from week5_content_aware_allocator.ipynb."""

import json
from pathlib import Path

REPO_ROOT = Path("/home/gunavenkat/Downloads/CS787-RAG-Project")
NOTEBOOK_PATH = REPO_ROOT / "notebooks/week5/week5_content_aware_allocator.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        exec_count = cell.get("execution_count")
        outputs = cell.get("outputs", [])
        print(f"\n==================== CELL {idx} (Exec #{exec_count}) ====================")
        first_line = "".join(cell["source"]).split("\n")[0]
        print(f"Code: {first_line[:80]}...")
        for out in outputs:
            output_type = out.get("output_type")
            if output_type == "stream":
                print(f"[STREAM {out.get('name')}]:")
                print("".join(out.get("text", [])))
            elif output_type in ["display_data", "execute_result"]:
                data = out.get("data", {})
                if "text/plain" in data:
                    print(f"[{output_type} text/plain]:")
                    print("".join(data["text/plain"]))
                if "image/png" in data:
                    print(f"[{output_type} image/png (base64 length: {len(data['image/png'])})]")
            elif output_type == "error":
                print(f"[ERROR]: {out.get('ename')}: {out.get('evalue')}")
