import ast
import json
from pathlib import Path

nb_path = Path("notebooks/week7/week7d_cross_encoder_fidelity_router.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Checking syntax for {len(nb['cells'])} cells in {nb_path}...")
error_count = 0
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        # Filter IPython magics like display() or %
        clean_lines = []
        for line in code.split("\n"):
            if line.strip().startswith("%") or line.strip().startswith("!"):
                clean_lines.append("# " + line)
            else:
                clean_lines.append(line)
        clean_code = "\n".join(clean_lines)
        try:
            ast.parse(clean_code)
        except SyntaxError as e:
            print(f"Syntax error in cell {idx}: {e}")
            error_count += 1

if error_count == 0:
    print("ALL CODE CELLS VALID SYNTAX!")
else:
    print(f"Found {error_count} syntax errors.")
