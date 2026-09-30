"""Verify notebook structure, execution status, and code syntax."""

import ast
import json
from pathlib import Path

def main():
    nb_path = Path("notebooks/week4/week4_feature_validation.ipynb")
    assert nb_path.exists(), f"Notebook {nb_path} does not exist!"
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    print(f"Loaded notebook {nb_path}")
    print(f"Total cells: {len(nb['cells'])}")
    
    code_cells = 0
    markdown_cells = 0
    
    for i, cell in enumerate(nb["cells"]):
        ctype = cell["cell_type"]
        if ctype == "markdown":
            markdown_cells += 1
        elif ctype == "code":
            code_cells += 1
            assert cell["execution_count"] is None, f"Cell {i} has non-None execution_count: {cell['execution_count']}"
            assert cell["outputs"] == [], f"Cell {i} has non-empty outputs!"
            
            # Check Python syntax
            source = "".join(cell["source"])
            # Remove ipython magics before ast parse
            clean_source = "\n".join([line for line in source.splitlines() if not line.strip().startswith("%")])
            try:
                ast.parse(clean_source)
            except SyntaxError as e:
                print(f"Syntax error in code cell {i}: {e}")
                raise e
        else:
            raise ValueError(f"Unknown cell type: {ctype}")
            
    print(f"Verification PASSED:")
    print(f"  Markdown cells: {markdown_cells}")
    print(f"  Code cells:     {code_cells}")
    print(f"  All code cells have execution_count=None and outputs=[]")
    print(f"  All Python code cells are syntactically valid!")

if __name__ == "__main__":
    main()
