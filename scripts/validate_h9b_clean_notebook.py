#!/usr/bin/env python3
"""Syntax and structural validation for the public clean replay notebook."""
import ast
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
notebook = json.loads(path.read_text(encoding="utf8"))
assert notebook["nbformat"] == 4
assert notebook["metadata"]["kernelspec"]["language"] == "python"
cells = notebook["cells"]
code_count = 0
for i, cell in enumerate(cells):
    assert cell["cell_type"] in ("markdown", "code")
    source = cell["source"]
    assert isinstance(source, (str, list))
    if isinstance(source, list):
        source = "".join(source)
    if cell["cell_type"] == "code":
        ast.parse(source, filename=f"{path.name}:cell{i}")
        code_count += 1
        assert cell["execution_count"] is None
        assert not cell["outputs"]
assert code_count >= 7
print("PUBLIC_CLEAN_NOTEBOOK_SYNTAX_PASS", len(cells), "cells", code_count, "code cells")
