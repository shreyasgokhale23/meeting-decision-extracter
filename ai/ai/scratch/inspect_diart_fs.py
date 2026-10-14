"""Inspect diart 0.9.2 API via filesystem only - no imports."""
import os, pathlib, sys

# Find diart installation path
import importlib.metadata as im
diart_dist = im.distribution("diart")
diart_path = None
for f in diart_dist.files:
    s = str(f)
    if s.endswith("diart/__init__.py"):
        diart_path = pathlib.Path(diart_dist.locate_file(f)).parent
        break

if not diart_path:
    # fallback: find via pip show path
    import subprocess
    result = subprocess.run(["pip", "show", "diart"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("Location:"):
            loc = line.split(":", 1)[1].strip()
            diart_path = pathlib.Path(loc) / "diart"
            break

print("diart path:", diart_path)
print()

# List all .py files
print("=== diart source files ===")
for root, dirs, files in os.walk(diart_path):
    dirs[:] = [d for d in dirs if not d.startswith("_")]
    for f in files:
        if f.endswith(".py"):
            rel = os.path.relpath(os.path.join(root, f), diart_path)
            print(" ", rel)
print()

# Read __init__.py to see what's exported
print("=== diart/__init__.py ===")
init = diart_path / "__init__.py"
if init.exists():
    print(init.read_text(encoding="utf-8", errors="replace"))
