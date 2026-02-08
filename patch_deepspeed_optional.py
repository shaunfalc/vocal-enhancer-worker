#!/usr/bin/env python3
"""Make deepspeed import optional in resemble_enhance enhancer/train.py for inference-only worker."""
import re
import sys
from pathlib import Path

# Discover train.py: may be at /app/resemble_enhance/enhancer/train.py or under cwd
root = Path("/app")
if not (root / "resemble_enhance").exists():
    root = Path(__file__).resolve().parent
train_py = root / "resemble_enhance" / "enhancer" / "train.py"

if not train_py.exists():
    print("patch failed: target file does not exist:", train_py, file=sys.stderr)
    sys.exit(1)

text = train_py.read_text()

# Match exact current upstream, or common variants (whitespace / alternate import)
old_patterns = [
    "from deepspeed import DeepSpeedConfig",
    "from deepspeed import DeepSpeedConfig ",  # trailing space
]
replacement = """try:
    from deepspeed import DeepSpeedConfig
except ModuleNotFoundError:
    DeepSpeedConfig = None  # inference-only; load_G/load_D are unused"""

patched = False
for old in old_patterns:
    if old in text:
        text = text.replace(old, replacement, 1)
        patched = True
        break

if not patched:
    # Fallback: regex for "from deepspeed import ..." so we tolerate minor changes
    match = re.search(r"^from deepspeed import \w+\s*$", text, re.MULTILINE)
    if match:
        text = text[: match.start()] + replacement + text[match.end() :]
        patched = True

if not patched:
    print("patch failed: no matching 'from deepspeed import' line found in", train_py, file=sys.stderr)
    sys.exit(1)

train_py.write_text(text)

# Verify the file now has the optional import
if "ModuleNotFoundError" not in train_py.read_text():
    print("patch verify failed: optional import not present after patch", file=sys.stderr)
    sys.exit(1)

print("Patched", train_py, ": deepspeed import is now optional")
