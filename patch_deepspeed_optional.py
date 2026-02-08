#!/usr/bin/env python3
"""Make deepspeed import optional in resemble_enhance enhancer/train.py for inference-only worker."""
import sys
from pathlib import Path

path = Path("/app/resemble_enhance/enhancer/train.py")
if not path.exists():
    print("patch failed: target file does not exist:", path, file=sys.stderr)
    sys.exit(1)
text = path.read_text()

old = "from deepspeed import DeepSpeedConfig"
new = """try:
    from deepspeed import DeepSpeedConfig
except ModuleNotFoundError:
    DeepSpeedConfig = None  # inference-only; load_G/load_D are unused"""

if old not in text:
    raise SystemExit("patch failed: expected import not found")
path.write_text(text.replace(old, new, 1))
print("Patched train.py: deepspeed import is now optional")
