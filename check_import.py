#!/usr/bin/env python3
"""Run at Docker build to verify resemble_enhance.enhancer.train imports without deepspeed."""
import sys
import traceback

try:
    from resemble_enhance.enhancer.train import Enhancer, HParams
    print("import OK")
except Exception:
    traceback.print_exc()
    sys.exit(1)
