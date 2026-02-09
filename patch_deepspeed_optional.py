#!/usr/bin/env python3
"""Make deepspeed optional in resemble_enhance for inference-only worker.

Patches the cloned repo in nine areas:
- enhancer/train.py: optional DeepSpeedConfig
- denoiser/train.py: optional DeepSpeedConfig (loaded when enhancer imports load_denoiser)
- utils/distributed.py: optional deepspeed + stub get_accelerator, init_distributed no-op when missing
- utils/engine.py: optional deepspeed, Engine=None and no-op init_distributed when missing
- enhancer/inference.py: load state with map_location=device, weights_only=True; Enhancer(hp) -> Enhancer(hp, device=device)
- enhancer/enhancer.py: __init__(hp, device="cpu"), load_denoiser(..., device) instead of "cpu"
- denoiser/inference.py: load_denoiser uses map_location=device, weights_only=True
- inference.py: inference_chunk computes abs_max after dwav.to(device) so normalize stays on same device
"""
import re
import sys
from pathlib import Path

# Discover resemble_enhance root: /app when in Docker, else directory containing this script
root = Path("/app")
if not (root / "resemble_enhance").exists():
    root = Path(__file__).resolve().parent
base = root / "resemble_enhance"


def patch_train_py():
    """Make DeepSpeedConfig import optional in enhancer/train.py."""
    path = base / "enhancer" / "train.py"
    if not _patch_deepspeed_import_in(path):
        return False
    print("Patched", path, ": deepspeed import optional")
    return True


def _patch_deepspeed_import_in(path):
    """Make 'from deepspeed import DeepSpeedConfig' optional in the given file."""
    if not path.exists():
        print("patch failed: target file does not exist:", path, file=sys.stderr)
        return False
    text = path.read_text()
    replacement = """try:
    from deepspeed import DeepSpeedConfig
except ModuleNotFoundError:
    DeepSpeedConfig = None  # inference-only; load_G/load_D are unused"""
    for old in ["from deepspeed import DeepSpeedConfig", "from deepspeed import DeepSpeedConfig "]:
        if old in text:
            text = text.replace(old, replacement, 1)
            break
    else:
        match = re.search(r"^from deepspeed import \w+\s*$", text, re.MULTILINE)
        if match:
            text = text[: match.start()] + replacement + text[match.end() :]
        else:
            print("patch failed: no matching 'from deepspeed import' in", path, file=sys.stderr)
            return False
    path.write_text(text)
    if "ModuleNotFoundError" not in path.read_text():
        print("patch verify failed:", path, file=sys.stderr)
        return False
    return True


def patch_denoiser_train_py():
    """Make DeepSpeedConfig import optional in denoiser/train.py (loaded via enhancer -> load_denoiser)."""
    path = base / "denoiser" / "train.py"
    if not _patch_deepspeed_import_in(path):
        return False
    print("Patched", path, ": deepspeed import optional")
    return True


def patch_distributed_py():
    """Make deepspeed optional in utils/distributed.py; stub get_accelerator, no-op init when missing."""
    path = base / "utils" / "distributed.py"
    if not path.exists():
        print("patch failed: target file does not exist:", path, file=sys.stderr)
        return False
    text = path.read_text()

    # Replace top-level deepspeed imports with try/except and stub
    old_imports = (
        "import deepspeed\n"
        "import torch\n"
        "from deepspeed.accelerator import get_accelerator\n"
        "from torch.distributed import broadcast_object_list"
    )
    new_imports = """try:
    import deepspeed
    from deepspeed.accelerator import get_accelerator
except ModuleNotFoundError:
    deepspeed = None
    def get_accelerator():
        return type("_Stub", (), {"communication_backend_name": lambda self: "nccl"})()
import torch
from torch.distributed import broadcast_object_list"""
    if old_imports not in text:
        print("patch failed: distributed.py expected import block not found", file=sys.stderr)
        return False
    text = text.replace(old_imports, new_imports, 1)

    # Make init_distributed skip deepspeed when deepspeed is None (accept any body indent: 1, 2, or 4 spaces)
    init_distributed_re = re.compile(
        r"(def init_distributed\(\):\n)"
        r"([ \t]+)(fix_unset_envs\(\)\n)"
        r"\2(deepspeed\.init_distributed\(get_accelerator\(\)\.communication_backend_name\(\)\)\n)"
        r"\2(torch\.cuda\.set_device\(local_rank\(\)\))",
        re.MULTILINE,
    )
    match = init_distributed_re.search(text)
    if not match:
        print("patch failed: distributed.py init_distributed block not found", file=sys.stderr)
        return False
    indent = match.group(2)
    inner = "    "  # 4 spaces for block under "if deepspeed is not None"
    text = (
        text[: match.start()]
        + match.group(1)
        + indent
        + match.group(3)
        + indent
        + "if deepspeed is not None:\n"
        + indent
        + inner
        + match.group(4)
        + indent
        + inner
        + match.group(5)
        + "\n"
        + text[match.end() :]
    )

    path.write_text(text)
    print("Patched", path, ": deepspeed optional, init_distributed no-op when missing")
    return True


def patch_engine_py():
    """Make deepspeed optional in utils/engine.py; Engine=None and no-op init_distributed when missing."""
    path = base / "utils" / "engine.py"
    if not path.exists():
        print("patch failed: target file does not exist:", path, file=sys.stderr)
        return False
    text = path.read_text()

    # Replace deepspeed imports with try/except; when missing, define Engine=None and no-op init_distributed
    old_block = """import deepspeed
import pandas as pd
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.utils import clip_grad_norm_
from torch import nn"""
    new_block = """try:
    import deepspeed
    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.engine import DeepSpeedEngine
    from deepspeed.runtime.utils import clip_grad_norm_
except ModuleNotFoundError:
    deepspeed = None
    DeepSpeedEngine = None
    get_accelerator = None
    clip_grad_norm_ = None
import pandas as pd
from torch import nn"""
    if old_block not in text:
        print("patch failed: engine.py expected import block not found", file=sys.stderr)
        return False
    text = text.replace(old_block, new_block, 1)

    # Make init_distributed no-op when deepspeed is None (accept any body indent)
    init_distributed_re = re.compile(
        r"(@cache\ndef init_distributed\(\):\n)"
        r"([ \t]+)(update_deepspeed_logger\(\)\n)"
        r"\2(fix_unset_envs\(\)\n)"
        r"\2(deepspeed\.init_distributed\(get_accelerator\(\)\.communication_backend_name\(\)\))",
        re.MULTILINE,
    )
    match = init_distributed_re.search(text)
    if not match:
        print("patch failed: engine.py init_distributed block not found", file=sys.stderr)
        return False
    indent = match.group(2)
    inner = "    "
    text = (
        text[: match.start()]
        + match.group(1)
        + indent
        + match.group(3)
        + indent
        + match.group(4)
        + indent
        + "if deepspeed is not None:\n"
        + indent
        + inner
        + match.group(5)
        + text[match.end() :]
    )

    # Wrap class Engine(DeepSpeedEngine): ... in try/except so we can set Engine=None when deepspeed missing
    # We need to replace "class Engine(DeepSpeedEngine):" with conditional: only define class when deepspeed present
    # Easiest: replace the whole class definition with a try block that defines the class, except set Engine = None
    class_start = "class Engine(DeepSpeedEngine):"
    if class_start not in text:
        print("patch failed: engine.py class Engine not found", file=sys.stderr)
        return False
    # Find the class and wrap it: before class, add "try:\n    " and indent the class body; after class (at next top-level def or end), add except block
    # Simpler approach: replace "class Engine(DeepSpeedEngine):" with "class Engine(DeepSpeedEngine if deepspeed is not None else object):" and in __init__ guard the super and init_distributed. But then load_checkpoint etc. would still call super() - complex.
    # Alternative: define Engine only when deepspeed is not None. So we need to indent the entire Engine class and put it under "if deepspeed is not None:" and add "else: Engine = None". That way the class is only defined when deepspeed exists.
    idx = text.find(class_start)
    if idx == -1:
        return False
    # Insert "if deepspeed is not None:\n    " before the class and indent the class body by 4 spaces until we hit a line at the same indent as "class Engine"
    # and add "else:\n    Engine = None"
    # This is error-prone. Better: in engine.py we already made the imports optional. So now DeepSpeedEngine can be None. So "class Engine(DeepSpeedEngine):" will fail when DeepSpeedEngine is None. So we must not define the class at all when deepspeed is None. So we need to wrap the entire class in "if deepspeed is not None:" and add "else: Engine = None". The class ends when we have a line that's not indented more than the class (next top-level). So find the span of the class (from "class Engine" to the last method's end). In the repo the class has methods that are indented with 1 space (from the fetch). So the class body is everything from the line after "class Engine(DeepSpeedEngine):" until we see a line that starts with 0 spaces (excluding blank lines). So we can find the end of the class by looking for the next line that matches ^[a-zA-Z@] (top-level) or end of file.
    # Simpler: use a multi-line replacement. The Engine class in the fetch goes from "class Engine(DeepSpeedEngine):" to " )\n" (end of load_checkpoint's return _try_each(...)). So the class is a big block. Let me try wrapping by replacing "class Engine(DeepSpeedEngine):" with "if deepspeed is not None:\n    class _Engine(DeepSpeedEngine):" and then at the end of the class (before "\n\n" or next @cache/def) add "\n    Engine = _Engine\nelse:\n    Engine = None". But the "end of the class" is hard to find. Let me try a different approach: in the except ModuleNotFoundError block we already set DeepSpeedEngine = None. So we need to make the class definition conditional: only run "class Engine(DeepSpeedEngine):" when DeepSpeedEngine is not None. So we could replace "class Engine(DeepSpeedEngine):" with "if DeepSpeedEngine is not None:\n    class Engine(DeepSpeedEngine):" but then the class body would need to be indented and we'd need "else:\n    Engine = None". So we need to indent the entire class body. The class body in the file is from the line after "class Engine(DeepSpeedEngine):" until the next top-level (e.g. blank line + something at column 0). So get the slice, indent it by 4 spaces, and wrap.
    lines = text.split("\n")
    class_lineno = None
    for i, line in enumerate(lines):
        if line.strip().startswith("class Engine(DeepSpeedEngine):"):
            class_lineno = i
            break
    if class_lineno is None:
        print("patch failed: engine.py class Engine line not found", file=sys.stderr)
        return False
    # Find end of class: next line that has indent 0 (or same as "class") and is not blank
    class_indent = len(lines[class_lineno]) - len(lines[class_lineno].lstrip())
    end_lineno = class_lineno + 1
    while end_lineno < len(lines):
        line = lines[end_lineno]
        if line.strip() and (len(line) - len(line.lstrip())) <= class_indent:
            break
        end_lineno += 1
    # Class is lines [class_lineno, end_lineno)
    before = "\n".join(lines[:class_lineno])
    after = "\n".join(lines[end_lineno:])
    first_class_line = lines[class_lineno]
    rest_of_class = lines[class_lineno + 1 : end_lineno]
    # Indent rest_of_class by 4 more spaces
    rest_indented = []
    for ln in rest_of_class:
        if ln.strip():
            rest_indented.append("    " + ln)
        else:
            rest_indented.append(ln)
    new_class_block = "if deepspeed is not None:\n    " + first_class_line + "\n" + "\n".join(rest_indented) + "\nelse:\n    Engine = None\n"
    text = before + "\n" + new_class_block + "\n" + after
    path.write_text(text)
    print("Patched", path, ": Engine defined only when deepspeed present")
    return True


def patch_enhancer_denoiser_device():
    """Make Enhancer pass device to load_denoiser and load_enhancer pass device to Enhancer."""
    # 1. enhancer/enhancer.py: add device param to __init__, use it in load_denoiser (allow any indent)
    path_enh = base / "enhancer" / "enhancer.py"
    if not path_enh.exists():
        print("patch failed: enhancer/enhancer.py not found", file=sys.stderr)
        return False
    text = path_enh.read_text()
    if re.search(r'def __init__\(self, hp: HParams, device="cpu"\):', text):
        pass  # already patched
    elif re.search(r'def __init__\(self, hp: HParams\):', text):
        text = re.sub(r'def __init__\(self, hp: HParams\):', 'def __init__(self, hp: HParams, device="cpu"):', text, count=1)
    else:
        print("patch failed: enhancer/enhancer.py __init__ signature not found", file=sys.stderr)
        return False
    if 'load_denoiser(self.hp.denoiser_run_dir, device)' in text:
        pass  # already patched
    elif re.search(r'load_denoiser\(self\.hp\.denoiser_run_dir,\s*["\']cpu["\']\)', text):
        text = re.sub(r'load_denoiser\(self\.hp\.denoiser_run_dir,\s*["\']cpu["\']\)', "load_denoiser(self.hp.denoiser_run_dir, device)", text, count=1)
    else:
        print("patch failed: enhancer/enhancer.py load_denoiser(..., \"cpu\") not found", file=sys.stderr)
        return False
    path_enh.write_text(text)
    print("Patched", path_enh, ": Enhancer __init__ accepts device, passes to load_denoiser")

    # 2. enhancer/inference.py: Enhancer(hp) -> Enhancer(hp, device=device)
    path_inf = base / "enhancer" / "inference.py"
    if not path_inf.exists():
        print("patch failed: enhancer/inference.py not found", file=sys.stderr)
        return False
    text = path_inf.read_text()
    if "Enhancer(hp, device=device)" in text:
        pass  # already patched
    elif re.search(r'\bEnhancer\s*\(\s*hp\s*\)', text):
        text = re.sub(r'\bEnhancer\s*\(\s*hp\s*\)', "Enhancer(hp, device=device)", text, count=1)
    else:
        print("patch failed: enhancer/inference.py Enhancer(hp) not found", file=sys.stderr)
        return False
    path_inf.write_text(text)
    print("Patched", path_inf, ": load_enhancer passes device to Enhancer")
    return True


def patch_denoiser_inference_map_location():
    """Load denoiser state onto target device (map_location=device)."""
    path = base / "denoiser" / "inference.py"
    if not path.exists():
        print("patch failed: denoiser/inference.py not found", file=sys.stderr)
        return False
    text = path.read_text()
    # Support both "cpu" and 'cpu' for robustness
    if re.search(r'map_location\s*=\s*["\']cpu["\']', text):
        text = re.sub(r'map_location\s*=\s*["\']cpu["\']', "map_location=device", text, count=1)
    else:
        print("patch failed: denoiser/inference.py torch.load map_location not found", file=sys.stderr)
        return False
    if "weights_only=True" not in text:
        text = re.sub(r'map_location=device\)', "map_location=device, weights_only=True)", text, count=1)
    path.write_text(text)
    print("Patched", path, ": load_denoiser uses map_location=device, weights_only=True")
    return True


def patch_inference_chunk_abs_max_device():
    """Compute abs_max after dwav.to(device) so normalize uses same device."""
    path = base / "inference.py"
    if not path.exists():
        print("patch failed: inference.py not found", file=sys.stderr)
        return False
    text = path.read_text()
    # Already patched if abs_max comes after dwav.to(device)
    if "dwav = dwav.to(device)\n abs_max = dwav.abs()" in text or "dwav = dwav.to(device)\n  abs_max = dwav.abs()" in text:
        path.write_text(text)
        print("Patched", path, ": inference_chunk abs_max on same device as dwav (already applied)")
        return True
    # Match block with flexible indentation (1 or 2 spaces; capture indent for replacement)
    # Pattern: indent length = ... \n indent abs_max = ... \n \n indent assert ... \n indent dwav = ... \n indent dwav = ... / abs_max
    old_pattern = re.compile(
        r"^([ \t]+)length = dwav\.shape\[-1\]\n"
        r"\1abs_max = dwav\.abs\(\)\.max\(\)\.clamp\(min=1e-7\)\n"
        r"\n+"  # blank line(s) between abs_max and assert
        r"\1assert dwav\.dim\(\) == 1, f\"Expected 1D waveform, got \{dwav\.dim\(\)\}D\"\n"
        r"\1dwav = dwav\.to\(device\)\n"
        r"\1dwav = dwav / abs_max( # Normalize)?",
        re.MULTILINE,
    )
    def repl(m):
        indent = m.group(1)
        comment = m.group(2) or " # Normalize"
        return (
            f"{indent}length = dwav.shape[-1]\n"
            f"{indent}assert dwav.dim() == 1, f\"Expected 1D waveform, got {{dwav.dim()}}D\"\n"
            f"{indent}dwav = dwav.to(device)\n"
            f"{indent}abs_max = dwav.abs().max().clamp(min=1e-7)\n"
            f"{indent}dwav = dwav / abs_max{comment}"
        )
    new_text, n = old_pattern.subn(repl, text, count=1)
    if n == 0:
        print("patch failed: inference.py inference_chunk block not found", file=sys.stderr)
        return False
    path.write_text(new_text)
    print("Patched", path, ": inference_chunk abs_max on same device as dwav")
    return True


def patch_inference_chunk_unnormalize_device():
    """Use abs_max.cpu() when unnormalizing so hwav (on CPU after model(...).cpu()) matches device."""
    path = base / "inference.py"
    if not path.exists():
        print("patch failed: inference.py not found", file=sys.stderr)
        return False
    text = path.read_text()
    # hwav is on CPU (from model(...)[0].cpu()); abs_max is on device -> use abs_max.cpu()
    if "hwav * abs_max.cpu()" in text:
        print("Patched", path, ": inference_chunk unnormalize device (already applied)")
        return True
    if re.search(r"hwav = hwav \* abs_max(\s*\# Unnormalize)?", text):
        text = re.sub(
            r"hwav = hwav \* abs_max(\s*\# Unnormalize)?",
            r"hwav = hwav * abs_max.cpu()\1",
            text,
            count=1,
        )
    else:
        print("patch failed: inference.py hwav * abs_max unnormalize line not found", file=sys.stderr)
        return False
    path.write_text(text)
    print("Patched", path, ": inference_chunk unnormalize uses abs_max.cpu()")
    return True


def patch_enhancer_inference_map_location():
    """Load enhancer state onto target device (regex so \"cpu\" or 'cpu' both match)."""
    path = base / "enhancer" / "inference.py"
    if not path.exists():
        print("patch failed: enhancer/inference.py not found", file=sys.stderr)
        return False
    text = path.read_text()
    # Replace map_location="cpu" or map_location='cpu' with map_location=device
    if re.search(r'map_location\s*=\s*["\']cpu["\']', text):
        text = re.sub(r'map_location\s*=\s*["\']cpu["\']', "map_location=device", text, count=1)
    else:
        print("patch failed: enhancer/inference.py expected torch.load map_location not found", file=sys.stderr)
        return False
    if "weights_only=True" not in text:
        text = re.sub(r'map_location=device\)', "map_location=device, weights_only=True)", text, count=1)
    path.write_text(text)
    print("Patched", path, ": load state with map_location=device, weights_only=True")
    return True


def main():
    if not base.exists():
        print("patch failed: resemble_enhance not found at", base, file=sys.stderr)
        sys.exit(1)
    ok = (
        patch_train_py()
        and patch_denoiser_train_py()
        and patch_distributed_py()
        and patch_engine_py()
        and patch_enhancer_inference_map_location()
        and patch_enhancer_denoiser_device()
        and patch_denoiser_inference_map_location()
        and patch_inference_chunk_abs_max_device()
        and patch_inference_chunk_unnormalize_device()
    )
    if not ok:
        sys.exit(1)
    print("All patches applied.")


if __name__ == "__main__":
    main()
