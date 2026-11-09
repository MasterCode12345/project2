"""Convert a multilingual translation model (NLLB-200, M2M100) to CTranslate2.

One-time step per model; transformers/torch are needed only here, in a
throwaway environment, never as project dependencies::

    uv run --with transformers --with torch --with sentencepiece --with accelerate \\
        python scripts/convert_mt_model.py nllb-600m m2m100-418m

Downloads the repo at the revision pinned in
``audio_translator.translate.MT_MODELS`` (only the files the conversion needs:
the PyTorch weights, configs and tokenizer files; ~2.5 GB for NLLB-600M,
~1.9 GB for M2M100-418M), converts it with int8 weights and writes
``~/.cache/audio-translator/models/<name>-ct2/`` (the only place the app
looks) with ``SOURCE.json`` naming the exact source. The output appears only
when complete (written to a temp dir, then renamed). Existing output is kept
unless ``--force``. Converting needs a few GB of free RAM (about the size of the
checkpoint, with ``low_cpu_mem_usage``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_translator.translate import MT_MODELS, MT_MODELS_DIR, SPM_FILE, mt_model_dir  # noqa: E402

ALLOW_PATTERNS = ["*.json", "pytorch_model.bin", SPM_FILE]  # not rust_model.ot / README / .gitattributes
COPY_FILES = [SPM_FILE]


def convert(name: str, out_root: Path, *, force: bool = False, quantization: str = "int8", log=print) -> Path:
    repo, revision, _family = MT_MODELS[name]
    out = mt_model_dir(name, out_root)
    if (out / "model.bin").is_file() and not force:
        log(f"{out} already exists (use --force to convert again)")
        return out

    from ctranslate2.converters import TransformersConverter
    from huggingface_hub import snapshot_download

    log(f"downloading {repo}@{revision[:8]} …")
    src = snapshot_download(repo, revision=revision, allow_patterns=ALLOW_PATTERNS)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=out.parent))
    try:
        log(f"converting to CTranslate2 ({quantization}) …")
        # low_cpu_mem_usage: load the checkpoint once instead of twice (peak RAM ~ model size, 8 GB machine)
        TransformersConverter(src, copy_files=COPY_FILES, low_cpu_mem_usage=True).convert(
            str(tmp), quantization=quantization, force=True)
        (tmp / "SOURCE.json").write_text(json.dumps(
            {"repo": repo, "revision": revision, "quantization": quantization,
             "converter": "ctranslate2.converters.TransformersConverter"}, indent=2) + "\n")
        missing = [f for f in ["model.bin", *COPY_FILES] if not (tmp / f).is_file()]
        if missing:
            raise RuntimeError(f"conversion output is missing {missing}")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    log(f"wrote {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("names", nargs="+", choices=list(MT_MODELS), help="model(s) to convert")
    p.add_argument("--quantization", default="int8", help="CTranslate2 weight type (default: int8)")
    p.add_argument("--force", action="store_true", help="convert again even if the output exists")
    args = p.parse_args(argv)
    try:
        for name in args.names:
            convert(name, MT_MODELS_DIR, force=args.force, quantization=args.quantization)
    except ImportError as exc:
        print(f"error: {exc}\nrun with: uv run --with transformers --with torch --with sentencepiece --with accelerate python "
              f"scripts/convert_mt_model.py {' '.join(args.names)}", file=sys.stderr)
        return 2
    except Exception as exc:  # network, conversion
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
