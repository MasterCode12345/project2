"""Convert VinAI PhoWhisper (transformers) to CTranslate2 for faster-whisper.

One-time step per model size; transformers/torch are needed only here, in a
throwaway environment, never as project dependencies::

    uv run --with transformers --with torch python scripts/convert_phowhisper.py small
    uv run --with transformers --with torch python scripts/convert_phowhisper.py medium

Downloads ``vinai/PhoWhisper-<size>`` at the revision pinned in
``audio_translator.asr.LOCAL_MODELS``, converts it (float16 weights; the app
loads them as int8 on CPU) and writes ``~/.cache/audio-translator/models/phowhisper-<size>-ct2/``
(the only place the app looks) with a
``SOURCE.json`` naming the exact source. The output appears only when
complete (written to a temp dir, then renamed). Existing output is kept
unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_translator.asr import LOCAL_MODELS, LOCAL_MODELS_DIR, local_model_dir  # noqa: E402

COPY_FILES = ["tokenizer.json", "preprocessor_config.json"]
SIZES = sorted(n.removeprefix("phowhisper-") for n in LOCAL_MODELS)


def convert(size: str, out_root: Path, *, force: bool = False, quantization: str = "float16",
            log=print) -> Path:
    name = f"phowhisper-{size}"
    repo, revision = LOCAL_MODELS[name]
    out = local_model_dir(name, out_root)
    if (out / "model.bin").is_file() and not force:
        log(f"{out} already exists (use --force to convert again)")
        return out

    from ctranslate2.converters import TransformersConverter
    from huggingface_hub import snapshot_download

    log(f"downloading {repo}@{revision[:8]} …")
    src = snapshot_download(repo, revision=revision)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=out.parent))
    try:
        log(f"converting to CTranslate2 ({quantization}) …")
        TransformersConverter(src, copy_files=COPY_FILES).convert(str(tmp), quantization=quantization,
                                                                  force=True)
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
    p.add_argument("sizes", nargs="+", choices=SIZES, help="PhoWhisper size(s) to convert")
    p.add_argument("--quantization", default="float16", help="CTranslate2 weight type (default: float16)")
    p.add_argument("--force", action="store_true", help="convert again even if the output exists")
    args = p.parse_args(argv)
    try:
        for size in args.sizes:
            convert(size, LOCAL_MODELS_DIR, force=args.force, quantization=args.quantization)
    except ImportError as exc:
        print(f"error: {exc}\nrun with: uv run --with transformers --with torch python "
              f"scripts/convert_phowhisper.py {' '.join(args.sizes)}", file=sys.stderr)
        return 2
    except Exception as exc:  # network, conversion
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
