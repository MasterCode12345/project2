"""Convert the Vietnamese casing/punctuation restorer (dragonSwing/vibert-capu) to CTranslate2.

One-time step; transformers/torch are needed only here, in a throwaway
environment, never as project dependencies::

    uv run --with "transformers<5" --with torch python scripts/convert_capu_model.py vibert-capu

Downloads, at the revisions pinned in ``audio_translator.restore.RESTORE_MODELS``,
only ``pytorch_model.bin`` (~462 MB), ``config.json`` and ``vocabulary/labels.txt``
of the tagger and ``config.json`` + ``vocab.txt`` of its base model
(FPTAI/vibert-base-cased). No code from either repo is run: the checkpoint is
read with ``torch.load(weights_only=True)`` into a plain ``transformers.BertModel``
(embeddings grown by one row for the ``$START`` token, as the tagger was
trained). The encoder is converted to CTranslate2 (int8); the label head is
saved as ``heads.npz``. Output: ``~/.cache/audio-translator/models/vibert-capu-ct2/``
with ``SOURCE.json``, written to a temp dir and renamed when complete.

Before the rename, the converted model is checked against the PyTorch model
(fp32) on a few sentences: the WordPiece tokens of ``restore.WordPiece`` must
equal the Hugging Face tokenizer's, and the predicted labels must agree on at
least ``--min-agreement`` of the positions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_translator.restore import (  # noqa: E402
    HEADS_FILE, LABELS_FILE, RESTORE_MODELS, START_TOKEN, VOCAB_FILE, CapuRestorer, restore_model_dir)
from audio_translator.translate import MT_MODELS_DIR  # noqa: E402

CHECK_SENTENCES = [  # lowercase, unpunctuated, as PhoWhisper writes
    "tôi là sinh viên năm ba của đại học bách khoa hà nội",
    "hôm nay trời đẹp chúng ta đi chơi nhé",
    "bạn có biết ga tàu ở đâu không",
    "theo đó thủ tướng dự kiến tiếp bộ trưởng nông nghiệp mỹ",
    "các nhà khoa học hi vọng sẽ hiểu được cách thức hình thành của các hành tinh đặc biệt là trái đất",
    "cuộc họp bắt đầu lúc ba giờ chiều và kết thúc lúc bốn giờ rưỡi",
    "công ty viettel vừa ra mắt mạng năm gờ ở hải phòng",
    "em gửi file slide cho thầy qua email rồi ạ",
]


def _torch_labels(model, tokenizer, w, b, labels, words_batch):
    """Reference: Hugging Face tokenizer + PyTorch BertModel (fp32), first sub-token of each word."""
    import torch

    out_tokens, out_labels = [], []
    for words in words_batch:
        enc = tokenizer([START_TOKEN, *words], is_split_into_words=True, add_special_tokens=False,
                        return_tensors="pt")
        ids = enc.word_ids(0)
        offs = [0] + [i for i in range(1, len(ids)) if ids[i] != ids[i - 1]]
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state[0].numpy()
        logits = hidden[offs] @ w.T + b
        out_tokens.append(tokenizer.convert_ids_to_tokens(enc["input_ids"][0]))
        out_labels.append([labels[i] for i in logits.argmax(-1)])
    return out_tokens, out_labels


def convert(name: str, out_root: Path, *, force: bool = False, quantization: str = "int8",
            min_agreement: float = 0.95, log=print) -> Path:
    repo, revision, base_repo, base_revision, langs = RESTORE_MODELS[name]
    out = restore_model_dir(name, out_root)
    if (out / "model.bin").is_file() and not force:
        log(f"{out} already exists (use --force to convert again)")
        return out

    import numpy as np
    import torch
    from ctranslate2.converters import TransformersConverter
    from huggingface_hub import snapshot_download
    from transformers import BertConfig, BertModel, BertTokenizerFast

    log(f"downloading {repo}@{revision[:8]} and {base_repo}@{base_revision[:8]} …")
    src = Path(snapshot_download(repo, revision=revision,
                                 allow_patterns=["pytorch_model.bin", "config.json", "vocabulary/labels.txt"]))
    base = Path(snapshot_download(base_repo, revision=base_revision, allow_patterns=["config.json", "vocab.txt"]))

    state = torch.load(src / "pytorch_model.bin", map_location="cpu", weights_only=True)
    cfg = BertConfig.from_pretrained(base)
    base_vocab = cfg.vocab_size  # resize_token_embeddings updates cfg in place
    bert = BertModel(cfg)
    bert.resize_token_embeddings(base_vocab + 1)  # + $START (special_tokens_fix in the tagger's config)
    enc_state = {k[len("bert."):]: v for k, v in state.items() if k.startswith("bert.")}
    enc_state.pop("embeddings.position_ids", None)  # a buffer older transformers saved; recomputed
    missing, unexpected = bert.load_state_dict(enc_state, strict=False)
    if unexpected or any(not k.startswith("pooler.") for k in missing):
        raise RuntimeError(f"checkpoint does not match BertModel: missing {missing}, unexpected {unexpected}")
    bert.eval()
    w = state["classifier.weight"].float().numpy()
    b = state["classifier.bias"].float().numpy()
    labels = (src / "vocabulary" / "labels.txt").read_text(encoding="utf-8").split("\n")[:w.shape[0]]
    tokenizer = BertTokenizerFast(vocab_file=str(base / "vocab.txt"), do_lower_case=False)
    tokenizer.add_tokens([START_TOKEN])
    if tokenizer.convert_tokens_to_ids(START_TOKEN) != base_vocab:
        raise RuntimeError("unexpected $START id")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=out.parent))
    hf_dir = Path(tempfile.mkdtemp(prefix=f".{name}-hf-", dir=out.parent))
    try:
        bert.save_pretrained(hf_dir)
        tokenizer.save_pretrained(hf_dir)
        log(f"converting the encoder to CTranslate2 ({quantization}) …")
        TransformersConverter(str(hf_dir)).convert(str(tmp), quantization=quantization, force=True)
        np.savez(tmp / HEADS_FILE, classifier_weight=w, classifier_bias=b)
        (tmp / LABELS_FILE).write_text("\n".join(labels) + "\n", encoding="utf-8")
        shutil.copyfile(base / "vocab.txt", tmp / VOCAB_FILE)
        (tmp / "SOURCE.json").write_text(json.dumps(
            {"repo": repo, "revision": revision, "base": base_repo, "base_revision": base_revision,
             "license": "cc-by-sa-4.0", "quantization": quantization,
             "converter": "ctranslate2.converters.TransformersConverter (encoder) + numpy label head"},
            indent=2) + "\n")

        log("checking the converted model against PyTorch …")
        words_batch = [s.split() for s in CHECK_SENTENCES]
        ref_tokens, ref_labels = _torch_labels(bert, tokenizer, w, b, labels, words_batch)
        r = CapuRestorer(name, root=tmp.parent)
        r.model_dir = tmp
        r.load()
        for words, toks in zip(words_batch, ref_tokens):
            mine = [p for x in [START_TOKEN, *words] for p in r._wp.word(x)]
            if mine != toks:
                raise RuntimeError(f"WordPiece mismatch for {' '.join(words)!r}: {mine} != {toks}")
        got = r._label_batch(words_batch)
        same = sum(x == y for g, f in zip(got, ref_labels) for x, y in zip(g, f))
        total = sum(len(f) for f in ref_labels)
        log(f"labels: {same}/{total} positions agree with PyTorch fp32")
        for s in CHECK_SENTENCES[:4]:
            log(f"  {s!r} -> {r.restore(s, langs[0])!r}")
        if same < min_agreement * total:
            raise RuntimeError(f"label agreement {same / total:.3f} below {min_agreement}")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(hf_dir, ignore_errors=True)
    log(f"wrote {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("names", nargs="+", choices=list(RESTORE_MODELS), help="model(s) to convert")
    p.add_argument("--quantization", default="int8", help="CTranslate2 weight type (default: int8)")
    p.add_argument("--min-agreement", type=float, default=0.95,
                   help="fraction of label positions that must match PyTorch (default 0.95)")
    p.add_argument("--force", action="store_true", help="convert again even if the output exists")
    args = p.parse_args(argv)
    try:
        for name in args.names:
            convert(name, MT_MODELS_DIR, force=args.force, quantization=args.quantization,
                    min_agreement=args.min_agreement)
    except ImportError as exc:
        print(f"error: {exc}\nrun with: uv run --with \"transformers<5\" --with torch python "
              f"scripts/convert_capu_model.py {' '.join(args.names)}", file=sys.stderr)
        return 2
    except Exception as exc:  # network, conversion, check
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
