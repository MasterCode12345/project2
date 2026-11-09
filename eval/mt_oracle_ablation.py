"""Which part of casing/punctuation does translation need? Oracle variants of the source.

Each FLEURS source sentence is translated in five forms, from what ASR gives
to the reference:

* ``norm``: lowercase, no punctuation (``text_norm`` in mt_bench);
* ``sent``: ``norm`` + capital first letter + final period (``restore.sentence_case``);
* ``case+sent``: the reference casing (names...), no punctuation, + final period;
* ``punct+sent``: lowercase with the reference punctuation, capital first letter;
* ``full``: the reference sentence (``text``).

``sent -> case+sent`` is what restoring names' casing could bring at best and
``sent -> punct+sent`` what restoring punctuation could; mt_bench measures what
a real restorer gets. Same data, scores and backends as ``eval/mt_bench.py``::

    uv run --group eval python eval/mt_oracle_ablation.py --fleurs-src-dir ~/data/fleurs_vi \\
        --fleurs-tgt-tsv ~/data/fleurs_en/test.tsv --backends argos,nllb-600m \\
        --out results/mt_vi_en_oracle.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mt_bench as mt  # noqa: E402

from audio_translator.languages import normalize  # noqa: E402
from audio_translator.restore import sentence_case  # noqa: E402

COLUMNS = ["backend", "variant", "n_sents", "bleu", "chrf", "bleu_lc", "chrf_lc"]


def strip_punct_keep_case(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join("".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text).split())


def variants(lang: str) -> dict:
    return {
        "norm": lambda s: normalize(s, lang),
        "sent": lambda s: sentence_case(normalize(s, lang)),
        "case+sent": lambda s: sentence_case(strip_punct_keep_case(s)),
        "punct+sent": lambda s: sentence_case(unicodedata.normalize("NFC", s).lower()),
        "full": lambda s: s,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", default="vi")
    p.add_argument("--tgt", default="en")
    p.add_argument("--fleurs-src-dir", type=Path, required=True)
    p.add_argument("--fleurs-tgt-tsv", type=Path, required=True)
    p.add_argument("--backends", default="argos,nllb-600m", help="argos and/or converted MT_MODELS names")
    p.add_argument("--out", type=Path, default=Path("results/mt_oracle.csv"))
    args = p.parse_args(argv)
    try:
        pairs = mt.load_fleurs_pairs(args.fleurs_src_dir.expanduser(), args.fleurs_tgt_tsv.expanduser())
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    backends = [mt.argos_backend([mt.default_argos_package()]) if b == "argos" else mt.ct2_model_backend(b)
                for b in mt._csv_list(args.backends)]
    refs = [x.ref for x in pairs]
    scorer = mt.Scorer(args.tgt)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for b in backends:
            tr = b.make(args.src, args.tgt)
            tr.translate(pairs[0].src)  # warm-up
            for name, fn in variants(args.src).items():
                hyps, _secs, _failed = mt.translate_all(tr, [fn(x.src) for x in pairs])
                s = scorer.score(hyps, refs)
                row = {"backend": b.name, "variant": name, "n_sents": s["n"],
                       **{k: f"{s[k]:.2f}" for k in ("bleu", "chrf", "bleu_lc", "chrf_lc")}}
                w.writerow(row)
                f.flush()
                print("  ".join(f"{k}={row[k]}" for k in COLUMNS), flush=True)
    print(scorer.signatures())
    return 0


if __name__ == "__main__":
    sys.exit(main())
