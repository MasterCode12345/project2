"""Glossary bench (plan-glossary.md): does protecting proper names with placeholders work?

Per backend (Argos = the app's default, NLLB-600M, M2M100-418M, Google) and placeholder form
(x, zqx, bracket, under, inline), on the text of three sets:

* ``eval/data/mt_glossary_vi_en.tsv``: sentences with glossary terms (``expect``), generic
  sentences that must match nothing, and two common phrases to check (``check``);
* ``eval/data/mt_custom_vi_en.tsv`` and FLEURS vi->en (``--fleurs-*``, ``--limit`` with ``--seed``).

Columns: term accuracy before/after (expected English terms found verbatim, ignoring case),
placeholder survival (term sentences whose placeholders all came back), fallbacks, leaked
placeholders (must be 0), sentences without terms whose output is identical to the plain
translation, BLEU/chrF before/after per set, and mean / P95 translation ms before/after.
``--samples`` writes the before/after translation of every term sentence side by side.
A backend that fails (Google blocked, model not converted) is reported as not measurable.
Never downloads anything.

    uv run --group eval python eval/glossary_bench.py --fleurs-src-dir ~/data/fleurs_vi \\
        --fleurs-tgt-tsv ~/data/fleurs_en/test.tsv --limit 100 --out results/mt_glossary_vi_en.csv \\
        --samples results/mt_glossary_vi_en_samples.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mt_bench as mb  # noqa: E402

from audio_translator.glossary import FORMS, INLINE, Glossary, GlossaryTranslator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ALL_FORMS = [*FORMS, INLINE]
COLUMNS = ["backend", "form", "corpus", "n", "term_sents", "terms_total", "terms_ok_before", "terms_ok_after",
           "term_acc_before", "term_acc_after", "survived", "fallbacks", "leaks", "no_term_sents", "no_term_identical",
           "bleu_before", "bleu_after", "chrf_before", "chrf_after", "ms_mean_before", "ms_mean_after",
           "ms_p95_before", "ms_p95_after", "failed"]
LEAK = re.compile(r"(?<![\w-])X\d+(?!\w)|ZQX|\[\s*\d+\s*\]|__T\d+__|__", re.IGNORECASE)


def load_glossary_set(path: Path):
    rows = [r for r in csv.DictReader((ln for ln in path.read_text(encoding="utf-8").splitlines()
                                       if ln.strip() and not ln.startswith("#")), delimiter="\t",
                                      quoting=csv.QUOTE_NONE)]
    return [dict(r, expect=[x for x in (r.get("expect") or "").split(";") if x.strip()]) for r in rows]


def term_hits(hyp: str | None, expect: list[str]) -> int:
    if hyp is None:
        return 0
    low = hyp.casefold()
    return sum(e.casefold() in low for e in expect)


def timed(translator, sources, pause: float = 0.0):
    out, secs, failed = [], [], 0
    for s in sources:
        t0 = time.perf_counter()
        try:
            out.append(" ".join((translator.translate(s) or "").split()))
            secs.append(time.perf_counter() - t0)
        except Exception:
            out.append(None)
            failed += 1
        if pause:
            time.sleep(pause)
    return out, secs, failed


def ms(secs):
    if not secs:
        return "", ""
    a = np.array(secs) * 1000
    return f"{a.mean():.1f}", f"{np.percentile(a, 95):.1f}"


def bench_backend(name, translator, sets, gl, forms, scorer, pause, emit, samples, log, checks=None):
    """sets: [(corpus, sources, refs, expects)]. Plain translations once, then one pass per form."""
    plain = {}
    for corpus, src, refs, _exp in sets:
        plain[corpus] = timed(translator, src, pause)
        if plain[corpus][2] == len(src):
            log(f"{name}: every call failed on {corpus}: not measurable")
            return
    if checks:  # common phrases outside the glossary: right without help?
        outs = plain["glossary"][0]
        for i, want in checks:
            log(f"{name} phrase check: {want!r} {'OK' if outs[i] and want.casefold() in outs[i].casefold() else 'MISSING'}"
                f" <- {outs[i]!r}")
    for form in forms:
        wrapped = GlossaryTranslator(translator, gl, form)
        for corpus, src, refs, exp in sets:
            p_out, p_secs, p_fail = plain[corpus]
            g_out, g_secs, g_fail, used = [], [], 0, {}
            survived = fallbacks = 0
            term_rows = [i for i, s in enumerate(src) if gl.find(s)]
            for i, s in enumerate(src):
                if i not in term_rows and form != forms[0] and corpus != "glossary":
                    g_out.append(p_out[i])  # no term: the wrapper calls the inner translator unchanged
                    continue  # (measured once, with the first form)
                t0 = time.perf_counter()
                try:
                    out = " ".join((wrapped.translate(s) or "").split())
                    g_secs.append(time.perf_counter() - t0)
                except Exception:
                    out, g_fail = None, g_fail + 1
                if i in term_rows and out is not None:
                    used[i] = wrapped.last
                    if wrapped.last == "fallback":
                        fallbacks += 1
                    else:
                        survived += 1
                g_out.append(out)
                if pause and (i in term_rows or form == forms[0]):
                    time.sleep(pause)
            no_term = [i for i in range(len(src)) if i not in term_rows]
            identical = sum(g_out[i] == p_out[i] for i in no_term)
            leaks = sum(bool(o and LEAK.search(o)) and not (p and LEAK.search(p)) for o, p in zip(g_out, p_out))
            tot = sum(len(e) for e in exp)
            ok_b = sum(term_hits(h, e) for h, e in zip(p_out, exp))
            ok_a = sum(term_hits(h, e) for h, e in zip(g_out, exp))
            sb, sa = scorer.score(p_out, refs), scorer.score(g_out, refs)
            mb_, pb_ = ms(p_secs)
            if form == forms[0] or corpus == "glossary":
                ma_, pa_ = ms(g_secs)
            else:  # no-term sentences reuse the plain output: time only the term sentences + plain rest
                ma_, pa_ = ms(g_secs + [p_secs[k] for k in no_term if k < len(p_secs)])
            emit({"backend": name, "form": form, "corpus": corpus, "n": len(src), "term_sents": len(term_rows),
                  "terms_total": tot, "terms_ok_before": ok_b, "terms_ok_after": ok_a,
                  "term_acc_before": f"{ok_b / tot:.3f}" if tot else "", "term_acc_after": f"{ok_a / tot:.3f}" if tot else "",
                  "survived": survived, "fallbacks": fallbacks, "leaks": leaks, "no_term_sents": len(no_term),
                  "no_term_identical": identical,
                  "bleu_before": f"{sb['bleu']:.2f}" if sb else "", "bleu_after": f"{sa['bleu']:.2f}" if sa else "",
                  "chrf_before": f"{sb['chrf']:.2f}" if sb else "", "chrf_after": f"{sa['chrf']:.2f}" if sa else "",
                  "ms_mean_before": mb_, "ms_mean_after": ma_, "ms_p95_before": pb_, "ms_p95_after": pa_,
                  "failed": p_fail + g_fail})
            if samples is not None and corpus in ("glossary", "custom"):
                for i in term_rows:
                    samples.append({"backend": name, "form": form, "corpus": corpus, "src": src[i],
                                    "before": p_out[i] or "", "after": g_out[i] or "",
                                    "glossary": used.get(i, "")})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--glossary-set", type=Path, default=ROOT / "eval/data/mt_glossary_vi_en.tsv")
    p.add_argument("--custom-tsv", type=Path, default=ROOT / "eval/data/mt_custom_vi_en.tsv")
    p.add_argument("--fleurs-src-dir", type=Path)
    p.add_argument("--fleurs-tgt-tsv", type=Path)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backends", default="argos,nllb-600m,m2m100-418m,google")
    p.add_argument("--forms", default=",".join(ALL_FORMS))
    p.add_argument("--user-glossary", type=Path, default=None)
    p.add_argument("--google-pause", type=float, default=0.4, help="seconds between Google calls")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=Path, default=None)
    args = p.parse_args(argv)

    gl, problems = Glossary.load("vi", "en", args.user_glossary)
    for x in problems:
        print("glossary:", x, flush=True)
    gset = load_glossary_set(args.glossary_set)
    sets = [("glossary", [r["src"] for r in gset], [r["ref"] for r in gset], [r["expect"] for r in gset])]
    custom, _reviewed = mb.load_custom(args.custom_tsv)
    sets.append(("custom", [x.src for x in custom], [x.ref for x in custom], [[] for _ in custom]))
    if args.fleurs_src_dir and args.fleurs_tgt_tsv:
        fl = mb.subset(mb.load_fleurs_pairs(args.fleurs_src_dir.expanduser(), args.fleurs_tgt_tsv.expanduser()),
                       args.limit, args.seed)
        sets.append(("fleurs", [x.src for x in fl], [x.ref for x in fl], [[] for _ in fl]))
    for corpus, src, *_ in sets:
        print(f"{corpus}: {len(src)} sentences, {sum(bool(gl.find(s)) for s in src)} with a glossary term", flush=True)
    generic = [r["src"] for r in gset if r["category"] == "generic"]
    assert all(not gl.find(s) for s in generic), "a generic sentence matched a glossary term"

    makers = {"argos": lambda: mb.argos_backend([mb.default_argos_package()]),
              "google": mb.google_backend}
    forms = [f for f in args.forms.split(",") if f]
    scorer = mb.Scorer("en")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    samples = [] if args.samples else None
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()

        def emit(row):
            w.writerow(row)
            f.flush()
            print("  ".join(f"{k}={row[k]}" for k in COLUMNS if row[k] != ""), flush=True)

        for name in [b for b in args.backends.split(",") if b]:
            try:
                backend = makers[name]() if name in makers else mb.ct2_model_backend(name)
                tr = backend.make("vi", "en")
                if tr is None:
                    raise RuntimeError("no vi->en model installed")
                load = getattr(tr, "load", None)
                if load is not None:
                    load()
                tr.translate("Xin chào.")  # warm-up, untimed
            except Exception as exc:
                print(f"{name}: not measurable ({exc})", flush=True)
                continue
            bench_backend(name, tr, sets, gl, forms, scorer, args.google_pause if name == "google" else 0.0,
                          emit, samples, lambda m: print(m, flush=True),
                          checks=[(i, r["check"]) for i, r in enumerate(gset) if (r.get("check") or "").strip()])
    if samples is not None:
        with open(args.samples, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["backend", "form", "corpus", "src", "before", "after", "glossary"],
                               delimiter="\t")
            w.writeheader()
            w.writerows(samples)
    print(scorer.signatures())
    return 0


if __name__ == "__main__":
    sys.exit(main())
