"""Summarise a real-app latency log (``audio-translator --latency-log PATH``).

Per stage and end to end, over **played** segments: n, mean, P50, P95, max
(ms); then the dropped segments by reason. ``--markdown`` prints tables ready
to paste into the README::

    uv run python eval/latency_report.py ~/gr2_latency_cli.csv
    uv run python eval/latency_report.py ~/gr2_latency_*.csv --session 2 --markdown

Stages (see ``audio_translator.latency_log.DURATIONS``): ``segment_ms`` is
end of speech (last voiced frame) -> segment cut, so it **includes** the
silence wait (``silence_ms``, 600 ms) that tells the utterance has ended;
``*_wait_ms`` are queue waits; ``e2e_ms`` = first audio out - end of speech.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_translator.latency_log import DROPPED, DURATIONS, PLAYED  # noqa: E402

STAGES = [d[0] for d in DURATIONS]


def load(paths: list[Path], session: int | None = None) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if session is None or r.get("session_id") == str(session):
                    r["_file"] = p.name
                    rows.append(r)
    return rows


def stage_stats(rows: list[dict]) -> list[tuple[str, int, float, float, float, float]]:
    """(stage, n, mean, p50, p95, max) over played rows; stages with no value are left out."""
    out = []
    played = [r for r in rows if r.get("status") == PLAYED]
    for st in STAGES:
        v = np.array([float(r[st]) for r in played if r.get(st, "") != ""])
        if v.size:
            out.append((st, int(v.size), float(v.mean()), float(np.percentile(v, 50)),
                        float(np.percentile(v, 95)), float(v.max())))
    return out


def drop_counts(rows: list[dict]) -> collections.Counter:
    c = collections.Counter()
    for r in rows:
        if r.get("status") == DROPPED:
            detail = r.get("detail", "")
            why = detail.split(":", 1)[0] if r.get("reason") == "no_speech" and detail else ""
            c[(r.get("reason", ""), why)] += 1
    return c


def health_counts(rows: list[dict]) -> list[tuple[str, int, int, str]]:
    """(level, played, dropped, most frequent reason) by ``health_max``; [] for logs without it."""
    out = []
    for lvl in ("green", "yellow", "red"):
        sel = [r for r in rows if r.get("health_max") == lvl]
        if sel:
            why = collections.Counter(r.get("health_reason", "") for r in sel if r.get("health_reason"))
            out.append((lvl, sum(r.get("status") == PLAYED for r in sel), sum(r.get("status") == DROPPED for r in sel),
                        why.most_common(1)[0][0] if why else ""))
    return out


def report(rows: list[dict], markdown: bool = False) -> str:
    lines = []
    played = sum(r.get("status") == PLAYED for r in rows)
    dropped = sum(r.get("status") == DROPPED for r in rows)
    configs = sorted({tuple(r.get(k, "") for k in ("asr_model", "vad", "asr_filter", "restore", "translator"))
                      for r in rows})
    sessions = sorted({(r["_file"], r.get("session_id", "")) for r in rows})
    lines.append(f"{len(rows)} segment(s) in {len(sessions)} session(s): {played} played, {dropped} dropped")
    for c in configs:
        lines.append("config: asr_model={} vad={} asr_filter={} restore={} translator={}".format(*c))
    words = [int(r["n_words_vi"]) for r in rows if r.get("status") == PLAYED and r.get("n_words_vi")]
    if words:
        lines.append(f"words per played segment (vi): mean {np.mean(words):.1f}, max {max(words)}")
    lines.append("")
    stats = stage_stats(rows)
    if markdown:
        lines += ["| Khâu | n | TB (ms) | P50 | P95 | Max |", "|---|---|---|---|---|---|"]
        lines += [f"| `{s}` | {n} | {m:.0f} | {p50:.0f} | {p95:.0f} | {mx:.0f} |" for s, n, m, p50, p95, mx in stats]
    else:
        lines.append(f"{'stage':<16} {'n':>4} {'mean':>8} {'p50':>8} {'p95':>8} {'max':>8}   (ms)")
        lines += [f"{s:<16} {n:>4} {m:>8.0f} {p50:>8.0f} {p95:>8.0f} {mx:>8.0f}" for s, n, m, p50, p95, mx in stats]
    if not stats:
        lines.append("(no played segment with timestamps)")
    lights = health_counts(rows)
    if lights:
        lines.append("")
        if markdown:
            lines += ["| Đèn cao nhất | Đã phát | Đã bỏ | Lý do hay gặp nhất |", "|---|---|---|---|"]
            lines += [f"| {lvl} | {p} | {d} | {why or '–'} |" for lvl, p, d, why in lights]
        else:
            lines.append("speaker indicator, highest level per segment (played / dropped, top reason):")
            lines += [f"  {lvl:<7} {p:>3} / {d:<3} {why}" for lvl, p, d, why in lights]
    drops = drop_counts(rows)
    if drops:
        lines.append("")
        if markdown:
            lines += ["| Lý do bỏ | Chi tiết | Số đoạn |", "|---|---|---|"]
            lines += [f"| `{r}` | {w or '–'} | {n} |" for (r, w), n in drops.most_common()]
        else:
            lines.append("dropped segments by reason:")
            lines += [f"  {r}{f' ({w})' if w else ''}: {n}" for (r, w), n in drops.most_common()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv", nargs="+", type=Path, help="latency log(s) written with --latency-log")
    p.add_argument("--session", type=int, default=None, help="only this session_id")
    p.add_argument("--markdown", action="store_true", help="print Markdown tables")
    args = p.parse_args(argv)
    missing = [str(x) for x in args.csv if not x.expanduser().is_file()]
    if missing:
        print(f"error: not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    rows = load([x.expanduser() for x in args.csv], args.session)
    if not rows:
        print("error: no rows (empty log or wrong --session)", file=sys.stderr)
        return 1
    print(report(rows, args.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
