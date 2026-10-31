"""Command-line front end for the orchestrator."""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from .asr import LOCAL_MODELS, convert_command
from .config import Settings
from .denoise import NS_MODES
from .model_manager import CATALOG, NOT_DOWNLOADED
from .models import ERROR, RESULT, STATUS, STOPPED, Event


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audio-translator --cli",
                                description="Vietnamese speech -> English speech into a virtual mic")
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--input", default=None, help="input device name (substring); default: system mic")
    p.add_argument("--output", default="BlackHole", help="output device name (substring); default: BlackHole")
    p.add_argument("--list-models", action="store_true",
                   help="list Whisper models with size and download status, and exit")
    p.add_argument("--model", default="medium", choices=list(CATALOG),
                   help="Whisper model (default: medium; see --list-models)")
    p.add_argument("--asr-prompt", default=None, metavar="TEXT",
                   help="Whisper initial prompt, e.g. a Vietnamese sentence with diacritics (default: none)")
    p.add_argument("--translator", default="offline", choices=["offline", "google"],
                   help="translation backend (default: offline, falls back to google)")
    d = Settings()
    p.add_argument("--ns", default=d.ns_mode, choices=list(NS_MODES),
                   help="noise suppression before VAD/ASR (experimental; default: off)")
    p.add_argument("--ns-mix", type=_unit_float, default=d.ns_mix,
                   help=f"observation-adding weight of the raw signal, 0..1 (default: {d.ns_mix})")
    p.add_argument("--ns-gate-db", type=_gate_db, default=d.ns_gate_snr_db,
                   help=f"skip suppression above this input SNR in dB, or 'none' to always "
                        f"suppress (default: {d.ns_gate_snr_db})")
    return p


def _unit_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {v}")
    return v


def _gate_db(text: str) -> float | None:
    if text.strip().lower() == "none":
        return None
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number of dB or 'none', got {text!r}") from None


def print_devices(out=sys.stdout) -> int:
    from .devices import list_devices

    try:
        ins, outs = list_devices("input"), list_devices("output")
    except Exception as exc:
        print(f"Cannot query audio devices: {exc}", file=sys.stderr)
        return 1
    print("Input devices:", file=out)
    for d in ins:
        print(f"  [{d.index}] {d.name} ({d.max_channels} ch, {d.samplerate:.0f} Hz)", file=out)
    print("Output devices:", file=out)
    for d in outs:
        print(f"  [{d.index}] {d.name} ({d.max_channels} ch, {d.samplerate:.0f} Hz)", file=out)
    if not any("blackhole" in d.name.lower() for d in outs):
        print("\nBlackHole not found. Install it with: brew install blackhole-2ch", file=out)
    return 0


def print_models(selected: str | None = None, manager=None, out=None) -> int:
    from .model_manager import ModelManager

    out = out or sys.stdout
    mgr = manager or ModelManager(lambda name: None)  # factory unused: nothing is loaded here
    print("Whisper models:", file=out)
    to_convert = []
    for m in mgr.models():
        mark = "*" if m.name == selected else " "
        print(f" {mark} {m.name:<17} ~{m.size_label:>7}  {m.status}", file=out)
        if m.name in LOCAL_MODELS and m.status == NOT_DOWNLOADED:
            to_convert.append(m.name)
    print("Not-downloaded hub models are fetched on first use. * = --model", file=out)
    for name in to_convert:
        print(f"'{name}' must be converted once first: {convert_command(name)}", file=out)
    return 0


def format_event(ev: Event) -> str:
    if ev.kind == RESULT:
        lat = ev.data.get("latency")
        lat_s = f"{lat:.2f} s" if lat is not None else "n/a"
        return (f"[seg {ev.segment_id}] VI: {ev.data.get('source_text', '')}\n"
                f"          EN: {ev.data.get('text', '')}\n"
                f"          latency (first_audio_out - speech_end): {lat_s}")
    return f"[{ev.kind}] {ev.message}"


def main(argv: list[str] | None = None, orchestrator_factory=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        return print_devices()
    if args.list_models:
        return print_models(args.model)

    from .orchestrator import Orchestrator

    settings = Settings(input_device=args.input, output_device=args.output, model_size=args.model,
                        asr_initial_prompt=args.asr_prompt, translator=args.translator,
                        ns_mode=args.ns, ns_mix=args.ns_mix, ns_gate_snr_db=args.ns_gate_db)
    done = threading.Event()
    failed = []

    def on_event(ev: Event) -> None:
        print(format_event(ev), flush=True)
        if ev.kind == ERROR:
            failed.append(ev.message)
        if ev.kind == STATUS and ev.data.get("state") == STOPPED:
            done.set()

    orch = (orchestrator_factory or Orchestrator)(settings, on_event=on_event)
    interrupts = []

    def on_sigint(*_):
        interrupts.append(1)
        if len(interrupts) > 1:  # second Ctrl+C: give up waiting (e.g. during model load)
            raise KeyboardInterrupt
        print("Stopping…", flush=True)
        threading.Thread(target=orch.stop, daemon=True).start()

    prev = signal.signal(signal.SIGINT, on_sigint)
    try:
        orch.start()
        print("Speak Vietnamese. Press Ctrl+C to stop.", flush=True)
        while not done.wait(0.2):
            pass
    except KeyboardInterrupt:
        return 130
    finally:
        signal.signal(signal.SIGINT, prev)
        orch.shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
