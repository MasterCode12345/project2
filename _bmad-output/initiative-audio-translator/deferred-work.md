- source_plan: `/Users/trantrungnghia/GR2/_bmad-output/initiative-audio-translator/plan-audio-translator-mvp.md`
  summary: Full PySide6 GUI beyond the minimal window — language-pair selection, settings dialog (model size, VAD params), live latency stats panel.
  evidence: Plan exceeded 1600 tokens; user chose Split, then asked for a minimal window, which is now in the MVP plan.
- source_plan: `/Users/trantrungnghia/GR2/_bmad-output/initiative-audio-translator/plan-audio-translator-mvp.md`
  summary: Measurement report — CSV export of per-stage timestamps, mean/P95 end-to-end latency, segmentation wait time, WER evaluation.
  evidence: Plan exceeded 1600 tokens; user chose Split. Pipeline already records timestamps; reporting can ship independently.
- source_plan: `/Users/trantrungnghia/GR2/_bmad-output/initiative-audio-translator/plan-model-selection.md`
  summary: GUI "Model:" dropdown in the main window — lists the catalog with downloaded/loaded/in-use status, preloads on change via `orch.select_model`, disabled while a session runs (user decision: switch only when stopped).
  evidence: Plan exceeded 1600 tokens; user chose Split. The dropdown is a thin layer over the ModelManager/orchestrator API shipped first.
- source_plan: `/Users/trantrungnghia/GR2/_bmad-output/initiative-audio-translator/plan-asr-vietnamese-accuracy.md`
  summary: Manually run one live CLI session with `--model phowhisper-small` (mic + BlackHole) to close acceptance criterion 1.
  evidence: Headless environment; only wiring tests and a direct WhisperASR transcription were run (unverified, would be medium if it failed).
