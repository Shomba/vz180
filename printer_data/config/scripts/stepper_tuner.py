#!/home/shoomba/klippy-env/bin/python
"""
Stepper Audio Tuner para Klipper (VZ180 AWD CoreXY)
Usa o microfone da webcam USB (PipeWire/PulseAudio) para medir a frequencia
acustica real emitida pelos motores de passo, calcular o fator de afinacao
(em Hz e cents) e gerar a resposta de frequencia da estrutura da impressora.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

MOONRAKER_URL = "http://127.0.0.1:7125"
SAMPLE_RATE = 48000
CALIBRATION_OUT = Path("/home/shoomba/printer_data/config/stepper_audio_calibration.json")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

TEST_NOTES = [
    ("C4", 261.63),
    ("E4", 329.63),
    ("G4", 392.00),
    ("A4", 440.00),
    ("C5", 523.25),
    ("E5", 659.25),
    ("G5", 783.99),
    ("A5", 880.00),
]


def hz_to_note_name(hz: float) -> tuple[str, float]:
    """Convert frequency in Hz to nearest musical note name and deviation in cents."""
    if hz <= 0:
        return ("N/A", 0.0)
    midi_float = 69.0 + 12.0 * math.log2(hz / 440.0)
    midi_int = int(round(midi_float))
    cents = (midi_float - midi_int) * 100.0
    name = NOTE_NAMES[midi_int % 12]
    octave = (midi_int // 12) - 1
    return (f"{name}{octave}", cents)


def find_camera_mic() -> str:
    """Auto-detect the USB camera microphone source in PipeWire/PulseAudio."""
    out = subprocess.check_output(["pactl", "list", "short", "sources"], text=True)
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            src = parts[1]
            if "usb" in src.lower() and "camera" in src.lower() and "monitor" not in src.lower():
                return src
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and "alsa_input" in parts[1]:
            return parts[1]
    raise RuntimeError("Nenhum microfone encontrado via pactl.")


def moonraker_get(path: str) -> dict:
    with urllib.request.urlopen(f"{MOONRAKER_URL}{path}", timeout=10) as r:
        return json.loads(r.read().decode())


def run_gcode(script: str, timeout: float = 30.0) -> None:
    encoded = urllib.parse.quote(script)
    req = urllib.request.Request(
        f"{MOONRAKER_URL}/printer/gcode/script?script={encoded}",
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read().decode())
        if res.get("result") != "ok":
            raise RuntimeError(f"Erro G-code: {res}")


def record_audio(source: str, duration_s: float) -> np.ndarray:
    """Record mono s16le 48kHz audio from PipeWire/PulseAudio source using parec."""
    n_bytes = int(SAMPLE_RATE * duration_s) * 2
    cmd = [
        "parec",
        "--device", source,
        "--format=s16le",
        f"--rate={SAMPLE_RATE}",
        "--channels=1",
        "--raw",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    chunks = []
    remaining = n_bytes
    try:
        while remaining > 0:
            data = proc.stdout.read(min(8192, remaining))
            if not data:
                break
            chunks.append(data)
            remaining -= len(data)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    raw = b"".join(chunks)
    if len(raw) < 2:
        return np.zeros(1, dtype=np.float64)
    samples = np.frombuffer(raw[: len(raw) - (len(raw) % 2)], dtype=np.int16).astype(np.float64)
    return samples / 32768.0


def compute_spectrum(samples: np.ndarray, f_min: float = 120.0, f_max: float = 4000.0) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass filter and compute high-resolution zero-padded magnitude spectrum."""
    if len(samples) < 256:
        return np.array([0.0]), np.array([0.0])
    samples = samples - np.mean(samples)
    nyq = SAMPLE_RATE / 2.0
    b, a = butter(4, [f_min / nyq, f_max / nyq], btype="band")
    filtered = filtfilt(b, a, samples)
    window = np.hanning(len(filtered))
    n_fft = 1 << int(math.ceil(math.log2(len(filtered) * 4)))
    spec = np.abs(np.fft.rfft(filtered * window, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)
    mask = (freqs >= f_min) & (freqs <= f_max)
    return freqs[mask], spec[mask]


def analyze_tone(
    samples: np.ndarray,
    noise_freqs: np.ndarray,
    noise_spec: np.ndarray,
    expected_hz: float,
) -> dict:
    """Estimate dominant frequency, fundamental (HPS), SNR, and pitch deviation."""
    # Trim first/last 15% to avoid acceleration/deceleration ramps
    trim = int(len(samples) * 0.15)
    core = samples[trim : len(samples) - trim] if len(samples) > 1000 else samples
    freqs, spec = compute_spectrum(core)

    # Interpolate noise floor to match frequency grid and subtract
    noise_interp = np.interp(freqs, noise_freqs, noise_spec)
    # Normalize noise floor to same window length ratio
    clean_spec = np.maximum(spec - 1.5 * noise_interp, 0.0)

    # Parabolic interpolation around peak for sub-Hz precision
    peak_idx = int(np.argmax(clean_spec))
    peak_mag = float(clean_spec[peak_idx])
    median_floor = float(np.median(clean_spec) + 1e-9)
    snr_db = 20.0 * math.log10((peak_mag + 1e-9) / median_floor)

    if 0 < peak_idx < len(clean_spec) - 1:
        y0, y1, y2 = clean_spec[peak_idx - 1], clean_spec[peak_idx], clean_spec[peak_idx + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        step = freqs[1] - freqs[0]
        peak_hz = float(freqs[peak_idx] + delta * step)
    else:
        peak_hz = float(freqs[peak_idx])

    # Also check energy specifically near harmonic multiples (0.25x, 0.5x, 1x, 2x, 4x)
    candidates = {}
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        target = expected_hz * mult
        if freqs[0] <= target <= freqs[-1]:
            win = (freqs >= target * 0.94) & (freqs <= target * 1.06)
            if np.any(win):
                sub_idx = int(np.argmax(clean_spec[win]))
                sub_freqs = freqs[win]
                sub_mags = clean_spec[win]
                candidates[str(mult)] = {
                    "hz": round(float(sub_freqs[sub_idx]), 2),
                    "mag": round(float(sub_mags[sub_idx]), 4),
                }

    ratio = peak_hz / expected_hz if expected_hz > 0 else 0.0
    note_name, cents_nearest = hz_to_note_name(peak_hz)
    cents_vs_expected = 1200.0 * math.log2(peak_hz / expected_hz) if peak_hz > 0 and expected_hz > 0 else 0.0
    rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(core ** 2))) + 1e-9)

    return {
        "expected_hz": round(expected_hz, 2),
        "measured_peak_hz": round(peak_hz, 2),
        "measured_note": note_name,
        "cents_from_nearest_note": round(cents_nearest, 1),
        "cents_vs_expected": round(cents_vs_expected, 1),
        "ratio": round(ratio, 4),
        "snr_db": round(snr_db, 1),
        "rms_db": round(rms_db, 1),
        "harmonic_candidates": candidates,
    }


def get_printer_microsteps() -> int:
    res = moonraker_get("/printer/objects/query?configfile")
    settings = res["result"]["status"]["configfile"]["settings"]
    return int(settings["stepper_x"]["microsteps"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Afinador acustico de motores de passo via microfone da webcam")
    parser.add_argument("--source", default=None, help="Fonte PulseAudio/PipeWire do microfone (auto-detectado por padrao)")
    parser.add_argument("--duration", type=float, default=0.45, help="Duracao de cada nota de teste em segundos (padrao: 0.45)")
    parser.add_argument("--no-audio-mode", action="store_true", help="Nao ativar _AUDIO_MODE_ON (testar no modo normal)")
    parser.add_argument("--steps-per-mm", type=float, default=5.0, help="Passos cheios por mm (padrao: 5.0 para 200 passos / 40mm)")
    args = parser.parse_args()

    source = args.source or find_camera_mic()
    print(f"[Microfone] Usando: {source}")

    info = moonraker_get("/printer/info")["result"]
    if info["state"] != "ready":
        print(f"ERRO: Klipper nao esta pronto (state={info['state']}).")
        return 1

    orig_ms = get_printer_microsteps()
    use_audio_mode = not args.no_audio_mode
    scale = (orig_ms / 4.0) if use_audio_mode else 1.0
    print(f"[Klipper] microsteps={orig_ms} | _AUDIO_MODE_ON={use_audio_mode} | scale={scale}")

    print("[Ruido Ambiente] Gravando referencia de silencio (0.8s)...")
    noise_samples = record_audio(source, 0.8)
    noise_freqs, noise_spec = compute_spectrum(noise_samples)
    noise_rms = 20.0 * math.log10(float(np.sqrt(np.mean(noise_samples ** 2))) + 1e-9)
    print(f"[Ruido Ambiente] Nivel RMS: {noise_rms:.1f} dBFS")

    # Home XY if needed and center toolhead
    homed = moonraker_get("/printer/objects/query?toolhead")["result"]["status"]["toolhead"]["homed_axes"]
    if "x" not in homed or "y" not in homed:
        print("[Klipper] Fazendo home X Y...")
        run_gcode("G28 X Y", timeout=60.0)

    run_gcode("G90\nG1 X90 Y87 F9000\nM400")

    results = []
    direction = 1
    try:
        if use_audio_mode:
            run_gcode("_AUDIO_MODE_ON")

        print("\nNota | Alvo (Hz) | Medido (Hz) | Nota Real | Ratio  | Desvio (cents) | SNR (dB) | RMS (dB)")
        print("-" * 92)

        for note_label, target_hz in TEST_NOTES:
            # Physical speed for target_hz at steps_per_mm full steps/mm
            v_phys = target_hz / args.steps_per_mm  # mm/s
            # Keep physical travel within +-30mm around center by capping or oscillating
            dur = args.duration
            dist_phys = v_phys * dur
            if dist_phys > 50.0:
                # Split into back-and-forth segments of 40mm each so we never leave the center
                seg_dist = 35.0
                n_segs = max(2, int(math.ceil(dist_phys / seg_dist)))
                seg_cmd = seg_dist / scale
                f_cmd = (v_phys * 60.0) / scale
                moves = ["G91"]
                d = direction
                for _ in range(n_segs):
                    moves.append(f"G1 X{d * seg_cmd:.4f} F{f_cmd:.2f}")
                    d = -d
                if n_segs % 2 == 1:
                    moves.append(f"G1 X{d * seg_cmd:.4f} F{f_cmd:.2f}")
                moves.append("M400")
                gcode = "\n".join(moves)
                rec_dur = (n_segs * seg_dist / v_phys) + 0.12
            else:
                dist_cmd = dist_phys / scale
                f_cmd = (v_phys * 60.0) / scale
                # Move out and back to stay centered at X90
                gcode = (
                    f"G91\n"
                    f"G1 X{direction * dist_cmd:.4f} F{f_cmd:.2f}\n"
                    f"G1 X{-direction * dist_cmd:.4f} F{f_cmd:.2f}\n"
                    f"M400"
                )
                rec_dur = (dur * 2.0) + 0.12

            # Record in background thread while G-code plays so pipe never blocks
            rec_holder: list[np.ndarray] = []

            def _rec_worker() -> None:
                rec_holder.append(record_audio(source, rec_dur))

            t_rec = threading.Thread(target=_rec_worker, daemon=True)
            t_rec.start()
            time.sleep(0.06)
            run_gcode(gcode)
            t_rec.join(timeout=rec_dur + 3.0)
            samples = rec_holder[0] if rec_holder else np.zeros(1, dtype=np.float64)
            analysis = analyze_tone(samples, noise_freqs, noise_spec, target_hz)
            analysis["target_note"] = note_label
            results.append(analysis)

            print(
                f"{note_label:<4} | {target_hz:9.2f} | {analysis['measured_peak_hz']:11.2f} | "
                f"{analysis['measured_note']:<9} | {analysis['ratio']:6.3f} | "
                f"{analysis['cents_vs_expected']:+14.1f} | {analysis['snr_db']:8.1f} | {analysis['rms_db']:8.1f}"
            )
            direction = -direction
            time.sleep(0.15)

    finally:
        if use_audio_mode:
            run_gcode("G90\n_AUDIO_MODE_OFF")
        else:
            run_gcode("G90")

    # Compute median ratio across valid detections (SNR > 8 dB)
    valid = [r for r in results if r["snr_db"] >= 8.0]
    ratios = [r["ratio"] for r in valid] if valid else [r["ratio"] for r in results]
    median_ratio = float(np.median(ratios)) if ratios else 1.0
    correction_factor = 1.0 / median_ratio if median_ratio > 0 else 1.0

    # Find loudest resonant note
    loudest = max(results, key=lambda r: r["rms_db"]) if results else None

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mic_source": source,
        "microsteps_config": orig_ms,
        "audio_mode_4ms_1p7a": use_audio_mode,
        "noise_floor_rms_db": round(noise_rms, 1),
        "median_measured_to_commanded_ratio": round(median_ratio, 4),
        "recommended_feedrate_multiplier": round(correction_factor, 4),
        "loudest_resonance": {
            "note": loudest["target_note"],
            "measured_hz": loudest["measured_peak_hz"],
            "rms_db": loudest["rms_db"],
        } if loudest else None,
        "notes": results,
    }
    CALIBRATION_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== RESUMO DA AFINACAO ===")
    print(f"Razao mediana (Medido / Comandado): {median_ratio:.4f}x")
    print(f"Fator de correcao recomendado no Feedrate (F): {correction_factor:.4f}x")
    if loudest:
        print(f"Melhor ressonancia da estrutura: {loudest['target_note']} ({loudest['measured_peak_hz']} Hz, {loudest['rms_db']} dBFS)")
    print(f"Calibracao salva em: {CALIBRATION_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
