#!/home/shoomba/klippy-env/bin/python
"""
Stepper Audio Tuner para Klipper CoreXY AWD (VZ180)
Afina cada grupo de motores CoreXY INDIVIDUALMENTE usando o microfone da webcam USB:
  - Motor A (stepper_x + stepper_x1): diagonal +45 deg (dX = +d, dY = +d) -> Motor B parado
  - Motor B (stepper_y + stepper_y1): diagonal -45 deg (dX = +d, dY = -d) -> Motor A parado
  - Motor Z (stepper_z, opcional via --motors A,B,Z): movimento vertical isolado
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

OCTAVE_NOTES_XY = [
    ("C2", 65.41),
    ("G2", 98.00),
    ("C3", 130.81),
    ("G3", 196.00),
    ("C4", 261.63),
    ("G4", 392.00),
    ("C5", 523.25),
    ("G5", 783.99),
    ("C6", 1046.50),
    ("G6", 1567.98),
    ("C7", 2093.00),
    ("G7", 3135.96),
    ("C8", 4186.01),
    ("E8", 5274.04),
]

OCTAVE_NOTES_Z = [
    ("C2", 65.41),
    ("G2", 98.00),
    ("C3", 130.81),
    ("G3", 196.00),
    ("C4", 261.63),
    ("G4", 392.00),
    ("C5", 523.25),
    ("G5", 783.99),
    ("C6", 1046.50),
    ("G6", 1567.98),
    ("C7", 2093.00),
    ("F7", 2793.83),
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


def compute_spectrum(samples: np.ndarray, f_min: float = 55.0, f_max: float = 8000.0) -> tuple[np.ndarray, np.ndarray]:
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
    """Estimate dominant frequency, SNR, RMS volume, and pitch deviation."""
    trim = int(len(samples) * 0.15)
    core = samples[trim : len(samples) - trim] if len(samples) > 1000 else samples
    freqs, spec = compute_spectrum(core)

    noise_interp = np.interp(freqs, noise_freqs, noise_spec)
    clean_spec = np.maximum(spec - 1.5 * noise_interp, 0.0)

    peak_idx = int(np.argmax(clean_spec))
    peak_mag = float(clean_spec[peak_idx])
    median_floor = float(np.median(clean_spec) + 1e-9)
    snr_db = 20.0 * math.log10((peak_mag + 1e-9) / median_floor)

    if 0 < peak_idx < len(clean_spec) - 1:
        y0, y1, y2 = clean_spec[peak_idx - 1], clean_spec[peak_idx], clean_spec[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        step = freqs[1] - freqs[0]
        peak_hz = float(freqs[peak_idx] + delta * step)
    else:
        peak_hz = float(freqs[peak_idx])

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
    }


def get_stepper_settings() -> dict:
    res = moonraker_get("/printer/objects/query?configfile")
    return res["result"]["status"]["configfile"]["settings"]


def build_corexy_isolated_move(
    motor: str,
    target_hz: float,
    dur: float,
    steps_per_mm_xy: float,
    scale_xy: float,
    direction: int,
) -> tuple[str, float]:
    """
    Build G-code that spins ONLY CoreXY Motor A (dX=+d, dY=+d) or Motor B (dX=+d, dY=-d).
    In CoreXY:
      dA = dX + dY,  dB = dX - dY
      For Motor A only: dX = +dA/2, dY = +dA/2 -> dB = 0, toolhead speed = vA / sqrt(2)
      For Motor B only: dX = +dB/2, dY = -dB/2 -> dA = 0, toolhead speed = vB / sqrt(2)
    """
    y_sign = 1.0 if motor == "A" else -1.0
    v_motor = target_hz / steps_per_mm_xy  # belt speed in mm/s
    v_toolhead = v_motor / math.sqrt(2.0)  # Cartesian diagonal speed in mm/s
    dist_motor = v_motor * dur             # belt travel in mm
    # Axis travel per segment = dist_motor / 2; keep within +-25mm of center (50mm belt travel)
    max_belt_seg = 50.0
    f_cmd = (v_toolhead * 60.0) / scale_xy

    if dist_motor > max_belt_seg:
        n_segs = max(2, int(math.ceil(dist_motor / 40.0)))
        if n_segs % 2 == 1:
            n_segs += 1  # even number of segments returns to exact center
        seg_belt = 40.0
        axis_cmd = (seg_belt / 2.0) / scale_xy
        moves = ["G91"]
        d = direction
        for _ in range(n_segs):
            moves.append(f"G1 X{d * axis_cmd:.4f} Y{d * y_sign * axis_cmd:.4f} F{f_cmd:.2f}")
            d = -d
        moves.append("M400")
        rec_dur = (n_segs * seg_belt / v_motor) + 0.14
        return "\n".join(moves), rec_dur

    axis_cmd = (dist_motor / 2.0) / scale_xy
    gcode = (
        f"G91\n"
        f"G1 X{direction * axis_cmd:.4f} Y{direction * y_sign * axis_cmd:.4f} F{f_cmd:.2f}\n"
        f"G1 X{-direction * axis_cmd:.4f} Y{-direction * y_sign * axis_cmd:.4f} F{f_cmd:.2f}\n"
        f"M400"
    )
    rec_dur = (dur * 2.0) + 0.14
    return gcode, rec_dur


def build_z_isolated_move(
    target_hz: float,
    dur: float,
    steps_per_mm_z: float,
    direction: int,
) -> tuple[str, float]:
    """Build isolated vertical Z move oscillating safely around current Z."""
    v_z = target_hz / steps_per_mm_z  # mm/s
    dist_z = min(v_z * dur, 10.0)
    f_cmd = v_z * 60.0
    actual_dur = dist_z / v_z if v_z > 0 else dur
    gcode = (
        f"G91\n"
        f"G1 Z{direction * dist_z:.4f} F{f_cmd:.2f}\n"
        f"G1 Z{-direction * dist_z:.4f} F{f_cmd:.2f}\n"
        f"M400"
    )
    return gcode, (actual_dur * 2.0) + 0.14


def tune_single_motor(
    motor: str,
    source: str,
    noise_freqs: np.ndarray,
    noise_spec: np.ndarray,
    dur: float,
    steps_per_mm_xy: float,
    steps_per_mm_z: float,
    scale_xy: float,
    mode: str = "standard",
    max_xy_vel: float = 800.0,
    max_z_vel: float = 80.0,
) -> dict:
    label_map = {
        "A": "Motor A (stepper_x + stepper_x1, diagonal +45 deg, B parado)",
        "B": "Motor B (stepper_y + stepper_y1, diagonal -45 deg, A parado)",
        "Z": "Motor Z (stepper_z, vertical isolado)",
    }
    if mode == "octaves":
        note_list = OCTAVE_NOTES_XY if motor in ("A", "B") else OCTAVE_NOTES_Z
    else:
        note_list = TEST_NOTES

    print(f"\n=== Afinando {label_map[motor]} ===")
    print("Nota | Vel(mm/s) | Alvo (Hz) | Medido (Hz) | Nota Real | Ratio  | Desvio (cents) | SNR (dB) | RMS (dB)")
    print("-" * 105)

    results = []
    direction = 1
    for note_label, target_hz in note_list:
        if motor in ("A", "B"):
            v_phys = (target_hz / steps_per_mm_xy) / math.sqrt(2.0)
            if v_phys > max_xy_vel:
                continue
            gcode, rec_dur = build_corexy_isolated_move(
                motor, target_hz, dur, steps_per_mm_xy, scale_xy, direction
            )
        else:
            v_phys = target_hz / steps_per_mm_z
            if v_phys > max_z_vel:
                continue
            gcode, rec_dur = build_z_isolated_move(target_hz, dur, steps_per_mm_z, direction)

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
        analysis["velocity_mm_s"] = round(v_phys, 1)
        results.append(analysis)

        print(
            f"{note_label:<4} | {v_phys:9.1f} | {target_hz:9.2f} | {analysis['measured_peak_hz']:11.2f} | "
            f"{analysis['measured_note']:<9} | {analysis['ratio']:6.3f} | "
            f"{analysis['cents_vs_expected']:+14.1f} | {analysis['snr_db']:8.1f} | {analysis['rms_db']:8.1f}"
        )
        direction = -direction
        time.sleep(0.12)

    valid = [r for r in results if r["snr_db"] >= 8.0]
    ratios = [r["ratio"] for r in valid] if valid else [r["ratio"] for r in results]
    median_ratio = float(np.median(ratios)) if ratios else 1.0
    correction_factor = 1.0 / median_ratio if median_ratio > 0 else 1.0
    loudest = max(results, key=lambda r: r["rms_db"]) if results else None

    print(f"--> Motor {motor}: Razao mediana = {median_ratio:.4f}x | Multiplicador F = {correction_factor:.4f}x")
    return {
        "description": label_map[motor],
        "median_measured_to_commanded_ratio": round(median_ratio, 4),
        "recommended_feedrate_multiplier": round(correction_factor, 4),
        "loudest_resonance": {
            "note": loudest["target_note"],
            "measured_hz": loudest["measured_peak_hz"],
            "rms_db": loudest["rms_db"],
        } if loudest else None,
        "notes": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Afinador acustico individual de motores CoreXY via microfone da webcam")
    parser.add_argument("--source", default=None, help="Fonte PulseAudio/PipeWire do microfone")
    parser.add_argument("--motors", default="A,B", help="Motores para afinar separados por virgula (ex: A,B ou A,B,Z)")
    parser.add_argument("--mode", choices=["standard", "octaves"], default="standard", help="Modo: standard (C4-A5) ou octaves (C2-E8)")
    parser.add_argument("--max-xy-vel", type=float, default=800.0, help="Velocidade maxima XY fisica em mm/s (padrao: 800)")
    parser.add_argument("--max-z-vel", type=float, default=100.0, help="Velocidade maxima Z fisica em mm/s (padrao: 100)")
    parser.add_argument("--duration", type=float, default=0.35, help="Duracao por nota em segundos (padrao: 0.35)")
    parser.add_argument("--no-audio-mode", action="store_true", help="Nao ativar _AUDIO_MODE_ON")
    args = parser.parse_args()

    motors = [m.strip().upper() for m in args.motors.split(",") if m.strip().upper() in ("A", "B", "Z")]
    if not motors:
        print("ERRO: informe --motors com A, B e/ou Z.")
        return 1

    source = args.source or find_camera_mic()
    print(f"[Microfone] Usando: {source}")

    info = moonraker_get("/printer/info")["result"]
    if info["state"] != "ready":
        print(f"ERRO: Klipper nao esta pronto (state={info['state']}).")
        return 1

    cfg = get_stepper_settings()
    orig_ms_xy = int(cfg["stepper_x"]["microsteps"])
    rot_xy = float(cfg["stepper_x"]["rotation_distance"])
    steps_per_mm_xy = 200.0 / rot_xy
    rot_z = float(cfg["stepper_z"]["rotation_distance"])
    steps_per_mm_z = 200.0 / rot_z
    klipper_max_z_vel = float(cfg["printer"].get("max_z_velocity", 80.0))
    eff_max_z_vel = min(args.max_z_vel, klipper_max_z_vel)

    use_audio_mode = not args.no_audio_mode
    scale_xy = (orig_ms_xy / 4.0) if use_audio_mode else 1.0
    print(f"[Klipper] XY microsteps={orig_ms_xy} ({steps_per_mm_xy:.2f} passos/mm) | _AUDIO_MODE_ON={use_audio_mode} | scale={scale_xy}")

    print("[Ruido Ambiente] Gravando referencia de silencio (0.8s)...")
    noise_samples = record_audio(source, 0.8)
    noise_freqs, noise_spec = compute_spectrum(noise_samples)
    noise_rms = 20.0 * math.log10(float(np.sqrt(np.mean(noise_samples ** 2))) + 1e-9)
    print(f"[Ruido Ambiente] Nivel RMS: {noise_rms:.1f} dBFS")

    homed = moonraker_get("/printer/objects/query?toolhead")["result"]["status"]["toolhead"]["homed_axes"]
    if ("A" in motors or "B" in motors) and ("x" not in homed or "y" not in homed):
        print("[Klipper] Fazendo home X Y...")
        run_gcode("G28 X Y", timeout=60.0)
    if "Z" in motors and "z" not in homed:
        print("[Klipper] Fazendo home Z...")
        run_gcode("G28 Z", timeout=60.0)

    if "A" in motors or "B" in motors:
        run_gcode("G90\nG1 X90 Y87 F9000\nM400")
    if "Z" in motors:
        run_gcode("G90\nG1 Z35 F1800\nM400")

    motor_results = {}
    audio_enabled = False
    try:
        if use_audio_mode and ("A" in motors or "B" in motors):
            run_gcode("_AUDIO_MODE_ON")
            audio_enabled = True

        for m in motors:
            if m == "Z" and audio_enabled:
                run_gcode("G90\n_AUDIO_MODE_OFF")
                audio_enabled = False
            motor_results[f"motor_{m}"] = tune_single_motor(
                m, source, noise_freqs, noise_spec, args.duration, steps_per_mm_xy, steps_per_mm_z, scale_xy,
                mode=args.mode, max_xy_vel=args.max_xy_vel, max_z_vel=eff_max_z_vel,
            )
    finally:
        if audio_enabled:
            run_gcode("G90\n_AUDIO_MODE_OFF")
        else:
            run_gcode("G90")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mic_source": source,
        "microsteps_xy_config": orig_ms_xy,
        "audio_mode_4ms_1p7a": use_audio_mode,
        "noise_floor_rms_db": round(noise_rms, 1),
        "motors": motor_results,
    }
    CALIBRATION_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nCalibracao individual salva em: {CALIBRATION_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
