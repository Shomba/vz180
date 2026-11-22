#!/home/shoomba/klippy-env/bin/python
"""
Amostrador Acustico Automatico & Compilador de SoundFont (.sf2 / .sfz / .wav)
para os Eixos X, Y e Z da VZ180 AWD CoreXY.

1. Posiciona a impressora no centro seguro (X=90, Y=87, Z=25).
2. Captura 1.0s de perfil de ruido ambiente das ventoinhas via microfone USB (48 kHz).
3. Executa varreduras cromaticas reais (C3 a G5, MIDI 48 a 79 = 32 semitons por eixo)
   em blocos sincronizados de 8 notas para:
     - Eixo X (CoreXY Canal A, _AUDIO_MODE_COMPACTO_ON)
     - Eixo Y (CoreXY Canal B, _AUDIO_MODE_COMPACTO_ON)
     - Eixo Z (Vertical TMC2208, _AUDIO_MODE_Z_ON)
4. Segmenta cada nota via envelope espectral sintonizado na fundamental + harmonicos,
   remove o piso de ruido das ventoinhas via subtracao espectral suave, calcula pontos
   de loop de sustentacao (zero-crossing + correlacao de fase) sem cliques e salva:
     - Samples WAV 16-bit 48 kHz individuais (/home/shoomba/mainsail/soundfont/samples/{x,y,z}/)
     - SoundFont 2.01 completo (/home/shoomba/mainsail/soundfont/VZ180_Stepper_XYZ.sf2)
       com 4 Presets (000: Eixo X, 001: Eixo Y, 002: Eixo Z, 003: Conjunto Split XYZ)
     - Instrumentos .sfz + pacote .zip (/home/shoomba/mainsail/soundfont/VZ180_SoundFont_Bundle.zip)
     - Manifesto JSON (/home/shoomba/mainsail/soundfont/manifest.json) para o Simulador WebAudio no Mainsail.
"""

from __future__ import annotations

import io
import json
import math
import struct
import subprocess
import sys
import threading
import time
import wave
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft, istft

sys.path.insert(0, "/home/shoomba/printer_data/config/scripts")
import stepper_tuner as st

SAMPLE_RATE = 48000
MIDI_MIN = 48  # C3 (130.81 Hz)
MIDI_MAX = 79  # G5 (783.99 Hz)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

OUT_DIR = Path("/home/shoomba/mainsail/soundfont")
SAMPLES_DIR = OUT_DIR / "samples"
SF2_PATH = OUT_DIR / "VZ180_Stepper_XYZ.sf2"
CFG_SF2_PATH = Path("/home/shoomba/printer_data/config/VZ180_Stepper_XYZ.sf2")
ZIP_PATH = OUT_DIR / "VZ180_SoundFont_Bundle.zip"
MANIFEST_PATH = OUT_DIR / "manifest.json"

PULSES_PER_CMD_MM_XY = 80.0
PULSES_PER_CMD_MM_Z = 1200.00075


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_safe_name(midi: int) -> str:
    return midi_to_name(midi).replace("#", "s")


def record_raw_48k(source: str, duration_s: float) -> np.ndarray:
    n_samples = int(SAMPLE_RATE * duration_s)
    n_bytes = n_samples * 2
    cmd = [
        "parec",
        "--device", source,
        "--format=s16le",
        f"--rate={SAMPLE_RATE}",
        "--channels=1",
        "--raw",
        "--latency-msec=20",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        raw = proc.stdout.read(n_bytes)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except Exception:
            proc.kill()
    if not raw or len(raw) < 2:
        return np.zeros(n_samples, dtype=np.float64)
    arr = np.frombuffer(raw[: len(raw) - (len(raw) % 2)], dtype=np.int16).astype(np.float64) / 32768.0
    return arr


def spectral_denoise(
    audio: np.ndarray,
    noise_mag_profile: np.ndarray,
    hp_sos: np.ndarray,
    over_subtract: float = 1.20,
    floor_gain: float = 0.10,
) -> np.ndarray:
    """Gentle STFT spectral subtraction that removes continuous fan noise while preserving motor harmonics."""
    filtered = sosfiltfilt(hp_sos, audio - np.mean(audio))
    f, t, Zxx = stft(filtered, fs=SAMPLE_RATE, nperseg=1024, noverlap=768)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    clean_mag = np.maximum(mag - over_subtract * noise_mag_profile[:, None], floor_gain * mag)
    _, rec = istft(clean_mag * np.exp(1j * phase), fs=SAMPLE_RATE, nperseg=1024, noverlap=768)
    return rec[: len(audio)]


def calc_actual_note_dur(axis: str, midi: int, note_dur_s: float) -> float:
    hz = midi_to_hz(midi)
    if axis in ("x", "y"):
        d_cmd = (hz / PULSES_PER_CMD_MM_XY) * note_dur_s
        dist = math.hypot(0.5 * d_cmd, 0.5 * d_cmd)
        accel = 3000.0
    else:
        dist = (hz / PULSES_PER_CMD_MM_Z) * note_dur_s
        accel = 60.0
    v = dist / note_dur_s
    if (v * v) / accel <= dist:
        return note_dur_s + (v / accel)
    return 2.0 * math.sqrt(dist / accel)


def slice_batch_notes(
    axis: str,
    batch_audio: np.ndarray,
    midis: list[int],
    lead_in_s: float,
    note_dur_s: float,
    gap_s: float,
    noise_mag_profile: np.ndarray,
    hp_sos: np.ndarray,
) -> list[tuple[int, np.ndarray, dict]]:
    """
    Locate each note in a recorded batch using high-resolution narrow-band (+/-1.8%, +/-30 cents)
    spectral energy at target_hz (zero overlap with adjacent semitones!) + monotonic sequence constraint.
    """
    clean_full = spectral_denoise(batch_audio, noise_mag_profile, hp_sos)
    win_size = 2048
    n_fft_track = 8192  # 5.86 Hz bin resolution @ 48 kHz
    hop = 240           # 5 ms hop @ 48 kHz
    hann = np.hanning(win_size)
    fft_freqs = np.fft.rfftfreq(n_fft_track, d=1.0 / SAMPLE_RATE)

    n_frames = max(1, (len(clean_full) - win_size) // hop)
    spec_matrix = np.zeros((n_frames, len(fft_freqs)), dtype=np.float64)
    for f_i in range(n_frames):
        frame = clean_full[f_i * hop : f_i * hop + win_size] * hann
        spec_matrix[f_i] = np.abs(np.fft.rfft(frame, n=n_fft_track))

    # Subtract median background spectrum across the batch so continuous fan tones vanish
    bg_spec = np.percentile(spec_matrix, 25, axis=0)
    fg_matrix = np.maximum(spec_matrix - 1.35 * bg_spec[None, :], 0.0)

    box_frames = max(4, int((note_dur_s * 0.78) * SAMPLE_RATE / hop))
    box_kernel = np.ones(box_frames, dtype=np.float64)
    min_step_frames = int((note_dur_s + gap_s * 0.65) * SAMPLE_RATE / hop)
    nom_step_frames = int((note_dur_s + gap_s) * SAMPLE_RATE / hop)

    results = []
    prev_onset_frame: int | None = None

    for idx, midi in enumerate(midis):
        target_hz = midi_to_hz(midi)
        # Strict +/-1.8% (+/-31 cents) mask around fundamental & 2nd harmonic — never overlaps adjacent semitones (+/-5.9%)!
        h_mask = np.zeros_like(fft_freqs, dtype=bool)
        for mult, weight in ((1.0, 1.0), (2.0, 0.45)):
            hf = target_hz * mult
            if hf < SAMPLE_RATE * 0.45:
                h_mask |= (fft_freqs >= hf * 0.982) & (fft_freqs <= hf * 1.018)

        note_energy = np.sum(fg_matrix[:, h_mask] ** 2, axis=1)
        if len(note_energy) > box_frames:
            conv = np.convolve(note_energy, box_kernel, mode="valid")
        else:
            conv = note_energy

        if prev_onset_frame is None:
            # Note 0 must start within the first 1.1s of the recording
            f_lo = 0
            f_hi = min(len(conv) - 1, int(1.10 * SAMPLE_RATE / hop))
        else:
            # Subsequent note idx must start roughly 1 stride after prev_onset_frame
            f_lo = max(prev_onset_frame + min_step_frames, prev_onset_frame + nom_step_frames - int(0.14 * SAMPLE_RATE / hop))
            f_hi = min(len(conv) - 1, prev_onset_frame + nom_step_frames + int(0.22 * SAMPLE_RATE / hop))

        if f_lo >= len(conv):
            f_lo = max(0, len(conv) - 1)
        if f_hi < f_lo:
            f_hi = f_lo

        best_frame = f_lo + int(np.argmax(conv[f_lo : f_hi + 1]))
        prev_onset_frame = best_frame
        onset_sample = best_frame * hop + int(0.015 * SAMPLE_RATE)

        # Extract note_dur_s sample starting slightly inside onset
        want_len = int(round(note_dur_s * SAMPLE_RATE))
        start_idx = max(0, min(len(clean_full) - want_len, onset_sample))
        note_wav = clean_full[start_idx : start_idx + want_len].copy()

        # Measure acoustic pitch & fundamental clarity on the steady middle 70%
        mid0 = int(len(note_wav) * 0.15)
        mid1 = int(len(note_wav) * 0.85)
        core = note_wav[mid0:mid1]
        rms = float(np.sqrt(np.mean(core ** 2)) + 1e-9)
        rms_db = round(20.0 * math.log10(rms), 1)

        n_fft_meas = 32768
        sp_meas = np.abs(np.fft.rfft(core * np.hanning(len(core)), n=n_fft_meas))
        f_meas = np.fft.rfftfreq(n_fft_meas, d=1.0 / SAMPLE_RATE)

        # Measure energy near fundamental (or dominant harmonic)
        f1_mask = (f_meas >= target_hz * 0.93) & (f_meas <= target_hz * 1.07)
        if np.any(f1_mask):
            f1_indices = np.where(f1_mask)[0]
            p_idx = int(f1_indices[int(np.argmax(sp_meas[f1_indices]))])
            meas_hz = float(f_meas[p_idx])
        else:
            meas_hz = target_hz
        cents_err = round(1200.0 * math.log2(max(meas_hz, 1.0) / target_hz), 1)

        # Normalize sample with smooth 6 ms fade-in and 8 ms fade-out, preserving natural timbre
        peak = float(np.max(np.abs(note_wav)) + 1e-9)
        target_peak = min(0.88, max(0.35, rms * 3.8))
        note_wav = note_wav * (target_peak / peak)

        fade_in = int(0.005 * SAMPLE_RATE)
        fade_out = int(0.007 * SAMPLE_RATE)
        note_wav[:fade_in] *= np.linspace(0.0, 1.0, fade_in)
        note_wav[-fade_out:] *= np.linspace(1.0, 0.0, fade_out)

        # Find seamless sustain loop points aligned to exact multiple of fundamental period
        loop_start, loop_end = find_seamless_loop(note_wav, target_hz)

        meta = {
            "midi": midi,
            "note": midi_to_name(midi),
            "target_hz": round(target_hz, 2),
            "measured_hz": round(meas_hz, 2),
            "cents_error": cents_err,
            "rms_db": rms_db,
            "duration_s": round(len(note_wav) / SAMPLE_RATE, 4),
            "loop_start": int(loop_start),
            "loop_end": int(loop_end),
            "loop_start_s": round(loop_start / SAMPLE_RATE, 4),
            "loop_end_s": round(loop_end / SAMPLE_RATE, 4),
        }
        results.append((midi, note_wav, meta))
    return results


def find_seamless_loop(wav_arr: np.ndarray, target_hz: float) -> tuple[int, int]:
    """
    Find positive zero-crossing loop_start (~20% into sample) and loop_end (~78% into sample)
    separated by an integer number of fundamental cycles, and apply a 24-sample loop crossfade.
    """
    n = len(wav_arr)
    period_samples = SAMPLE_RATE / max(target_hz, 40.0)

    # Search for positive-going zero crossing near 20% of sample
    s_search0 = int(n * 0.18)
    s_search1 = int(n * 0.32)
    loop_start = s_search0
    best_slope = -1.0
    for i in range(s_search0, min(s_search1, n - 2)):
        if wav_arr[i] <= 0.0 and wav_arr[i + 1] > 0.0:
            slope = float(wav_arr[i + 1] - wav_arr[i])
            if slope > best_slope:
                best_slope = slope
                loop_start = i + 1
                break

    # Number of whole fundamental cycles that fit between ~20% and ~78%
    target_span = int(n * 0.56)
    n_cycles = max(4, int(round(target_span / period_samples)))
    ideal_end = loop_start + int(round(n_cycles * period_samples))
    e_search0 = max(loop_start + int(period_samples * 2), ideal_end - int(period_samples * 1.5))
    e_search1 = min(n - int(0.02 * SAMPLE_RATE), ideal_end + int(period_samples * 1.5))

    # Pick zero-crossing in [e_search0, e_search1] with highest waveform correlation to loop_start
    win_corr = min(96, int(period_samples * 0.8))
    ref_vec = wav_arr[loop_start : loop_start + win_corr]
    ref_norm = float(np.linalg.norm(ref_vec) + 1e-9)

    loop_end = min(n - 64, max(loop_start + 200, ideal_end))
    best_corr = -2.0
    for i in range(e_search0, max(e_search0 + 1, e_search1 - win_corr)):
        if wav_arr[i] <= 0.0 and wav_arr[i + 1] > 0.0:
            cand_vec = wav_arr[i + 1 : i + 1 + win_corr]
            if len(cand_vec) == len(ref_vec):
                corr = float(np.dot(ref_vec, cand_vec)) / (ref_norm * float(np.linalg.norm(cand_vec) + 1e-9))
                if corr > best_corr:
                    best_corr = corr
                    loop_end = i + 1

    # Blend 24 samples before loop_end with 24 samples before loop_start for zero-click looping
    xf = 24
    if loop_start >= xf and loop_end >= loop_start + xf * 2:
        alpha = np.linspace(0.0, 1.0, xf)
        wav_arr[loop_end - xf : loop_end] = (
            (1.0 - alpha) * wav_arr[loop_end - xf : loop_end]
            + alpha * wav_arr[loop_start - xf : loop_start]
        )

    return loop_start, loop_end


def save_wav_16bit(path: Path, samples: np.ndarray) -> np.ndarray:
    pcm16 = np.clip(np.round(samples * 32767.0), -32767, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    return pcm16


def build_axis_batch_gcode(axis: str, midis: list[int], note_dur_s: float, gap_s: float) -> str:
    """
    Build relative G91 G-code for an 8-note chromatic batch on axis ('x', 'y', or 'z'),
    alternating direction (+1 / -1) so the carriage oscillates tightly around center.
    """
    lines = ["G91"]
    cum_dx = 0.0
    cum_dy = 0.0
    cum_dz = 0.0
    direction = 1
    gap_ms = int(round(gap_s * 1000))

    if axis in ("x", "y"):
        lines.append("SET_VELOCITY_LIMIT ACCEL=3000")
    else:
        lines.append("SET_VELOCITY_LIMIT ACCEL=60")

    for midi in midis:
        hz = midi_to_hz(midi)
        if axis == "x":
            # CoreXY Motor A (+45 deg): dX = dY = dA / 2
            dA_cmd = direction * (hz / PULSES_PER_CMD_MM_XY) * note_dur_s
            dx = 0.5 * dA_cmd
            dy = 0.5 * dA_cmd
            dist = math.hypot(dx, dy)
            feed = (dist / note_dur_s) * 60.0
            lines.append(f"G1 X{dx:.4f} Y{dy:.4f} F{feed:.1f}")
            cum_dx += dx
            cum_dy += dy
        elif axis == "y":
            # CoreXY Motor B (-45 deg): dX = dB / 2, dY = -dB / 2
            dB_cmd = direction * (hz / PULSES_PER_CMD_MM_XY) * note_dur_s
            dx = 0.5 * dB_cmd
            dy = -0.5 * dB_cmd
            dist = math.hypot(dx, dy)
            feed = (dist / note_dur_s) * 60.0
            lines.append(f"G1 X{dx:.4f} Y{dy:.4f} F{feed:.1f}")
            cum_dx += dx
            cum_dy += dy
        elif axis == "z":
            dz = direction * (hz / PULSES_PER_CMD_MM_Z) * note_dur_s
            feed = (abs(dz) / note_dur_s) * 60.0
            lines.append(f"G1 Z{dz:.5f} F{feed:.1f}")
            cum_dz += dz

        lines.append(f"G4 P{gap_ms}")
        direction = -direction

    # Return to exact batch origin
    if abs(cum_dx) > 1e-5 or abs(cum_dy) > 1e-5:
        lines.append(f"G1 X{-cum_dx:.4f} Y{-cum_dy:.4f} F1800")
    if abs(cum_dz) > 1e-6:
        lines.append(f"G1 Z{-cum_dz:.5f} F120")
    lines.append("G90")
    lines.append("M400")
    return "\n".join(lines)


def riff_chunk(fourcc: bytes, payload: bytes) -> bytes:
    pad = b"\x00" if (len(payload) % 2 == 1) else b""
    return fourcc + struct.pack("<I", len(payload)) + payload + pad


def riff_list(list_type: bytes, payload: bytes) -> bytes:
    return riff_chunk(b"LIST", list_type + payload)


def zstr(text: str, fixed_len: int | None = None) -> bytes:
    b = text.encode("ascii", errors="replace")
    if fixed_len is not None:
        return b[: fixed_len - 1].ljust(fixed_len, b"\x00")
    b_null = b + b"\x00"
    if len(b_null) % 2 == 1:
        b_null += b"\x00"
    return b_null


def compile_sf2(
    axis_samples: dict[str, list[tuple[int, np.ndarray, dict]]],
    out_sf2_path: Path,
) -> None:
    """
    Compile a complete SoundFont 2.01 (.sf2) binary file containing:
      - Preset 000 (Bank 0, Prog 0): VZ180 Eixo X (Motor A)
      - Preset 001 (Bank 0, Prog 1): VZ180 Eixo Y (Motor B)
      - Preset 002 (Bank 0, Prog 2): VZ180 Eixo Z (Vertical)
      - Preset 003 (Bank 0, Prog 3): VZ180 Conjunto XYZ (Split Z/X/Y)
      - Preset 004 (Bank 0, Prog 4): VZ180 Ensemble 3 Eixos (Layered X+Y+Z)
    """
    # 1. INFO-list chunk
    ifil = riff_chunk(b"ifil", struct.pack("<HH", 2, 1))
    isng = riff_chunk(b"isng", zstr("EMU8000"))
    inam = riff_chunk(b"INAM", zstr("VZ180 AWD CoreXY + Z Stepper SoundFont"))
    ieng = riff_chunk(b"IENG", zstr("VZ180 Acoustic Sampler"))
    icmt = riff_chunk(
        b"ICMT",
        zstr("Amostras acusticas reais (48kHz 16-bit) de cada frequencia dos eixos X, Y e Z da impressora 3D VZ180."),
    )
    info_list = riff_list(b"INFO", ifil + isng + inam + ieng + icmt)

    # 2. Build smpl pool & shdr records
    smpl_pcm_parts: list[bytes] = []
    shdr_records: list[bytes] = []
    sample_index_map: dict[tuple[str, int], int] = {}
    current_sample_offset = 0  # in 16-bit sample words

    for axis in ("x", "y", "z"):
        for midi, wav_float, meta in axis_samples[axis]:
            pcm16 = np.clip(np.round(wav_float * 32767.0), -32767, 32767).astype(np.int16)
            n_words = len(pcm16)
            pad46 = np.zeros(48, dtype=np.int16)  # >= 46 zero words required by SF2 spec
            smpl_pcm_parts.append(pcm16.tobytes() + pad46.tobytes())

            start_w = current_sample_offset
            end_w = current_sample_offset + n_words
            loop_start_w = current_sample_offset + int(meta["loop_start"])
            loop_end_w = current_sample_offset + int(meta["loop_end"])
            current_sample_offset += n_words + len(pad46)

            sname = f"VZ_{axis.upper()}_{midi}_{midi_to_safe_name(midi)}"
            # Fine pitch correction in cents if measured_hz differs slightly, capped to [-50, 50]
            pitch_corr = int(max(-50, min(50, round(-meta["cents_error"]))))

            # sfSample struct: 20s IIIII I b b H H (46 bytes)
            shdr_rec = struct.pack(
                "<20sIIIIIBbHH",
                zstr(sname, 20),
                start_w,
                end_w,
                loop_start_w,
                loop_end_w,
                SAMPLE_RATE,
                midi,
                pitch_corr,
                0,
                1,  # monoSample
            )
            sample_idx = len(shdr_records)
            shdr_records.append(shdr_rec)
            sample_index_map[(axis, midi)] = sample_idx

    # Terminal EOS shdr record
    shdr_records.append(struct.pack("<20sIIIIIBbHH", zstr("EOS", 20), 0, 0, 0, 0, 0, 0, 0, 0, 0))
    sdta_list = riff_list(b"sdta", riff_chunk(b"smpl", b"".join(smpl_pcm_parts)))

    # 3. Build Instruments (inst, ibag, imod, igen)
    # Instruments:
    #   0: VZ180_Eixo_X
    #   1: VZ180_Eixo_Y
    #   2: VZ180_Eixo_Z
    #   3: VZ180_Split_XYZ (C1..B3 -> Z, C4..B4 -> X, C5..C7 -> Y)
    inst_defs = [
        ("VZ180 Eixo X", [("x", m) for m, _, _ in axis_samples["x"]]),
        ("VZ180 Eixo Y", [("y", m) for m, _, _ in axis_samples["y"]]),
        ("VZ180 Eixo Z", [("z", m) for m, _, _ in axis_samples["z"]]),
        (
            "VZ180 Split XYZ",
            [("z", m) for m, _, _ in axis_samples["z"] if m < 60]
            + [("x", m) for m, _, _ in axis_samples["x"] if 60 <= m < 72]
            + [("y", m) for m, _, _ in axis_samples["y"] if m >= 72],
        ),
    ]

    inst_records: list[bytes] = []
    ibag_records: list[bytes] = []
    igen_records: list[bytes] = []

    # SF2 Generator IDs:
    # 38 = releaseVolEnv (timecents: 1200 * log2(seconds); -3200 ~= 0.15s)
    # 43 = keyRange (lo | (hi << 8))
    # 53 = sampleID
    # 54 = sampleModes (1 = loop continuously during key press)
    # 58 = overridingRootKey
    for inst_name, zones in inst_defs:
        inst_bag_idx = len(ibag_records)
        inst_records.append(struct.pack("<20sH", zstr(inst_name, 20), inst_bag_idx))

        sorted_zones = sorted(zones, key=lambda z: z[1])
        for z_i, (ax, midi) in enumerate(sorted_zones):
            lo_key = 0 if z_i == 0 else sorted_zones[z_i - 1][1] + 1
            hi_key = 127 if z_i == len(sorted_zones) - 1 else midi
            if lo_key > hi_key:
                lo_key = hi_key

            gen_idx = len(igen_records)
            ibag_records.append(struct.pack("<HH", gen_idx, 0))

            samp_id = sample_index_map[(ax, midi)]
            # keyRange generator (must be first in zone!)
            igen_records.append(struct.pack("<HH", 43, (lo_key & 0xFF) | ((hi_key & 0xFF) << 8)))
            # sampleModes = 1 (continuous loop)
            igen_records.append(struct.pack("<HH", 54, 1))
            # releaseVolEnv = -3400 timecents (~140 ms natural stepper decay)
            igen_records.append(struct.pack("<Hh", 38, -3400))
            # overridingRootKey = midi
            igen_records.append(struct.pack("<HH", 58, midi))
            # sampleID (must be last in zone!)
            igen_records.append(struct.pack("<HH", 53, samp_id))

    # Terminal EOI records
    inst_records.append(struct.pack("<20sH", zstr("EOI", 20), len(ibag_records)))
    ibag_records.append(struct.pack("<HH", len(igen_records), 0))
    igen_records.append(struct.pack("<HH", 0, 0))
    imod_records = [struct.pack("<HHhHH", 0, 0, 0, 0, 0)]

    # 4. Build Presets (phdr, pbag, pmod, pgen)
    # Presets:
    #   Prog 0: Eixo X (Motor A) -> inst 0
    #   Prog 1: Eixo Y (Motor B) -> inst 1
    #   Prog 2: Eixo Z (Vertical) -> inst 2
    #   Prog 3: Conjunto Split XYZ -> inst 3
    #   Prog 4: Ensemble 3 Eixos XYZ -> layered inst 0 + inst 1 + inst 2
    preset_defs = [
        ("Eixo X (CoreXY A)", 0, 0, [0]),
        ("Eixo Y (CoreXY B)", 1, 0, [1]),
        ("Eixo Z (Vertical)", 2, 0, [2]),
        ("Conjunto Split XYZ", 3, 0, [3]),
        ("Ensemble 3 Eixos XYZ", 4, 0, [0, 1, 2]),
    ]

    phdr_records: list[bytes] = []
    pbag_records: list[bytes] = []
    pgen_records: list[bytes] = []

    for pname, prog_num, bank_num, inst_ids in preset_defs:
        pbag_idx = len(pbag_records)
        phdr_records.append(
            struct.pack("<20sHHHIII", zstr(pname, 20), prog_num, bank_num, pbag_idx, 0, 0, 0)
        )
        for inst_id in inst_ids:
            pgen_idx = len(pgen_records)
            pbag_records.append(struct.pack("<HH", pgen_idx, 0))
            # Generator 41 = instrument
            pgen_records.append(struct.pack("<HH", 41, inst_id))

    # Terminal EOP records
    phdr_records.append(
        struct.pack("<20sHHHIII", zstr("EOP", 20), 255, 255, len(pbag_records), 0, 0, 0)
    )
    pbag_records.append(struct.pack("<HH", len(pgen_records), 0))
    pgen_records.append(struct.pack("<HH", 0, 0))
    pmod_records = [struct.pack("<HHhHH", 0, 0, 0, 0, 0)]

    pdta_payload = (
        riff_chunk(b"phdr", b"".join(phdr_records))
        + riff_chunk(b"pbag", b"".join(pbag_records))
        + riff_chunk(b"pmod", b"".join(pmod_records))
        + riff_chunk(b"pgen", b"".join(pgen_records))
        + riff_chunk(b"inst", b"".join(inst_records))
        + riff_chunk(b"ibag", b"".join(ibag_records))
        + riff_chunk(b"imod", b"".join(imod_records))
        + riff_chunk(b"igen", b"".join(igen_records))
        + riff_chunk(b"shdr", b"".join(shdr_records))
    )
    pdta_list = riff_list(b"pdta", pdta_payload)

    sfbk_riff = riff_chunk(b"RIFF", b"sfbk" + info_list + sdta_list + pdta_list)
    out_sf2_path.parent.mkdir(parents=True, exist_ok=True)
    out_sf2_path.write_bytes(sfbk_riff)


def write_sfz_files(axis_samples: dict[str, list[tuple[int, np.ndarray, dict]]]) -> list[Path]:
    sfz_paths = []
    titles = {
        "x": "VZ180_Eixo_X_CoreXY_A",
        "y": "VZ180_Eixo_Y_CoreXY_B",
        "z": "VZ180_Eixo_Z_Vertical",
    }
    for axis, sfz_name in titles.items():
        lines = [
            f"// {sfz_name} — Amostras Acusticas Reais 48kHz 16-bit da Impressora 3D VZ180",
            "<group>",
            "loop_mode=loop_continuous",
            "ampeg_release=0.14",
            "",
        ]
        items = sorted(axis_samples[axis], key=lambda item: item[0])
        for idx, (midi, _, meta) in enumerate(items):
            lokey = 0 if idx == 0 else items[idx - 1][0] + 1
            hikey = 127 if idx == len(items) - 1 else midi
            fname = f"samples/{axis}/{axis}_{midi}_{midi_to_safe_name(midi)}.wav"
            lines.append(
                f"<region> sample={fname} pitch_keycenter={midi} lokey={lokey} hikey={hikey} "
                f"loop_start={meta['loop_start']} loop_end={meta['loop_end']}"
            )
        sfz_path = OUT_DIR / f"{sfz_name}.sfz"
        sfz_path.write_text("\n".join(lines) + "\n")
        sfz_paths.append(sfz_path)
    return sfz_paths


def main() -> int:
    source = st.find_camera_mic()
    print(f"[Sampler] Microfone USB detectado: {source}", flush=True)

    info = st.moonraker_get("/printer/info")["result"]
    if info["state"] != "ready":
        print(f"ERRO: Klipper nao esta ready (state={info['state']})", flush=True)
        return 1

    # Home if needed and move to acoustic center (X=90, Y=87, Z=25)
    homed = st.moonraker_get("/printer/objects/query?toolhead")["result"]["status"]["toolhead"]["homed_axes"]
    if "x" not in homed or "y" not in homed or "z" not in homed:
        print("[Sampler] Executando G28...", flush=True)
        st.run_gcode("G28", timeout=90.0)

    print("[Sampler] Posicionando em X=90 Y=87 Z=25 e medindo perfil de ruido das ventoinhas (1.0s)...", flush=True)
    st.run_gcode("G90\nSET_VELOCITY_LIMIT ACCEL=10000\nG1 X90 Y87 Z25 F7800\nM400")
    time.sleep(0.25)

    noise_audio = record_raw_48k(source, 1.0)
    nyq = SAMPLE_RATE / 2.0
    hp_sos = butter(4, 75.0 / nyq, btype="high", output="sos")
    noise_hp = sosfiltfilt(hp_sos, noise_audio - np.mean(noise_audio))
    _, _, Z_noise = stft(noise_hp, fs=SAMPLE_RATE, nperseg=1024, noverlap=768)
    noise_mag_profile = np.median(np.abs(Z_noise), axis=1)
    noise_rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(noise_hp ** 2))) + 1e-9)
    print(f"[Sampler] Ruido de fundo das ventoinhas: {noise_rms_db:.1f} dBFS", flush=True)

    all_midis = list(range(MIDI_MIN, MIDI_MAX + 1))  # 32 notes: C3 (48) .. G5 (79)
    batches = [all_midis[i : i + 8] for i in range(0, len(all_midis), 8)]
    note_dur_s = 0.28
    gap_s = 0.12
    lead_in_s = 0.14

    axis_samples: dict[str, list[tuple[int, np.ndarray, dict]]] = {"x": [], "y": [], "z": []}
    manifest_axes: dict[str, dict[str, dict]] = {"x": {}, "y": {}, "z": {}}

    try:
        # Activate Audio Mode on XY and Z
        st.run_gcode("_AUDIO_MODE_COMPACTO_ON\n_AUDIO_MODE_Z_ON\nM400")
        time.sleep(0.15)

        for axis, axis_label in [
            ("x", "Eixo X (CoreXY Canal A +45 deg)"),
            ("y", "Eixo Y (CoreXY Canal B -45 deg)"),
            ("z", "Eixo Z (Vertical TMC2208)"),
        ]:
            print(f"\n[Sampler] === Gravando 32 Semitons de {axis_label} (C3 a G5) ===", flush=True)
            for b_idx, batch_midis in enumerate(batches):
                n_notes = len(batch_midis)
                gcode = build_axis_batch_gcode(axis, batch_midis, note_dur_s, gap_s)
                rec_dur = lead_in_s + n_notes * (note_dur_s + gap_s) + 0.45

                rec_holder: list[np.ndarray] = []
                t_rec = threading.Thread(
                    target=lambda: rec_holder.append(record_raw_48k(source, rec_dur)),
                    daemon=True,
                )
                t_rec.start()
                time.sleep(lead_in_s)
                st.run_gcode(gcode, timeout=25.0)
                t_rec.join(timeout=rec_dur + 3.0)

                sliced = slice_batch_notes(
                    axis,
                    rec_holder[0],
                    batch_midis,
                    lead_in_s,
                    note_dur_s,
                    gap_s,
                    noise_mag_profile,
                    hp_sos,
                )
                for midi, wav_arr, meta in sliced:
                    safe_n = midi_to_safe_name(midi)
                    rel_wav = f"samples/{axis}/{axis}_{midi}_{safe_n}.wav"
                    wav_path = OUT_DIR / rel_wav
                    save_wav_16bit(wav_path, wav_arr)
                    meta["wav_url"] = f"/soundfont/{rel_wav}"
                    axis_samples[axis].append((midi, wav_arr, meta))
                    manifest_axes[axis][str(midi)] = meta

                summary_str = ", ".join(
                    f"{m['note']}:{m['measured_hz']:.0f}Hz({m['rms_db']:.1f}dB)"
                    for _, _, m in sliced
                )
                print(f"  Bloco {b_idx + 1}/4 ({midi_to_name(batch_midis[0])}..{midi_to_name(batch_midis[-1])}): {summary_str}", flush=True)

            # Ensure exact center position after each axis
            st.run_gcode("G90\nSET_VELOCITY_LIMIT ACCEL=3000\nG1 X90 Y87 Z25 F3600\nM400")
            time.sleep(0.15)

    finally:
        print("\n[Sampler] Restaurando drivers TMC e aceleracao normal...", flush=True)
        st.run_gcode("G90\n_AUDIO_MODE_Z_OFF\n_AUDIO_MODE_OFF\nSET_VELOCITY_LIMIT ACCEL=25000\nM400")

    # Compile .sf2 SoundFont
    print(f"[Sampler] Compilando SoundFont 2.01: {SF2_PATH}", flush=True)
    compile_sf2(axis_samples, SF2_PATH)
    CFG_SF2_PATH.write_bytes(SF2_PATH.read_bytes())

    # Write .sfz files & bundle .zip
    sfz_paths = write_sfz_files(axis_samples)
    print(f"[Sampler] Criando pacote ZIP com .sf2 + .sfz + 96 samples WAV: {ZIP_PATH}", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(SF2_PATH, arcname=SF2_PATH.name)
        for sfz_p in sfz_paths:
            zf.write(sfz_p, arcname=sfz_p.name)
        for axis in ("x", "y", "z"):
            for midi, _, _ in axis_samples[axis]:
                safe_n = midi_to_safe_name(midi)
                p = OUT_DIR / f"samples/{axis}/{axis}_{midi}_{safe_n}.wav"
                zf.write(p, arcname=f"samples/{axis}/{p.name}")

    # Save manifest.json for WebAudio Virtual Instrument & G-Code Simulator
    manifest = {
        "title": "VZ180 AWD CoreXY + Z Stepper SoundFont",
        "sample_rate": SAMPLE_RATE,
        "midi_min": MIDI_MIN,
        "midi_max": MIDI_MAX,
        "note_min": midi_to_name(MIDI_MIN),
        "note_max": midi_to_name(MIDI_MAX),
        "noise_floor_db": round(noise_rms_db, 1),
        "sf2_url": "/soundfont/VZ180_Stepper_XYZ.sf2",
        "sf2_size_kb": round(SF2_PATH.stat().st_size / 1024.0, 1),
        "zip_url": "/soundfont/VZ180_SoundFont_Bundle.zip",
        "zip_size_kb": round(ZIP_PATH.stat().st_size / 1024.0, 1),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "axes": manifest_axes,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(
        f"[Sampler] Concluido! 96 amostras gravadas | SF2: {manifest['sf2_size_kb']} KB | ZIP: {manifest['zip_size_kb']} KB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
