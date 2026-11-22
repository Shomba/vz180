#!/home/shoomba/klippy-env/bin/python
"""
Executa a macro G-Code MEGALOVANIA (3 eixos X, Y e Z com Input Shaper OFF) na VZ180,
grava todo o audio em tempo real pelo microfone embutido da webcam (48 kHz 16-bit)
e salva tanto o WAV original quanto o WAV com reducao espectral do ruido das ventoinhas.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft, istft

sys.path.insert(0, "/home/shoomba/printer_data/config/scripts")
import stepper_tuner as st

SAMPLE_RATE = 48000
OUT_WEB_DIR = Path("/home/shoomba/mainsail/soundfont")
OUT_CFG_DIR = Path("/home/shoomba/printer_data/config")
OUT_ART_DIR = Path("/home/shoomba/.gemini/antigravity/brain/e94ad702-76ca-41e0-93ff-43cc29dc95cd")


def save_wav(path: Path, samples: np.ndarray) -> None:
    pcm16 = np.clip(np.round(samples * 32767.0), -32767, 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())


def main() -> int:
    source = st.find_camera_mic()
    print(f"[Rec] Microfone: {source}", flush=True)

    # 1. Home and position at X90 Y87 Z25 BEFORE starting recording so G28 noise is excluded
    homed = st.moonraker_get("/printer/objects/query?toolhead")["result"]["status"]["toolhead"]["homed_axes"]
    if "x" not in homed or "y" not in homed or "z" not in homed:
        print("[Rec] Executando G28 previo...", flush=True)
        st.run_gcode("G28", timeout=90.0)

    print("[Rec] Posicionando em X90 Y87 Z25...", flush=True)
    st.run_gcode("G90\nSET_VELOCITY_LIMIT ACCEL=10000\nG1 X90 Y87 Z25 F7800\nM400")
    time.sleep(0.3)

    # 2. Capture 1.0s of fan noise floor for the denoised version
    print("[Rec] Capturando 1.0s de perfil das ventoinhas...", flush=True)
    noise_arr = st.record_audio(source, 1.0)
    nyq = SAMPLE_RATE / 2.0
    hp_sos = butter(4, 75.0 / nyq, btype="high", output="sos")
    noise_hp = sosfiltfilt(hp_sos, noise_arr - np.mean(noise_arr))
    _, _, Z_noise = stft(noise_hp, fs=SAMPLE_RATE, nperseg=2048, noverlap=1536)
    noise_mag = np.median(np.abs(Z_noise), axis=1)

    # 3. Start streaming parec capture while MEGALOVANIA executes
    chunks: list[bytes] = []
    stop_flag = threading.Event()

    def capture_worker() -> None:
        cmd = [
            "parec",
            "--device", source,
            "--format=s16le",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
            "--raw",
            "--latency-msec=25",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            while not stop_flag.is_set():
                data = proc.stdout.read(4800 * 2)  # 100 ms chunks
                if not data:
                    break
                chunks.append(data)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except Exception:
                proc.kill()

    t_cap = threading.Thread(target=capture_worker, daemon=True)
    t_cap.start()
    time.sleep(0.20)

    print("[Rec] Tocando macro MEGALOVANIA (3 Eixos X, Y, Z com Input Shaper OFF)...", flush=True)
    t0 = time.time()
    st.run_gcode("MEGALOVANIA\nM400", timeout=150.0)
    elapsed = time.time() - t0
    time.sleep(0.35)
    stop_flag.set()
    t_cap.join(timeout=3.0)

    raw_bytes = b"".join(chunks)
    raw_samples = np.frombuffer(raw_bytes[: len(raw_bytes) - (len(raw_bytes) % 2)], dtype=np.int16).astype(np.float64) / 32768.0
    dur_s = len(raw_samples) / SAMPLE_RATE
    print(f"[Rec] Gravacao concluida! Duracao: {dur_s:.2f}s (Execucao G-code: {elapsed:.2f}s)", flush=True)

    # High-pass raw slightly at 60 Hz & normalize to -1.0 dBFS peak
    hp60 = butter(4, 60.0 / nyq, btype="high", output="sos")
    raw_clean_dc = sosfiltfilt(hp60, raw_samples - np.mean(raw_samples))
    raw_norm = raw_clean_dc * (0.89 / float(np.max(np.abs(raw_clean_dc)) + 1e-9))

    # Spectral fan-noise reduction version
    filt = sosfiltfilt(hp_sos, raw_samples - np.mean(raw_samples))
    _, _, Zxx = stft(filt, fs=SAMPLE_RATE, nperseg=2048, noverlap=1536)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    clean_mag = np.maximum(mag - 1.25 * noise_mag[:, None], 0.08 * mag)
    _, denoised = istft(clean_mag * np.exp(1j * phase), fs=SAMPLE_RATE, nperseg=2048, noverlap=1536)
    denoised = denoised[: len(raw_samples)]
    denoised_norm = denoised * (0.89 / float(np.max(np.abs(denoised)) + 1e-9))

    raw_name = "MEGALOVANIA_3_Eixos_Bruto.wav"
    clean_name = "MEGALOVANIA_3_Eixos_Limpo.wav"

    for d in (OUT_WEB_DIR, OUT_CFG_DIR, OUT_ART_DIR):
        save_wav(d / raw_name, raw_norm)
        save_wav(d / clean_name, denoised_norm)

    # Also encode MP3 if ffmpeg is available for lightweight browser streaming
    if shutil.which("ffmpeg"):
        for stem in ("MEGALOVANIA_3_Eixos_Bruto", "MEGALOVANIA_3_Eixos_Limpo"):
            wav_p = OUT_WEB_DIR / f"{stem}.wav"
            mp3_p = OUT_WEB_DIR / f"{stem}.mp3"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(wav_p), "-b:a", "192k", str(mp3_p)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if mp3_p.exists():
                shutil.copy2(mp3_p, OUT_ART_DIR / mp3_p.name)
                shutil.copy2(mp3_p, OUT_CFG_DIR / mp3_p.name)

    print(f"[Rec] Arquivos salvos em {OUT_WEB_DIR}, {OUT_CFG_DIR} e {OUT_ART_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
