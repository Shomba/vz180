#!/home/shoomba/klippy-env/bin/python
"""
Reinicia o Klipper com [controller_fan steppers] em 60% (max_power: 0.60),
faz home + posicionamento previo em (132.43, 129.43, 25.0), executa a macro
YOUNG_GIRL_A (Meia-Lua 60mm 1:30->7:30 + Ponte 3 Oitavas) e grava o audio
usando o MICROFONE EMBUTIDO DO NOTEBOOK (alsa_input.pci-0000_05_00.6.analog-stereo).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft, istft

sys.path.insert(0, "/home/shoomba/printer_data/config/scripts")
import stepper_tuner as st

SAMPLE_RATE = 48000
LAPTOP_MIC = "alsa_input.pci-0000_05_00.6.analog-stereo"
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
    print("[Rec] Reiniciando Klipper para aplicar ventoinha dos steppers em 60%...", flush=True)
    req = urllib.request.Request("http://127.0.0.1:7125/printer/gcode/script?script=FIRMWARE_RESTART", method="POST")
    urllib.request.urlopen(req, timeout=10).read()
    time.sleep(3.5)
    for _ in range(15):
        try:
            if st.moonraker_get("/printer/info")["result"]["state"] == "ready":
                break
        except Exception:
            pass
        time.sleep(1.0)

    print(f"[Rec] Microfone embutido do notebook: {LAPTOP_MIC}", flush=True)

    # 1. Home and position at 1:30 corner (X=132.43, Y=129.43, Z=25.0) BEFORE starting recording
    print("[Rec] Fazendo G28 previo e posicionando em X132.43 Y129.43 Z25.0...", flush=True)
    st.run_gcode("G28", timeout=90.0)
    st.run_gcode("G90\nSET_VELOCITY_LIMIT ACCEL=10000\nG1 X132.430 Y129.430 Z25.000 F7800\nM400")
    time.sleep(0.5)

    # 2. Capture 1.0s ambient noise floor from laptop mic
    print("[Rec] Capturando 1.0s de ruido ambiente no microfone do notebook...", flush=True)
    noise_arr = st.record_audio(LAPTOP_MIC, 1.0)
    nyq = SAMPLE_RATE / 2.0
    hp_sos = butter(4, 75.0 / nyq, btype="high", output="sos")
    noise_hp = sosfiltfilt(hp_sos, noise_arr - np.mean(noise_arr))
    _, _, Z_noise = stft(noise_hp, fs=SAMPLE_RATE, nperseg=2048, noverlap=1536)
    noise_mag = np.median(np.abs(Z_noise), axis=1)

    # 3. Start streaming parec capture on laptop mic while YOUNG_GIRL_A plays
    chunks: list[bytes] = []
    stop_flag = threading.Event()

    def capture_worker() -> None:
        cmd = [
            "parec",
            "--device", LAPTOP_MIC,
            "--format=s16le",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
            "--raw",
            "--latency-msec=25",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            while not stop_flag.is_set():
                data = proc.stdout.read(4800 * 2)
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
    time.sleep(0.25)

    print("[Rec] Executando macro YOUNG_GIRL_A (103.5s, 3 Eixos + Meia-Lua 60mm)...", flush=True)
    t0 = time.time()
    st.run_gcode("YOUNG_GIRL_A\nM400", timeout=200.0)
    elapsed = time.time() - t0
    time.sleep(0.4)
    stop_flag.set()
    t_cap.join(timeout=3.0)

    raw_bytes = b"".join(chunks)
    raw_samples = np.frombuffer(raw_bytes[: len(raw_bytes) - (len(raw_bytes) % 2)], dtype=np.int16).astype(np.float64) / 32768.0
    dur_s = len(raw_samples) / SAMPLE_RATE
    print(f"[Rec] Gravacao concluida! Duracao: {dur_s:.2f}s (Execucao G-code: {elapsed:.2f}s)", flush=True)

    hp60 = butter(4, 60.0 / nyq, btype="high", output="sos")
    raw_dc = sosfiltfilt(hp60, raw_samples - np.mean(raw_samples))
    raw_norm = raw_dc * (0.89 / float(np.max(np.abs(raw_dc)) + 1e-9))

    filt = sosfiltfilt(hp_sos, raw_samples - np.mean(raw_samples))
    _, _, Zxx = stft(filt, fs=SAMPLE_RATE, nperseg=2048, noverlap=1536)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    clean_mag = np.maximum(mag - 1.15 * noise_mag[:, None], 0.10 * mag)
    _, denoised = istft(clean_mag * np.exp(1j * phase), fs=SAMPLE_RATE, nperseg=2048, noverlap=1536)
    denoised = denoised[: len(raw_samples)]
    denoised_norm = denoised * (0.89 / float(np.max(np.abs(denoised)) + 1e-9))

    for stem, arr in [
        ("YOUNG_GIRL_A_Gravado_Notebook_Bruto", raw_norm),
        ("YOUNG_GIRL_A_Gravado_Notebook_Limpo", denoised_norm),
    ]:
        wav_p = OUT_WEB_DIR / f"{stem}.wav"
        save_wav(wav_p, arr)
        mp3_p = OUT_ART_DIR / f"{stem}.mp3"
        # 48 kbps mono 32 kHz -> ~650 KB (passes Gateway!)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_p), "-ac", "1", "-ar", "32000", "-b:a", "48k", str(mp3_p)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        shutil.copy2(mp3_p, OUT_WEB_DIR / mp3_p.name)
        shutil.copy2(mp3_p, OUT_CFG_DIR / mp3_p.name)
        print(f"  -> {mp3_p.name}: {mp3_p.stat().st_size/1024:.1f} KB", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
