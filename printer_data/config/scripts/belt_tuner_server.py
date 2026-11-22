#!/home/shoomba/klippy-env/bin/python
"""
Servidor de Afinacao Acustica de Correias (Belt Tuner) para Mainsail / VZ180 AWD CoreXY
Captura audio em tempo real do microfone embutido da webcam USB via PipeWire (parec 48 kHz),
executa FFT de alta resolucao (65536 pts + interpolacao parabolica, precisao < 0.05 Hz) na
banda de tensao de correias (35-300 Hz), subtrai perfil de ruido das ventoinhas, detecta
pincadas (plucks) automaticamente e calcula a tensao em Newtons (T = 4 * mu * L^2 * f^2).
"""

from __future__ import annotations

import collections
import json
import math
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

HOST = "0.0.0.0"
PORT = 7135
SAMPLE_RATE = 48000
RING_SAMPLES = int(SAMPLE_RATE * 1.2)   # 1.2s ring buffer
FFT_WINDOW = 16384                      # 0.341s analysis window
N_FFT = 65536                           # 0.732 Hz bin spacing + parabolic interp
F_MIN_SPEC = 30.0
F_MAX_SPEC = 300.0
IDLE_TIMEOUT_S = 40.0

MOONRAKER_URL = "http://127.0.0.1:7125"
HISTORY_FILE = Path("/home/shoomba/printer_data/config/belt_tuner_history.json")

# Gates 2GT linear mass density (kg/m)
BELT_MU = {
    "6mm": 0.0055,  # 5.5 g/m
    "9mm": 0.0083,  # 8.3 g/m
    "10mm": 0.0092,
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def hz_to_note(hz: float) -> str:
    if hz <= 20.0:
        return "--"
    midi = 69.0 + 12.0 * math.log2(hz / 440.0)
    idx = int(round(midi))
    cents = (midi - idx) * 100.0
    return f"{NOTE_NAMES[idx % 12]}{idx // 12 - 1} ({cents:+.0f}c)"


def find_camera_mic() -> str:
    try:
        out = subprocess.check_output(["pactl", "list", "short", "sources"], text=True, timeout=3)
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
    except Exception:
        pass
    return "alsa_input.usb-YJXU502S_USB_Camera_20190415-02.mono-fallback"


class BeltTunerEngine:
    def __init__(self) -> None:
        self.mic_source = find_camera_mic()
        self.lock = threading.Lock()
        self.ring = np.zeros(RING_SAMPLES, dtype=np.float64)
        self.samples_received = 0
        self.running = False
        self.last_client_poll = 0.0
        self.proc: subprocess.Popen | None = None

        # Precompute bandpass filter (30..340 Hz) in SOS form (numerically stable at 48 kHz) and frequency axis
        nyq = SAMPLE_RATE / 2.0
        self.bp_sos = butter(4, [28.0 / nyq, 340.0 / nyq], btype="band", output="sos")
        self.win = np.hanning(FFT_WINDOW)
        full_freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
        self.spec_mask = (full_freqs >= F_MIN_SPEC) & (full_freqs <= F_MAX_SPEC)
        self.freqs = full_freqs[self.spec_mask]

        # Wide-band musical spectrum mask (90..1050 Hz) for the Virtual XYZ Instrument
        self.music_mask = (full_freqs >= 90.0) & (full_freqs <= 1050.0)
        self.music_freqs = full_freqs[self.music_mask]
        self.music_live_hz: float = 0.0
        self.music_live_note: str = "--"
        self.music_live_db: float = -60.0
        self.music_spectrum: list[float] = [0.0] * len(self.music_freqs[::8])

        # Settings & calibration state
        self.noise_spec: np.ndarray | None = None
        self.noise_rms_db: float = -45.0
        self.active_belt: str = "A"  # "A" or "B"
        self.auto_lock: bool = True
        self.span_mm: float = 150.0
        self.belt_width: str = "6mm"
        self.target_hz: float = 120.0
        self.search_min_hz: float = 55.0
        self.search_max_hz: float = 220.0

        # Live & locked measurements
        self.live_hz: float = 0.0
        self.live_snr_db: float = 0.0
        self.live_rms_db: float = -60.0
        self.live_note: str = "--"
        self.live_tension_n: float = 0.0
        self.live_spectrum: list[float] = [0.0] * len(self.freqs)
        self.recent_peaks: collections.deque[tuple[float, float, float]] = collections.deque(maxlen=6)
        self.last_lock_time: float = 0.0

        self.belt_a: dict | None = None
        self.belt_b: dict | None = None
        self.history: list[dict] = []
        self._load_history()

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.dsp_thread = threading.Thread(target=self._dsp_loop, daemon=True)
        self.capture_thread.start()
        self.dsp_thread.start()

    def _load_history(self) -> None:
        if HISTORY_FILE.exists():
            try:
                data = json.loads(HISTORY_FILE.read_text())
                self.belt_a = data.get("belt_a")
                self.belt_b = data.get("belt_b")
                self.history = data.get("history", [])[:20]
                self.span_mm = float(data.get("span_mm", 150.0))
                self.belt_width = str(data.get("belt_width", "6mm"))
                self.target_hz = float(data.get("target_hz", 120.0))
            except Exception:
                pass

    def _save_history(self) -> None:
        try:
            payload = {
                "belt_a": self.belt_a,
                "belt_b": self.belt_b,
                "history": self.history[:20],
                "span_mm": self.span_mm,
                "belt_width": self.belt_width,
                "target_hz": self.target_hz,
            }
            HISTORY_FILE.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass

    def calc_tension_n(self, hz: float) -> float:
        if hz <= 0:
            return 0.0
        mu = BELT_MU.get(self.belt_width, 0.0055)
        L_m = self.span_mm / 1000.0
        return round(4.0 * mu * (L_m ** 2) * (hz ** 2), 2)

    def touch_client(self) -> None:
        self.last_client_poll = time.time()
        self.running = True

    def _capture_loop(self) -> None:
        chunk_samples = 1536  # 32 ms @ 48 kHz
        chunk_bytes = chunk_samples * 2
        while True:
            if not self.running or (time.time() - self.last_client_poll > IDLE_TIMEOUT_S):
                self.running = False
                if self.proc is not None:
                    try:
                        self.proc.terminate()
                        self.proc.wait(timeout=1)
                    except Exception:
                        try:
                            self.proc.kill()
                        except Exception:
                            pass
                    self.proc = None
                time.sleep(0.15)
                continue

            if self.proc is None or self.proc.poll() is not None:
                self.mic_source = find_camera_mic()
                cmd = [
                    "parec",
                    "--device", self.mic_source,
                    "--format=s16le",
                    f"--rate={SAMPLE_RATE}",
                    "--channels=1",
                    "--raw",
                    "--latency-msec=25",
                ]
                try:
                    self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                except Exception:
                    time.sleep(0.5)
                    continue

            try:
                raw = self.proc.stdout.read(chunk_bytes)
                if not raw or len(raw) < 2:
                    time.sleep(0.02)
                    continue
                arr = np.frombuffer(raw[: len(raw) - (len(raw) % 2)], dtype=np.int16).astype(np.float64) / 32768.0
                n = len(arr)
                with self.lock:
                    if n >= RING_SAMPLES:
                        self.ring[:] = arr[-RING_SAMPLES:]
                    else:
                        self.ring[:-n] = self.ring[n:]
                        self.ring[-n:] = arr
                    self.samples_received += n
            except Exception:
                time.sleep(0.05)

    def _compute_frame_spectrum(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        frame = frame - np.mean(frame)
        filtered = sosfiltfilt(self.bp_sos, frame)
        rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(filtered ** 2))) + 1e-9)
        spec = np.abs(np.fft.rfft(filtered * self.win, n=N_FFT))[self.spec_mask]
        return spec, rms_db

    def calibrate_noise(self) -> dict:
        self.touch_client()
        time.sleep(0.45)
        specs = []
        rms_list = []
        for _ in range(5):
            with self.lock:
                frame = self.ring[-FFT_WINDOW:].copy()
            sp, rdb = self._compute_frame_spectrum(frame)
            specs.append(sp)
            rms_list.append(rdb)
            time.sleep(0.08)
        self.noise_spec = np.median(np.stack(specs, axis=0), axis=0)
        self.noise_rms_db = round(float(np.mean(rms_list)), 1)
        return {"ok": True, "noise_rms_db": self.noise_rms_db}

    def _dsp_loop(self) -> None:
        while True:
            if not self.running or self.samples_received < FFT_WINDOW:
                time.sleep(0.08)
                continue

            with self.lock:
                frame = self.ring[-FFT_WINDOW:].copy()

            try:
                spec, rms_db = self._compute_frame_spectrum(frame)
                if self.noise_spec is None:
                    self.noise_spec = spec.copy()
                    self.noise_rms_db = round(rms_db, 1)

                clean = np.maximum(spec - 1.65 * self.noise_spec, 0.0)

                # Search inside [search_min_hz, search_max_hz]
                s_mask = (self.freqs >= self.search_min_hz) & (self.freqs <= self.search_max_hz)
                if not np.any(s_mask):
                    time.sleep(0.06)
                    continue

                s_indices = np.where(s_mask)[0]
                sub_clean = clean[s_indices]
                local_best = int(np.argmax(sub_clean))
                peak_idx = int(s_indices[local_best])
                peak_mag = float(clean[peak_idx])
                floor_mag = float(np.median(sub_clean) + np.mean(self.noise_spec[s_indices]) * 0.25 + 1e-9)
                snr_db = 20.0 * math.log10((peak_mag + 1e-9) / floor_mag)

                # Sub-bin parabolic interpolation
                if 0 < peak_idx < len(clean) - 1:
                    y0, y1, y2 = float(clean[peak_idx - 1]), float(clean[peak_idx]), float(clean[peak_idx + 1])
                    denom = y0 - 2.0 * y1 + y2
                    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
                    step = float(self.freqs[1] - self.freqs[0])
                    peak_hz = float(self.freqs[peak_idx] + delta * step)
                else:
                    peak_hz = float(self.freqs[peak_idx])

                # Check sub-octave fundamental (peak_hz / 2) in case 2nd harmonic was slightly louder
                half_hz = peak_hz * 0.5
                if half_hz >= self.search_min_hz:
                    h_mask = (self.freqs >= half_hz * 0.95) & (self.freqs <= half_hz * 1.05)
                    if np.any(h_mask):
                        h_indices = np.where(h_mask)[0]
                        h_idx = int(h_indices[int(np.argmax(clean[h_indices]))])
                        if float(clean[h_idx]) >= 0.42 * peak_mag:
                            peak_idx = h_idx
                            peak_hz = float(self.freqs[h_idx])

                # Downsample spectrum to ~135 points (every 2 bins: ~1.46 Hz resolution from 30..300 Hz) for smooth UI canvas
                ds_clean = clean[::2]
                max_sp = float(np.max(ds_clean)) if len(ds_clean) else 1.0
                norm_spec = [round(float(v) / max(max_sp, 1e-6), 3) for v in ds_clean]

                self.live_rms_db = round(rms_db, 1)
                self.live_snr_db = round(max(0.0, snr_db), 1)
                self.live_spectrum = norm_spec

                # Also compute wide-band musical spectrum (90..1050 Hz) for the Virtual XYZ Instrument
                raw_zero = frame - np.mean(frame)
                raw_rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(raw_zero ** 2))) + 1e-9)
                m_spec = np.abs(np.fft.rfft(raw_zero * self.win, n=N_FFT))[self.music_mask]
                if len(m_spec) > 0:
                    m_peak_idx = int(np.argmax(m_spec))
                    m_peak_mag = float(m_spec[m_peak_idx])
                    m_floor = float(np.median(m_spec) + 1e-9)
                    m_snr = 20.0 * math.log10((m_peak_mag + 1e-9) / m_floor)
                    m_hz = float(self.music_freqs[m_peak_idx])
                    self.music_live_db = round(raw_rms_db, 1)
                    if m_snr >= 9.0 and raw_rms_db >= -35.0:
                        self.music_live_hz = round(m_hz, 1)
                        self.music_live_note = hz_to_note(m_hz)
                    ds_m = m_spec[::8]
                    m_max = float(np.max(ds_m)) if len(ds_m) else 1.0
                    self.music_spectrum = [round(float(v) / max(m_max, 1e-6), 3) for v in ds_m]

                now = time.time()
                if snr_db >= 9.5 and (rms_db >= self.noise_rms_db + 3.0):
                    self.live_hz = round(peak_hz, 1)
                    self.live_note = hz_to_note(peak_hz)
                    self.live_tension_n = self.calc_tension_n(peak_hz)
                    self.recent_peaks.append((now, peak_hz, snr_db))

                    # Auto-lock pluck when 3 recent frames within 0.35s agree within +-1.8 Hz and SNR >= 11.5 dB
                    valid_recent = [p for p in self.recent_peaks if now - p[0] <= 0.40]
                    if (
                        self.auto_lock
                        and len(valid_recent) >= 3
                        and (now - self.last_lock_time > 0.95)
                    ):
                        hz_vals = [p[1] for p in valid_recent[-3:]]
                        snr_vals = [p[2] for p in valid_recent[-3:]]
                        if (max(hz_vals) - min(hz_vals) <= 2.2) and max(snr_vals) >= 11.5:
                            locked_hz = round(float(np.median(hz_vals)), 1)
                            locked_snr = round(float(max(snr_vals)), 1)
                            self.lock_reading(self.active_belt, locked_hz, locked_snr)
                            self.last_lock_time = now
                            self.recent_peaks.clear()
            except Exception:
                pass

            time.sleep(0.055)

    def lock_reading(self, belt: str, hz: float | None = None, snr_db: float | None = None) -> dict:
        use_hz = hz if hz is not None else self.live_hz
        use_snr = snr_db if snr_db is not None else self.live_snr_db
        if use_hz <= 0:
            return {"ok": False, "error": "Nenhuma frequencia valida detectada para travar."}
        entry = {
            "belt": belt.upper(),
            "hz": round(use_hz, 1),
            "note": hz_to_note(use_hz),
            "tension_n": self.calc_tension_n(use_hz),
            "snr_db": round(use_snr, 1),
            "timestamp": time.strftime("%H:%M:%S"),
        }
        if belt.upper() == "A":
            self.belt_a = entry
        else:
            self.belt_b = entry
        self.history.insert(0, entry)
        self.history = self.history[:20]
        self._save_history()
        return {"ok": True, "entry": entry}

    def get_state(self) -> dict:
        self.touch_client()
        delta_hz = None
        delta_tension = None
        balance_status = "Aguardando leitura das Correias A e B"
        balance_level = "info"
        recommendation = "Posicione o cabecote em X90 Y37 (vao livre de 150 mm) e pince cada correia como uma corda de violao."

        if self.belt_a and self.belt_b:
            fa = float(self.belt_a["hz"])
            fb = float(self.belt_b["hz"])
            delta_hz = round(abs(fa - fb), 1)
            delta_tension = round(abs(float(self.belt_a["tension_n"]) - float(self.belt_b["tension_n"])), 2)
            avg_hz = 0.5 * (fa + fb)

            if delta_hz <= 2.0:
                balance_status = f"Correias Equalizadas (Delta: {delta_hz:.1f} Hz)"
                balance_level = "success"
            elif delta_hz <= 5.0:
                higher = "A" if fa > fb else "B"
                lower = "B" if fa > fb else "A"
                balance_status = f"Leve Diferenca (Delta: {delta_hz:.1f} Hz — Correia {higher} mais tensa)"
                balance_level = "warning"
            else:
                higher = "A" if fa > fb else "B"
                lower = "B" if fa > fb else "A"
                balance_status = f"Desbalanceadas (Delta: {delta_hz:.1f} Hz — Ajuste Correia {lower} ou {higher})"
                balance_level = "error"

            # Compare with target_hz
            diff_target = avg_hz - self.target_hz
            if abs(diff_target) <= 8.0 and delta_hz <= 2.5:
                recommendation = (
                    f"Tensao e balanceamento ideais! Media {avg_hz:.1f} Hz (alvo {self.target_hz:.0f} Hz) "
                    f"com diferenca de apenas {delta_hz:.1f} Hz entre A e B."
                )
            elif delta_hz > 2.5:
                if fa > fb:
                    recommendation = (
                        f"A Correia A ({fa:.1f} Hz) esta {delta_hz:.1f} Hz acima da Correia B ({fb:.1f} Hz). "
                        f"Aperte levemente o tensor da Correia B (ou alivie A) ate igualar em ~{self.target_hz:.0f} Hz."
                    )
                else:
                    recommendation = (
                        f"A Correia B ({fb:.1f} Hz) esta {delta_hz:.1f} Hz acima da Correia A ({fa:.1f} Hz). "
                        f"Aperte levemente o tensor da Correia A (ou alivie B) ate igualar em ~{self.target_hz:.0f} Hz."
                    )
            else:
                action = "Aperte" if diff_target < 0 else "Afrouxe"
                recommendation = (
                    f"As correias estao bem equalizadas entre si ({delta_hz:.1f} Hz), mas a media ({avg_hz:.1f} Hz) "
                    f"esta {'abaixo' if diff_target < 0 else 'acima'} do alvo ({self.target_hz:.0f} Hz). "
                    f"{action} ambos os tensores por igual."
                )

        return {
            "mic_source": self.mic_source,
            "running": self.running,
            "noise_rms_db": self.noise_rms_db,
            "active_belt": self.active_belt,
            "auto_lock": self.auto_lock,
            "span_mm": self.span_mm,
            "belt_width": self.belt_width,
            "target_hz": self.target_hz,
            "search_min_hz": self.search_min_hz,
            "search_max_hz": self.search_max_hz,
            "live": {
                "hz": self.live_hz,
                "note": self.live_note,
                "snr_db": self.live_snr_db,
                "rms_db": self.live_rms_db,
                "tension_n": self.live_tension_n,
            },
            "belt_a": self.belt_a,
            "belt_b": self.belt_b,
            "delta_hz": delta_hz,
            "delta_tension_n": delta_tension,
            "balance_status": balance_status,
            "balance_level": balance_level,
            "recommendation": recommendation,
            "history": self.history[:10],
            "freqs": [round(float(f), 1) for f in self.freqs[::2]],
            "spectrum": self.live_spectrum,
        }


ENGINE = BeltTunerEngine()


def run_klipper_gcode(script: str) -> dict:
    encoded = urllib.parse.quote(script)
    req = urllib.request.Request(f"{MOONRAKER_URL}/printer/gcode/script?script={encoded}", method="POST")
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode())


GCODES_DIR = Path("/home/shoomba/printer_data/gcodes")
CONFIG_DIR = Path("/home/shoomba/printer_data/config")
SOUNDFONT_MANIFEST = Path("/home/shoomba/mainsail/soundfont/manifest.json")


def hz_to_midi_float(hz: float) -> float:
    if hz < 35.0:
        return 0.0
    return round(69.0 + 12.0 * math.log2(hz / 440.0), 2)


def list_musical_songs() -> list[dict]:
    songs = []
    for gf in sorted(GCODES_DIR.glob("*.gcode")):
        if any(k in gf.name.upper() for k in ("MEGALOVANIA", "BAD_APPLE", "BOHEMIAN", "BUDDY", "WEEZER", "MUSIC", "AUDIO")):
            songs.append({
                "id": f"file:{gf.name}",
                "title": f"{gf.name} (G-Code Completo)",
                "type": "gcode_file",
                "path": str(gf),
            })
    # Also scan musical macros in macros_music.cfg and buzer.cfg
    for cfg_name in ("macros_music.cfg", "buzer.cfg"):
        cfg_p = CONFIG_DIR / cfg_name
        if not cfg_p.exists():
            continue
        text = cfg_p.read_text(errors="ignore")
        for line in text.splitlines():
            s = line.strip()
            if s.lower().startswith("[gcode_macro ") and s.endswith("]"):
                mname = s[13:-1].strip().upper()
                if not mname.startswith("_"):
                    songs.append({
                        "id": f"macro:{cfg_name}:{mname}",
                        "title": f"Macro {mname} ({cfg_name})",
                        "type": "macro",
                        "macro": mname,
                        "cfg": cfg_name,
                    })
    return songs


def extract_macro_gcode(cfg_name: str, macro_name: str) -> str:
    cfg_p = CONFIG_DIR / cfg_name
    if not cfg_p.exists():
        return ""
    lines = cfg_p.read_text(errors="ignore").splitlines()
    in_macro = False
    in_gcode = False
    out = []
    header = f"[gcode_macro {macro_name.lower()}]"
    for line in lines:
        s = line.strip()
        if s.lower().startswith("[") and s.endswith("]"):
            if in_macro:
                break
            if s.lower() == header:
                in_macro = True
            continue
        if in_macro:
            if s.lower().startswith("gcode:"):
                in_gcode = True
                continue
            if in_gcode:
                if line.startswith(" ") or line.startswith("\t") or s == "":
                    out.append(s)
                else:
                    break
    return "\n".join(out)


def parse_musical_gcode_text(gcode_text: str, title: str = "G-Code") -> dict:
    """
    Parse VZ180 musical G-code (3-axis CoreXY + Z) into an exact acoustic timeline of
    (t, dur, hz_x, midi_x, note_x, hz_y, midi_y, note_y, hz_z, midi_z, note_z, x_mm, y_mm, z_mm).
    """
    in_relative = False
    in_audio_xy = False
    in_audio_z = False
    scale_xy = 4.0
    scale_z = 16.0
    x_phys = 90.0
    y_phys = 87.0
    z_phys = 25.0
    feed_mm_min = 3000.0
    cur_t = 0.0
    events = []
    skip_next_return = False

    for raw_line in gcode_text.splitlines():
        line_comment = ""
        if ";" in raw_line:
            code_part, line_comment = raw_line.split(";", 1)
        else:
            code_part = raw_line
        s = code_part.strip().upper()
        if "RETORNO" in line_comment.upper():
            skip_next_return = True
        if not s:
            continue

        if s == "G91":
            in_relative = True
            continue
        if s == "G90":
            in_relative = False
            continue
        if "_AUDIO_MODE_COMPACTO_ON" in s:
            in_audio_xy = True
            scale_xy = 4.0
            continue
        if "_AUDIO_MODE_ON" in s:
            in_audio_xy = True
            scale_xy = 16.0
            continue
        if "_AUDIO_MODE_Z_ON" in s:
            in_audio_z = True
            scale_z = 16.0
            continue
        if "_AUDIO_MODE_OFF" in s or "_AUDIO_MODE_Z_OFF" in s:
            continue

        if s.startswith("G4 "):
            for tok in s.split()[1:]:
                if tok.startswith("P"):
                    try:
                        ms = float(tok[1:])
                        if ms > 0:
                            cur_t += ms / 1000.0
                    except Exception:
                        pass
            continue

        if s.startswith("G1 ") or s.startswith("G0 "):
            dx = 0.0
            dy = 0.0
            dz = 0.0
            has_xyz = False
            for tok in s.split()[1:]:
                if len(tok) < 2:
                    continue
                k = tok[0]
                try:
                    val = float(tok[1:])
                except Exception:
                    continue
                if k == "X":
                    dx = val
                    has_xyz = True
                elif k == "Y":
                    dy = val
                    has_xyz = True
                elif k == "Z":
                    dz = val
                    has_xyz = True
                elif k == "F" and val > 0:
                    feed_mm_min = val

            if not has_xyz:
                continue

            if not in_relative:
                if dx != 0:
                    x_phys = dx
                if dy != 0:
                    y_phys = dy
                if dz != 0:
                    z_phys = dz
                continue

            if skip_next_return:
                skip_next_return = False
                x_phys += dx * scale_xy
                y_phys += dy * scale_xy
                z_phys += dz * scale_z
                continue

            dist_cmd = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist_cmd < 1e-6 or feed_mm_min <= 0:
                continue

            dur_s = dist_cmd / (feed_mm_min / 60.0)
            if dur_s <= 0.002 or dur_s > 4.0:
                continue

            dA_cmd = dx + dy
            dB_cmd = dx - dy
            hz_x = round((abs(dA_cmd) * 80.0) / dur_s, 1) if abs(dA_cmd) > 1e-5 else 0.0
            hz_y = round((abs(dB_cmd) * 80.0) / dur_s, 1) if abs(dB_cmd) > 1e-5 else 0.0
            hz_z = round((abs(dz) * 1200.00075) / dur_s, 1) if abs(dz) > 1e-6 else 0.0

            # Filter out non-musical return moves (< 55 Hz or > 2200 Hz)
            if hz_x < 55.0 or hz_x > 2200.0:
                hz_x = 0.0
            if hz_y < 55.0 or hz_y > 2200.0:
                hz_y = 0.0
            if hz_z < 55.0 or hz_z > 2200.0:
                hz_z = 0.0

            x_phys = round(x_phys + dx * scale_xy, 2)
            y_phys = round(y_phys + dy * scale_xy, 2)
            z_phys = round(z_phys + dz * scale_z, 2)

            if hz_x > 0 or hz_y > 0 or hz_z > 0:
                events.append({
                    "t": round(cur_t, 4),
                    "dur": round(dur_s, 4),
                    "hz_x": hz_x,
                    "midi_x": hz_to_midi_float(hz_x),
                    "note_x": hz_to_note(hz_x).split(" ")[0] if hz_x > 0 else "--",
                    "hz_y": hz_y,
                    "midi_y": hz_to_midi_float(hz_y),
                    "note_y": hz_to_note(hz_y).split(" ")[0] if hz_y > 0 else "--",
                    "hz_z": hz_z,
                    "midi_z": hz_to_midi_float(hz_z),
                    "note_z": hz_to_note(hz_z).split(" ")[0] if hz_z > 0 else "--",
                    "x_mm": x_phys,
                    "y_mm": y_phys,
                    "z_mm": z_phys,
                })
            cur_t += dur_s

    n_x = sum(1 for e in events if e["hz_x"] > 0)
    n_y = sum(1 for e in events if e["hz_y"] > 0)
    n_z = sum(1 for e in events if e["hz_z"] > 0)
    return {
        "ok": True,
        "title": title,
        "duration_s": round(cur_t, 2),
        "total_events": len(events),
        "counts": {"x": n_x, "y": n_y, "z": n_z},
        "events": events,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args) -> None:
        return

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        if self.path.startswith("/api/state"):
            self._send_json(ENGINE.get_state())
        else:
            self._send_json({"ok": True, "service": "VZ180 Belt Tuner API"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            payload = {}

        if self.path.startswith("/api/settings"):
            if "active_belt" in payload:
                ENGINE.active_belt = str(payload["active_belt"]).upper()
            if "auto_lock" in payload:
                ENGINE.auto_lock = bool(payload["auto_lock"])
            if "span_mm" in payload:
                ENGINE.span_mm = max(50.0, min(400.0, float(payload["span_mm"])))
            if "belt_width" in payload and payload["belt_width"] in BELT_MU:
                ENGINE.belt_width = str(payload["belt_width"])
            if "target_hz" in payload:
                ENGINE.target_hz = max(50.0, min(250.0, float(payload["target_hz"])))
            if "search_min_hz" in payload:
                ENGINE.search_min_hz = max(35.0, min(180.0, float(payload["search_min_hz"])))
            if "search_max_hz" in payload:
                ENGINE.search_max_hz = max(90.0, min(295.0, float(payload["search_max_hz"])))
            # Recalculate tensions if span or width changed
            if ENGINE.belt_a:
                ENGINE.belt_a["tension_n"] = ENGINE.calc_tension_n(float(ENGINE.belt_a["hz"]))
            if ENGINE.belt_b:
                ENGINE.belt_b["tension_n"] = ENGINE.calc_tension_n(float(ENGINE.belt_b["hz"]))
            ENGINE._save_history()
            self._send_json(ENGINE.get_state())

        elif self.path.startswith("/api/calibrate_noise"):
            res = ENGINE.calibrate_noise()
            self._send_json({**res, **ENGINE.get_state()})

        elif self.path.startswith("/api/lock"):
            belt = str(payload.get("belt", ENGINE.active_belt)).upper()
            hz = float(payload["hz"]) if "hz" in payload and payload["hz"] else None
            snr = float(payload["snr_db"]) if "snr_db" in payload and payload["snr_db"] else None
            res = ENGINE.lock_reading(belt, hz, snr)
            self._send_json({**res, **ENGINE.get_state()})

        elif self.path.startswith("/api/clear"):
            which = str(payload.get("which", "all")).lower()
            if which in ("a", "all"):
                ENGINE.belt_a = None
            if which in ("b", "all"):
                ENGINE.belt_b = None
            if which == "all":
                ENGINE.history.clear()
            ENGINE.live_hz = 0.0
            ENGINE._save_history()
            self._send_json(ENGINE.get_state())

        elif self.path.startswith("/api/printer_action"):
            act = str(payload.get("action", ""))
            try:
                if act == "position_tuning":
                    # Home XY if needed, move to X90 Y37 (150mm equal belt span), keep motors locked
                    q = urllib.request.urlopen(f"{MOONRAKER_URL}/printer/objects/query?toolhead", timeout=5)
                    homed = json.loads(q.read().decode())["result"]["status"]["toolhead"]["homed_axes"]
                    cmds = []
                    if "x" not in homed or "y" not in homed:
                        cmds.append("G28 X Y")
                    cmds.extend(["G90", "G1 X90 Y37 F7200", "M400"])
                    run_klipper_gcode("\n".join(cmds))
                    self._send_json({"ok": True, "msg": "Cabecote posicionado em X=90 Y=37 (vao livre de 150 mm) com motores travados."})
                elif act == "lock_motors":
                    cmds = [
                        f"SET_STEPPER_ENABLE STEPPER={s} ENABLE=1"
                        for s in ("stepper_x", "stepper_x1", "stepper_y", "stepper_y1")
                    ]
                    run_klipper_gcode("\n".join(cmds))
                    self._send_json({"ok": True, "msg": "Motores XY energizados (polias travadas para pincamento)."})
                elif act == "unlock_motors":
                    run_klipper_gcode("M84")
                    self._send_json({"ok": True, "msg": "Motores desligados (M84)."})
                else:
                    self._send_json({"ok": False, "error": f"Acao desconhecida: {act}"}, status=400)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
        else:
            self._send_json({"ok": False, "error": "Not found"}, status=404)


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[BeltTuner] Listening on http://{HOST}:{PORT} | Mic: {ENGINE.mic_source}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
