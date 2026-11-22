#!/home/shoomba/klippy-env/bin/python
"""
Compilador Acustico 3 Eixos & Simulador Offline por SoundFont para:
  Siinamota — "Young Girl A" (Shoujo A / 少女A)
  Fonte: https://onlinesequencer.net/3710759 (3.412 notas, 9 instrumentos, 130 BPM, 103.4s)

Distribuicao de Canais Solicitada:
  - Canal A (Eixo X / Motor A +45 deg, _AUDIO_MODE_COMPACTO_ON):
      Intro Lead (I41, t=64..192) + Trilha Vocal Principal (I23, t=192..896).
      Na Ponte antes do Refrao Final (t=448..576): Vocal na Oitava MEDIA (D#4..D#5).
  - Canal B (Eixo Y / Motor B -45 deg, _AUDIO_MODE_COMPACTO_ON):
      Guitarra Eletrica (I44, t=0..448 e t=576..832) + Outro (I52, t=832..896).
      Na Ponte antes do Refrao Final (t=448..576): Vocal na Oitava GRAVE (D#3..D#4, -1 oitava).
  - Canal Z (Eixo Z / Vertical TMC2208, _AUDIO_MODE_Z_ON):
      Bateria Real (I31, t=0..448 e t=576..896: Kick, Snare, Hi-Hat, Crash, Toms).
      Na Ponte antes do Refrao Final (t=448..576): Vocal na Oitava AGUDA (D#5..D#6, +1 oitava).

Renderiza automaticamente os 3 canais separados + mix completa no Instrumento Virtual (SoundFont)
antes de executar na impressora fisica.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import struct
import subprocess
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/shoomba/printer_data/config/scripts")
import bad_apple_compiler as bac

OS_BIN_PATH = Path("/home/shoomba/printer_data/config/scripts/midis/young_girl_a_os3710759.bin")
MACROS_MUSIC_CFG = Path("/home/shoomba/printer_data/config/macros_music.cfg")
FULL_GCODE_PATH = Path("/home/shoomba/printer_data/gcodes/YOUNG_GIRL_A_FULL.gcode")
SOUNDFONT_DIR = Path("/home/shoomba/mainsail/soundfont")
MANIFEST_PATH = SOUNDFONT_DIR / "manifest.json"
ART_DIR = Path("/home/shoomba/.gemini/antigravity/brain/e94ad702-76ca-41e0-93ff-43cc29dc95cd")
CFG_DIR = Path("/home/shoomba/printer_data/config")

SAMPLE_RATE = 48000
# OnlineSequencer time unit = 1 sixteenth note (4 units = 1 quarter-note beat @ 130 BPM)
# Use DIV = 480 ticks per quarter note -> 1 OS time unit = 120 ticks!
DIV = 480
TICKS_PER_UNIT = 120
BPM = 130.0
TEMPOS = [(0, int(round(60_000_000.0 / BPM)))]  # 461538 us/beat

BRIDGE_START = 448.0  # 51.7s ("Sutetekita yume o atsumete, chotto! chotto! ...")
BRIDGE_END = 576.0    # 66.5s (Chorus drop "Aa toki ni toki ni...")


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    res, shift = 0, 0
    while True:
        b = buf[i]
        i += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            return res, i
        shift += 7


def _parse_proto_msg(buf: bytes) -> dict[int, list]:
    fields: dict[int, list] = defaultdict(list)
    i, n = 0, len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        fn, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = _read_varint(buf, i)
            fields[fn].append(v)
        elif wt == 1:
            v = struct.unpack("<d", buf[i : i + 8])[0]
            i += 8
            fields[fn].append(v)
        elif wt == 2:
            ln, i = _read_varint(buf, i)
            fields[fn].append(buf[i : i + ln])
            i += ln
        elif wt == 5:
            v = struct.unpack("<f", buf[i : i + 4])[0]
            i += 4
            fields[fn].append(v)
    return fields


def load_os_sequence(path: Path) -> dict[int, list[tuple[float, float, int, float]]]:
    top = _parse_proto_msg(path.read_bytes())
    by_inst: dict[int, list[tuple[float, float, int, float]]] = defaultdict(list)
    for nb in top.get(2, []):
        nf = _parse_proto_msg(nb)
        p = int(nf[1][0]) if 1 in nf else 0
        t = float(nf[2][0]) if 2 in nf else 0.0
        length = float(nf[3][0]) if 3 in nf else 1.0
        inst = int(nf[4][0]) if 4 in nf else 0
        vol = float(nf[5][0]) if 5 in nf else 1.0
        by_inst[inst].append((t, length, p, vol))
    return by_inst


def clean_monophonic(notes: list[tuple[int, int, int]], artic_gap: int = 18) -> list[tuple[int, int, int]]:
    """Resolve overlaps and insert a crisp articulation gap when consecutive notes share the same pitch."""
    notes = sorted(notes, key=lambda x: (x[0], -x[2]))
    # Deduplicate same start tick (keep first = highest priority/pitch)
    dedup: list[tuple[int, int, int]] = []
    for s, e, p in notes:
        if dedup and dedup[-1][0] == s:
            continue
        dedup.append((s, e, p))
    out: list[tuple[int, int, int]] = []
    for i, (s, e, p) in enumerate(dedup):
        if i + 1 < len(dedup):
            ns, _, np_next = dedup[i + 1]
            if e > ns:
                e = ns
            if np_next == p and (ns - e) < artic_gap and (e - s) > artic_gap * 2:
                e = max(s + 20, ns - artic_gap)
        if e > s:
            out.append((s, e, p))
    return out


def build_tracks_from_os3710759() -> tuple[
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
]:
    by_inst = load_os_sequence(OS_BIN_PATH)

    # Group chords by start time for each instrument
    def group_by_time(inst_id: int) -> list[tuple[float, list[tuple[float, int]]]]:
        grp: dict[float, list[tuple[float, int]]] = defaultdict(list)
        for t, length, p, _v in by_inst.get(inst_id, []):
            t_q = round(t * 4.0) / 4.0
            grp[t_q].append((length, p))
        return sorted(grp.items())

    vA_raw: list[tuple[int, int, int]] = []
    vB_raw: list[tuple[int, int, int]] = []
    vZ_raw: list[tuple[int, int, int]] = []

    # ---------------- 1. CANAL A: INTRO RIFF (I41) + VOCAL PRINCIPAL (I23) ----------------
    # Intro Lead Riff (I41, t=64..192): take top voice transposed -12 into golden zone (58..70)
    for t, items in group_by_time(41):
        if 64.0 <= t < 192.0:
            items.sort(key=lambda x: x[1])
            length, p = items[-1]
            p_adj = p - 12  # 70..82 -> 58..70 (A#3..A#4)
            s_tk = int(round(t * TICKS_PER_UNIT))
            e_tk = int(round((t + max(0.5, length)) * TICKS_PER_UNIT))
            vA_raw.append((s_tk, e_tk, p_adj))

    # Vocal Principal (I23, t=192..896):
    # Note: in verse/bridge (t<576), I23 is in 75..87 -> shift -12 to 63..75 (D#4..D#5).
    # In chorus (t>=576), I23 has octave doubles like (72, 84) -> taking min pitch gives 65..77 (F4..F5)!
    for t, items in group_by_time(23):
        items.sort(key=lambda x: x[1])
        length_lo, p_lo = items[0]
        length_hi, p_hi = items[-1]

        if BRIDGE_START <= t < BRIDGE_END:
            # PONTE ANTES DO REFRAO FINAL (t=448..576):
            # Todos os 3 eixos tocam a trilha vocal em OITAVAS DIFERENTES!
            # Raw vocal in bridge is 75..87 (D#5..D#6):
            #   - Canal B (Eixo Y, Grave): p - 24 -> 51..63 (D#3..D#4)
            #   - Canal A (Eixo X, Media): p - 12 -> 63..75 (D#4..D#5)
            #   - Canal Z (Eixo Z, Aguda): p      -> 75..87 (D#5..D#6)
            p_raw = p_hi
            s_tk = int(round(t * TICKS_PER_UNIT))
            dur_tk = max(120, int(round(max(length_lo, length_hi) * TICKS_PER_UNIT)))
            e_tk = s_tk + dur_tk
            vA_raw.append((s_tk, e_tk, p_raw - 12))
            vB_raw.append((s_tk, e_tk, p_raw - 24))
            vZ_raw.append((s_tk, e_tk, p_raw))
        else:
            # Outside bridge: keep Canal A cleanly in F4..F5 (63..77)
            p_use = p_lo if p_lo <= 77 else (p_lo - 12)
            s_tk = int(round(t * TICKS_PER_UNIT))
            e_tk = int(round((t + max(0.5, length_lo)) * TICKS_PER_UNIT))
            vA_raw.append((s_tk, e_tk, p_use))

    # ---------------- 2. CANAL B: GUITARRA (I44) + OUTRO (I52) ----------------
    # Electric Guitar (I44, t=0..448 and t=576..832): take highest voice of each strummed chord (51..68)
    for t, items in group_by_time(44):
        if BRIDGE_START <= t < BRIDGE_END:
            continue
        items.sort(key=lambda x: x[1])
        length, p = items[-1]
        s_tk = int(round(t * TICKS_PER_UNIT))
        e_tk = int(round((t + max(0.5, min(length, 2.0))) * TICKS_PER_UNIT))
        vB_raw.append((s_tk, e_tk, p))

    # Outro Synth/Guitar (I52, t=832..896) after I44 finishes at 830
    for t, items in group_by_time(52):
        if t >= 832.0:
            items.sort(key=lambda x: x[1])
            length, p = items[-1]
            s_tk = int(round(t * TICKS_PER_UNIT))
            e_tk = int(round((t + max(1.0, length)) * TICKS_PER_UNIT))
            vB_raw.append((s_tk, e_tk, p))

    # ---------------- 3. CANAL Z: BATERIA REAL (I31) ----------------
    # Priority: Snare (38,40) > Kick (36) > Crash/China (49,52) > Toms (45,48) > Open Hat (46) > Closed Hat/Tamb (42,44,54)
    def drum_priority(p: int) -> int:
        if p in (38, 40):
            return 6
        if p == 36:
            return 5
        if p in (49, 52):
            return 4
        if p in (45, 48):
            return 3
        if p == 46:
            return 2
        return 1

    for t, items in group_by_time(31):
        if BRIDGE_START <= t < BRIDGE_END:
            # Bridge uses all 3 axes on the vocal track in different octaves!
            continue
        items.sort(key=lambda x: drum_priority(x[1]), reverse=True)
        _, p_drum = items[0]
        s_tk = int(round(t * TICKS_PER_UNIT))

        if p_drum in (38, 40):
            # Snare: sharp 2-step crack (F5 -> C#5)
            vZ_raw.append((s_tk, s_tk + 36, 77))
            vZ_raw.append((s_tk + 36, s_tk + 96, 73))
        elif p_drum == 36:
            # Kick: punchy pitch-drop thump (G4 -> C4)
            vZ_raw.append((s_tk, s_tk + 36, 67))
            vZ_raw.append((s_tk + 36, s_tk + 102, 60))
        elif p_drum in (49, 52):
            # Crash / China Cymbal: bright metallic accent (G#5 -> G5)
            vZ_raw.append((s_tk, s_tk + 48, 80))
            vZ_raw.append((s_tk + 48, s_tk + 108, 79))
        elif p_drum in (45, 48):
            # Tom
            vZ_raw.append((s_tk, s_tk + 90, 67 if p_drum == 48 else 64))
        elif p_drum == 46:
            # Open Hi-Hat
            vZ_raw.append((s_tk, s_tk + 72, 78))
        else:
            # Closed Hi-Hat / Tambourine / Rim
            vZ_raw.append((s_tk, s_tk + 48, 76))

    return (
        clean_monophonic(vA_raw, artic_gap=18),
        clean_monophonic(vB_raw, artic_gap=16),
        clean_monophonic(vZ_raw, artic_gap=8),
    )


def simulate_channels_with_soundfont(
    gcode_lines: list[str],
    out_prefix: str = "YOUNG_GIRL_A_OS_Sim",
) -> dict[str, Path]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    midi_min = int(manifest.get("midi_min", 48))
    midi_max = int(manifest.get("midi_max", 79))

    samples_bank: dict[str, dict[int, tuple[np.ndarray, dict]]] = {"x": {}, "y": {}, "z": {}}
    for ax in ("x", "y", "z"):
        for m_str, meta in manifest["axes"][ax].items():
            m_int = int(m_str)
            wav_p = SOUNDFONT_DIR / meta["wav_url"].replace("/soundfont/", "")
            with wave.open(str(wav_p), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
            samples_bank[ax][m_int] = (arr, meta)

    events = []
    cur_t = 0.0
    in_rel = True
    feed = 3000.0
    skip_return = False

    for line in gcode_lines:
        s = line.strip()
        if "Retorno" in s or "RETORNO" in s:
            skip_return = True
        if ";" in s:
            s = s.split(";", 1)[0].strip()
        if not s:
            continue
        if s == "G91":
            in_rel = True
            continue
        if s == "G90":
            in_rel = False
            continue
        if s.startswith("G4 "):
            for tok in s.split()[1:]:
                if tok.startswith("P"):
                    cur_t += float(tok[1:]) / 1000.0
            continue
        if in_rel and s.startswith("G1 "):
            if skip_return:
                skip_return = False
                continue
            dx = dy = dz = 0.0
            for tok in s.split()[1:]:
                if tok.startswith("X"):
                    dx = float(tok[1:])
                elif tok.startswith("Y"):
                    dy = float(tok[1:])
                elif tok.startswith("Z"):
                    dz = float(tok[1:])
                elif tok.startswith("F"):
                    feed = float(tok[1:])
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < 1e-6 or feed <= 0:
                continue
            dur = dist / (feed / 60.0)
            if dur <= 0.002 or dur > 4.0:
                continue
            hz_a = (abs(dx + dy) * 80.0) / dur if abs(dx + dy) > 1e-5 else 0.0
            hz_b = (abs(dx - dy) * 80.0) / dur if abs(dx - dy) > 1e-5 else 0.0
            hz_z = (abs(dz) * 1200.00075) / dur if abs(dz) > 1e-6 else 0.0
            if hz_a < 55 or hz_a > 2400:
                hz_a = 0.0
            if hz_b < 55 or hz_b > 2400:
                hz_b = 0.0
            if hz_z < 55 or hz_z > 2400:
                hz_z = 0.0
            events.append((cur_t, dur, hz_a, hz_b, hz_z))
            cur_t += dur

    total_samples = int((cur_t + 0.5) * SAMPLE_RATE)
    buf_a = np.zeros(total_samples, dtype=np.float64)
    buf_b = np.zeros(total_samples, dtype=np.float64)
    buf_z = np.zeros(total_samples, dtype=np.float64)

    def render_note(axis: str, hz: float, dur_s: float) -> np.ndarray:
        midi_fl = 69.0 + 12.0 * math.log2(hz / 440.0)
        m_key = max(midi_min, min(midi_max, int(round(midi_fl))))
        src_arr, meta = samples_bank[axis][m_key]
        rate = hz / float(meta["target_hz"])
        loop_s = int(meta["loop_start"])
        loop_e = max(loop_s + 64, int(meta["loop_end"]))
        loop_len = loop_e - loop_s

        n_out = max(16, int(round(dur_s * SAMPLE_RATE)))
        src_positions = np.arange(n_out, dtype=np.float64) * rate
        over = src_positions >= loop_s
        src_positions[over] = loop_s + np.mod(src_positions[over] - loop_s, loop_len)
        src_positions = np.clip(src_positions, 0, len(src_arr) - 2)

        idx0 = src_positions.astype(np.int64)
        frac = src_positions - idx0
        out = (1.0 - frac) * src_arr[idx0] + frac * src_arr[idx0 + 1]

        att = min(n_out // 3, int(0.004 * SAMPLE_RATE))
        rel = min(n_out // 3, int(0.010 * SAMPLE_RATE))
        if att > 1:
            out[:att] *= np.linspace(0.0, 1.0, att)
        if rel > 1:
            out[-rel:] *= np.linspace(1.0, 0.0, rel)
        return out

    for t0, dur, hz_a, hz_b, hz_z in events:
        i0 = int(round(t0 * SAMPLE_RATE))
        if hz_a > 0:
            w = render_note("x", hz_a, dur)
            i1 = min(total_samples, i0 + len(w))
            buf_a[i0:i1] += w[: i1 - i0]
        if hz_b > 0:
            w = render_note("y", hz_b, dur)
            i1 = min(total_samples, i0 + len(w))
            buf_b[i0:i1] += w[: i1 - i0]
        if hz_z > 0:
            w = render_note("z", hz_z, dur)
            i1 = min(total_samples, i0 + len(w))
            buf_z[i0:i1] += w[: i1 - i0]

    buf_mix = 1.05 * buf_a + 0.85 * buf_b + 0.90 * buf_z

    outputs = {
        "Canal_A_Vocal": buf_a,
        "Canal_B_Guitarra": buf_b,
        "Canal_Z_Bateria": buf_z,
        "Mix_Completa_ABZ": buf_mix,
    }
    saved_paths: dict[str, Path] = {}

    for tag, arr in outputs.items():
        peak = float(np.max(np.abs(arr)) + 1e-9)
        norm_arr = arr * (0.88 / peak)
        pcm16 = np.clip(np.round(norm_arr * 32767.0), -32767, 32767).astype(np.int16)

        wav_name = f"{out_prefix}_{tag}.wav"
        mp3_name = f"{out_prefix}_{tag}.mp3"
        wav_p = SOUNDFONT_DIR / wav_name
        mp3_p = SOUNDFONT_DIR / mp3_name

        with wave.open(str(wav_p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())

        # 48 kbps mono 32 kHz -> ~615 KB for 103s (passes Gateway cleanly!)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_p), "-ac", "1", "-ar", "32000", "-b:a", "48k", str(mp3_p)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        shutil.copy2(mp3_p, ART_DIR / mp3_name)
        shutil.copy2(mp3_p, CFG_DIR / mp3_name)
        saved_paths[tag] = ART_DIR / mp3_name

    return saved_paths


def main() -> int:
    vA, vB, vZ = build_tracks_from_os3710759()
    print(f"[OS3710759] Notas extraidas: A (Vocal+Intro)={len(vA)} | B (Guitarra+Ponte)={len(vB)} | Z (Bateria+Ponte)={len(vZ)}", flush=True)

    cmds, stats = bac.compile_3axis_gcode(vA, vB, vZ, DIV, TEMPOS)
    print(f"[OS3710759] Compilado 3 Eixos: {stats}", flush=True)

    # Save YOUNG_GIRL_A_FULL.gcode (full 103.4s from OnlineSequencer 3710759)
    full_lines = [
        "; Siinamota - Young Girl A (OnlineSequencer #3710759) — 3 Eixos (A=Vocal, B=Guitarra, Z=Bateria + Ponte 3 Oitavas)",
        f"; Duracao: {stats['duration_s']}s | Movimentos: {stats['moves']} | Bounding Box: X{stats['x_phys_range']} Y{stats['y_phys_range']} Z{stats['z_phys_range']}",
        "G28",
        "G90",
        "G1 X90.000 Y87.000 Z25.000 F6000",
        "_AUDIO_MODE_COMPACTO_ON",
        "_AUDIO_MODE_Z_ON",
        "G91",
        *cmds,
        "_AUDIO_MODE_Z_OFF",
        "_AUDIO_MODE_OFF",
        "G90",
        "G1 X90.000 Y87.000 Z25.000 F6000",
        "M84",
    ]
    FULL_GCODE_PATH.write_text("\n".join(full_lines) + "\n")

    # Update [gcode_macro YOUNG_GIRL_A] in macros_music.cfg
    macro_block = "\n".join([
        "[gcode_macro YOUNG_GIRL_A]",
        "description: Siinamota - Young Girl A (OS#3710759: A=Vocal, B=Guitarra, Z=Bateria, Ponte 3 Oitavas, 103s)",
        "gcode:",
        '  {% if "xyz" not in printer.toolhead.homed_axes %}',
        "    G28",
        "  {% endif %}",
        "  SAVE_GCODE_STATE NAME=young_girl_a_state",
        "  G90",
        "  G1 X90.000 Y87.000 Z25.000 F6000",
        "  _AUDIO_MODE_COMPACTO_ON",
        "  _AUDIO_MODE_Z_ON",
        "  G91",
        *[f"  {c}" for c in cmds],
        "  _AUDIO_MODE_Z_OFF",
        "  _AUDIO_MODE_OFF",
        "  G90",
        "  G1 X90.000 Y87.000 Z25.000 F6000",
        "  RESTORE_GCODE_STATE NAME=young_girl_a_state",
    ])

    cfg_text = MACROS_MUSIC_CFG.read_text()
    if "[gcode_macro YOUNG_GIRL_A]" in cfg_text:
        cfg_text = re.sub(
            r"\[gcode_macro YOUNG_GIRL_A\].*?(?=\n\[gcode_macro |\Z)",
            macro_block,
            cfg_text,
            flags=re.S,
        )
    else:
        cfg_text = cfg_text.rstrip() + "\n\n" + macro_block + "\n"
    MACROS_MUSIC_CFG.write_text(cfg_text)

    # Render the 3 separate channels + full mix with the Virtual Instrument (SoundFont)
    print("[OS3710759] Renderizando simulacao no Instrumento Virtual (Canais A, B, Z separados e Mix)...", flush=True)
    sim_paths = simulate_channels_with_soundfont(cmds, out_prefix="YOUNG_GIRL_A_OS_Sim")
    for k, p in sim_paths.items():
        print(f"  -> {k}: {p} ({p.stat().st_size / 1024.0:.1f} KB)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
