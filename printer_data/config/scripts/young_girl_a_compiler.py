#!/home/shoomba/klippy-env/bin/python
"""
Compilador Acustico 3 Eixos & Simulador Offline por SoundFont para:
  Siinamota (powapowa-P) — "Young Girl A" (Shoujo A / 少女A) (260 BPM, F minor / Ab major)

Distribuicao de Canais Solicitada:
  - Canal A (Eixo X / Motor A +45 deg, _AUDIO_MODE_COMPACTO_ON):
      Vocal Principal de Kagamine Rin extraido de shoujo_a_3924.ustx (719 notas, D#4..F5)
      com articulacao silabica preservada ("Boku no inochi...", "Aa toki ni...", "samui samui...").
  - Canal B (Eixo Y / Motor B -45 deg, _AUDIO_MODE_COMPACTO_ON):
      Instrumentos Principais: Riff iconico de Piano/Synth na Intro, Acordes & Arpejos sincopados
      (Dbmaj7 -> C7 -> Fm7 -> Eb/Ab7) no Verso, Stabs com pausas dramaticas no Pre-Refrao
      ("chotto! chotto!") e Harmonia + Synth Lead (VoicePart 1) no Refrao.
  - Canal Z (Eixo Z / Vertical TMC2208, _AUDIO_MODE_Z_ON):
      Percussao / Batida de Fundo Breakbeat (Bumbo com pitch-drop G4->Root, Caixa com transient
      F5->C#5, Chimbal staccato F#5 e viradas de caixa antes do drop do refrao).

Antes de rodar na impressora fisica, renderiza a simulacao acustica offline usando as 96 amostras
reais do SoundFont (/home/shoomba/mainsail/soundfont/samples/{x,y,z}/) em 4 arquivos:
  1. Canal A Isolado (Vocal)
  2. Canal B Isolado (Instrumentos Principais)
  3. Canal Z Isolado (Percussao de Fundo)
  4. Mix Completa (Canais A + B + Z)
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, "/home/shoomba/printer_data/config/scripts")
import bad_apple_compiler as bac
import midi_to_corexy as mc

USTX_PATH = Path("/home/shoomba/printer_data/config/scripts/midis/shoujo_a_ust/shoujo_a_3924.ustx")
MACROS_MUSIC_CFG = Path("/home/shoomba/printer_data/config/macros_music.cfg")
FULL_GCODE_PATH = Path("/home/shoomba/printer_data/gcodes/YOUNG_GIRL_A_FULL.gcode")
SOUNDFONT_DIR = Path("/home/shoomba/mainsail/soundfont")
MANIFEST_PATH = SOUNDFONT_DIR / "manifest.json"
ART_DIR = Path("/home/shoomba/.gemini/antigravity/brain/e94ad702-76ca-41e0-93ff-43cc29dc95cd")
CFG_DIR = Path("/home/shoomba/printer_data/config")

SAMPLE_RATE = 48000
DIV = 480
BAR_TICKS = 1920  # 4/4 @ 480 ticks/beat
BPM = 260.0
TEMPOS = [(0, int(round(60_000_000.0 / BPM)))]  # 230769 us/beat


def chord_for_bar(bar_idx: int) -> dict:
    """
    8-bar harmonic loop aligned to (bar_idx - 1) % 8:
      Bars 0..1: Dbmaj7 (C#4=61, F4=65, G#4=68, C5=72, D#5=75)
      Bars 2..3: C7     (C4=60,  E4=64, G4=67,  A#4=70, C5=72)
      Bars 4..5: Fm7    (F4=65,  G#4=68, C5=72, D#5=75, F5=77)
      Bars 6..7: Eb/Ab7 (D#4=63, G4=67, G#4=68, A#4=70, D#5=75)
    """
    phase = ((bar_idx - 1) % 8) // 2
    if phase == 0:
        return {
            "name": "Dbmaj7",
            "root_z": 61,  # C#4
            "arp": [61, 65, 68, 72, 68, 65, 68, 72],
            "riff": [73, 72, 68, 65, 68, 72, 73, 72],
            "stab": 65,
        }
    if phase == 1:
        return {
            "name": "C7",
            "root_z": 60,  # C4
            "arp": [60, 64, 67, 70, 67, 64, 67, 70],
            "riff": [72, 70, 67, 64, 67, 70, 72, 70],
            "stab": 64,
        }
    if phase == 2:
        return {
            "name": "Fm7",
            "root_z": 65,  # F4
            "arp": [65, 68, 72, 75, 72, 68, 72, 75],
            "riff": [77, 75, 72, 68, 72, 75, 77, 75],
            "stab": 68,
        }
    return {
        "name": "Eb7",
        "root_z": 63,  # D#4
        "arp": [63, 67, 70, 75, 70, 67, 68, 70],
        "riff": [75, 73, 72, 70, 68, 70, 72, 75],
        "stab": 67,
    }


def build_young_girl_a_tracks(bar_start: int = 17, bar_end: int = 113) -> tuple[
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
]:
    """
    Build (vA_vocal, vB_instruments, vZ_percussion) shifted so bar_start becomes tick 0.
    Default range [Bar 17 .. Bar 113] (96 bars @ 260 BPM = 88.6s):
      - Bar 17..25: Iconic 8-bar Instrumental Intro + Breakbeat Drums
      - Bar 25..57: Verse 1 & 2 ("Boku no inochi tsuttatte...", "Kuchiru made no aizou o...")
      - Bar 57..73: Pre-Chorus ("Sutetekita yume o atsumete, chotto! chotto! machigaeta kara, pa! pa!")
      - Bar 73..113: Full Chorus ("Aa toki ni toki ni... samui samui...", "Aa kimi no... tooi tooi...", "Nanbanme demo...")
    """
    data = yaml.safe_load(USTX_PATH.read_text(encoding="utf-8-sig"))
    vp_vocal = data["voice_parts"][0]
    pos_vocal = int(vp_vocal["position"])

    # Also read Chorus harmony (VoicePart 1) to enrich Canal B during the Chorus
    vp_harm = data["voice_parts"][1]
    pos_harm = int(vp_harm["position"])

    tk_offset = bar_start * BAR_TICKS
    tk_max = bar_end * BAR_TICKS

    # ---------------- 1. CANAL A: VOCAL PRINCIPAL ----------------
    vA: list[tuple[int, int, int]] = []
    raw_vocal = []
    for n in vp_vocal["notes"]:
        s = pos_vocal + int(n["position"])
        e = s + int(n["duration"])
        p = int(n["tone"])
        if s >= tk_offset and s < tk_max:
            raw_vocal.append((s - tk_offset, min(e, tk_max) - tk_offset, p))

    # Add a 30-tick (28ms) syllable articulation breath when consecutive notes share the exact same pitch
    for i, (s, e, p) in enumerate(raw_vocal):
        if i + 1 < len(raw_vocal):
            ns, _, np_next = raw_vocal[i + 1]
            if ns <= e and np_next == p and (e - s) > 90:
                e = e - 30
        if e > s:
            vA.append((s, e, p))

    # Build harmony lookup by bar/tick for Chorus Canal B enrichment
    harm_by_tick: list[tuple[int, int, int]] = []
    for n in vp_harm["notes"]:
        s = pos_harm + int(n["position"])
        e = s + int(n["duration"])
        p = int(n["tone"]) - 12  # Bring harmony into C4..F4 mid register to complement lead vocal
        if s >= tk_offset and s < tk_max:
            harm_by_tick.append((s - tk_offset, min(e, tk_max) - tk_offset, p))

    # ---------------- 2. CANAL B: INSTRUMENTOS PRINCIPAIS ----------------
    vB: list[tuple[int, int, int]] = []
    # ---------------- 3. CANAL Z: PERCUSSAO / BATIDA DE FUNDO ----------------
    vZ: list[tuple[int, int, int]] = []

    for bar in range(bar_start, bar_end):
        b0 = (bar - bar_start) * BAR_TICKS
        ch = chord_for_bar(bar)
        is_intro = (bar < 25) or (109 <= bar < 113)
        is_stop_break = bar in (64, 65, 66, 67, 136, 137, 138, 139)
        is_papa_break = bar in (71, 72, 140)
        is_prechorus = (57 <= bar < 64) or (68 <= bar < 71) or (129 <= bar < 136)
        is_chorus = (73 <= bar < 109) or (141 <= bar < 219)

        # === CANAL B (INSTRUMENTOS PRINCIPAIS) ===
        if is_intro:
            # Signature fast 8th-note piano/synth riff (8 notes per bar, 240 ticks each)
            for step_i, pitch in enumerate(ch["riff"]):
                s = b0 + step_i * 240
                e = s + 215
                vB.append((s, e, pitch))
        elif is_stop_break:
            # Dramatic staccato chord hit on Beat 0 ("chotto! ... chotto!") + silence break
            vB.append((b0, b0 + 720, ch["stab"]))
        elif is_papa_break:
            # "ra... pa! pa!" dramatic hits before Chorus drop
            if bar in (71, 140):
                vB.append((b0, b0 + 440, ch["stab"]))
                vB.append((b0 + 960, b0 + 1400, ch["stab"] + 4))
            else:
                # Bar 72: "pa!" hit on beat 0 + rising synth glissando into Chorus
                vB.append((b0, b0 + 440, ch["stab"] + 7))
                for g_i, gp in enumerate([63, 65, 68, 70]):
                    vB.append((b0 + 960 + g_i * 240, b0 + 960 + (g_i + 1) * 240 - 20, gp))
        elif is_prechorus:
            # Driving 4-beat rising piano arpeggio
            for beat_i in range(4):
                p = ch["arp"][beat_i * 2]
                s = b0 + beat_i * 480
                vB.append((s, s + 430, p))
        elif is_chorus:
            # High-energy driving synth/guitar accompaniment (8th-note pulse alternating root & chord tone)
            for step_i in range(8):
                s = b0 + step_i * 240
                e = s + 215
                p = ch["arp"][step_i]
                vB.append((s, e, p))
        else:
            # Verse: syncopated Rhodes/piano chord comping (beats 0, 1.5, 2.5, 3.0) + fill on odd bars
            pattern = [
                (0, 380, ch["arp"][0]),
                (720, 1100, ch["arp"][1]),
                (1200, 1580, ch["arp"][2]),
                (1680, 1890, ch["arp"][3]),
            ]
            for ds, de, p in pattern:
                vB.append((b0 + ds, b0 + de, p))

        # === CANAL Z (PERCUSSAO DE FUNDO / BREAKBEAT) ===
        if is_stop_break:
            # Heavy Kick + Crash accent on beat 0 ("chotto!") then total drum silence
            vZ.append((b0, b0 + 140, 72))             # Crash/Attack C5
            vZ.append((b0 + 140, b0 + 480, ch["root_z"]))  # Deep Kick sustain
        elif is_papa_break:
            if bar == 71:
                # Kick on 0 ("ra") and Snare hit on beat 2 ("pa!")
                vZ.append((b0, b0 + 160, 67))
                vZ.append((b0 + 160, b0 + 420, ch["root_z"]))
                vZ.append((b0 + 960, b0 + 1140, 77))
                vZ.append((b0 + 1140, b0 + 1400, 73))
            else:
                # Bar 72: Hit on beat 0 ("pa!") + 4-stroke Snare Roll fill (beats 2..4) into Chorus
                vZ.append((b0, b0 + 160, 77))
                vZ.append((b0 + 160, b0 + 440, ch["root_z"]))
                for r_i, rp in enumerate([74, 75, 77, 79]):
                    rs = b0 + 960 + r_i * 240
                    vZ.append((rs, rs + 200, rp))
        else:
            # Signature fast 260 BPM Vocaloid Breakbeat:
            # Beat 0 (0..420): Kick Drum (transient G4 -> bass root)
            vZ.append((b0, b0 + 130, 67))
            vZ.append((b0 + 130, b0 + 420, ch["root_z"]))
            # Beat 1 (480..680): Closed Hi-Hat tick (F#5=78)
            vZ.append((b0 + 480, b0 + 660, 78))
            # Beat 1.5 (720..900): Syncopated Ghost Kick (in Chorus & Intro)
            if is_chorus or is_intro:
                vZ.append((b0 + 720, b0 + 900, ch["root_z"]))
            # Beat 2 (960..1360): Snare Drum Crack (F5=77 -> C#5=73)
            vZ.append((b0 + 960, b0 + 1120, 77))
            vZ.append((b0 + 1120, b0 + 1360, 73))
            # Beat 3 (1440..1620): Open Hi-Hat tick (G5=79)
            vZ.append((b0 + 1440, b0 + 1620, 79))
            # Beat 3.5 (1680..1860): Pickup Kick into next bar
            vZ.append((b0 + 1680, b0 + 1860, 65))

    return vA, vB, vZ


def simulate_channels_with_soundfont(
    gcode_lines: list[str],
    out_prefix: str = "YOUNG_GIRL_A_Sim",
) -> dict[str, Path]:
    """
    Parse the compiled G-code and render 4 offline audio simulations using the 96 real
    recorded 48 kHz stepper motor samples from /home/shoomba/mainsail/soundfont/samples/{x,y,z}/:
      1. Canal A (Vocal)
      2. Canal B (Instrumentos Principais)
      3. Canal Z (Percussao de Fundo)
      4. Mix Completa (Canais A + B + Z)
    """
    manifest = json.loads(MANIFEST_PATH.read_text())
    midi_min = int(manifest.get("midi_min", 48))
    midi_max = int(manifest.get("midi_max", 79))

    # Load all 96 WAV samples into memory
    samples_bank: dict[str, dict[int, tuple[np.ndarray, dict]]] = {"x": {}, "y": {}, "z": {}}
    for ax in ("x", "y", "z"):
        for m_str, meta in manifest["axes"][ax].items():
            m_int = int(m_str)
            wav_p = SOUNDFONT_DIR / meta["wav_url"].replace("/soundfont/", "")
            with wave.open(str(wav_p), "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
            samples_bank[ax][m_int] = (arr, meta)

    # Parse G-code into (t_start, dur_s, hz_A, hz_B, hz_Z)
    events = []
    cur_t = 0.0
    in_rel = False
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
            if hz_a < 55 or hz_a > 2200:
                hz_a = 0.0
            if hz_b < 55 or hz_b > 2200:
                hz_b = 0.0
            if hz_z < 55 or hz_z > 2200:
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
        # Map positions beyond loop_s into [loop_s, loop_e)
        over = src_positions >= loop_s
        src_positions[over] = loop_s + np.mod(src_positions[over] - loop_s, loop_len)
        src_positions = np.clip(src_positions, 0, len(src_arr) - 2)

        idx0 = src_positions.astype(np.int64)
        frac = src_positions - idx0
        out = (1.0 - frac) * src_arr[idx0] + frac * src_arr[idx0 + 1]

        # Apply smooth 4 ms attack and 10 ms release envelope
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

    # Combined 3-channel mix (balanced so Vocal A stands out clearly over Instruments B and Percussion Z)
    buf_mix = 1.05 * buf_a + 0.80 * buf_b + 0.90 * buf_z

    outputs = {
        "Canal_A_Vocal": buf_a,
        "Canal_B_Instrumentos": buf_b,
        "Canal_Z_Percussao": buf_z,
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

        # Encode Gateway-friendly MP3 (< 700 KB, 64 kbps mono 32 kHz)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_p), "-ac", "1", "-ar", "32000", "-b:a", "64k", str(mp3_p)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        shutil.copy2(mp3_p, ART_DIR / mp3_name)
        shutil.copy2(mp3_p, CFG_DIR / mp3_name)
        saved_paths[tag] = ART_DIR / mp3_name

    return saved_paths


def main() -> int:
    # 1. Build 88.6s macro version (Intro + Verse 1/2 + Pre-Chorus + Full Chorus)
    vA_short, vB_short, vZ_short = build_young_girl_a_tracks(bar_start=17, bar_end=113)
    cmds_short, stats_short = bac.compile_3axis_gcode(
        vA_short,
        vB_short,
        vZ_short,
        DIV,
        TEMPOS,
    )
    print(f"[YoungGirlA] Macro (Intro..Refrao): {stats_short}", flush=True)

    # 2. Build full 186s G-code file version (Bar 17..219)
    vA_full, vB_full, vZ_full = build_young_girl_a_tracks(bar_start=17, bar_end=219)
    cmds_full, stats_full = bac.compile_3axis_gcode(
        vA_full,
        vB_full,
        vZ_full,
        DIV,
        TEMPOS,
    )
    print(f"[YoungGirlA] Full G-code: {stats_full}", flush=True)

    # Save YOUNG_GIRL_A_FULL.gcode
    full_lines = [
        "; Siinamota - Young Girl A (Shoujo A / 少女A) — 3 Eixos (A=Vocal, B=Instrumentos, Z=Percussao)",
        f"; Duracao: {stats_full['duration_s']}s | Movimentos: {stats_full['moves']} | Bounding Box: X[{stats_full['x_min']}..{stats_full['x_max']}] Y[{stats_full['y_min']}..{stats_full['y_max']}] Z[{stats_full['z_min']}..{stats_full['z_max']}]",
        "G28",
        "G90",
        "G1 X90.000 Y87.000 Z25.000 F6000",
        "_AUDIO_MODE_COMPACTO_ON",
        "_AUDIO_MODE_Z_ON",
        *cmds_full,
        "_AUDIO_MODE_Z_OFF",
        "_AUDIO_MODE_OFF",
        "G90",
        "G1 X90.000 Y87.000 Z25.000 F6000",
        "M84",
    ]
    FULL_GCODE_PATH.write_text("\n".join(full_lines) + "\n")

    # Add/Update [gcode_macro YOUNG_GIRL_A] in macros_music.cfg
    macro_block = "\n".join([
        "[gcode_macro YOUNG_GIRL_A]",
        "description: Siinamota - Young Girl A / Shoujo A (3 Eixos: A=Vocal, B=Instrumentos, Z=Percussao, 88s)",
        "gcode:",
        '  {% if "xyz" not in printer.toolhead.homed_axes %}',
        "    G28",
        "  {% endif %}",
        "  SAVE_GCODE_STATE NAME=young_girl_a_state",
        "  G90",
        "  G1 X90.000 Y87.000 Z25.000 F6000",
        "  _AUDIO_MODE_COMPACTO_ON",
        "  _AUDIO_MODE_Z_ON",
        *[f"  {c}" for c in cmds_short],
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

    # 3. Render the 3 separate channels + full mix using the Virtual Instrument / SoundFont
    print("[YoungGirlA] Renderizando simulacao no Instrumento Virtual (Canais A, B, Z separados e Mix)...", flush=True)
    sim_paths = simulate_channels_with_soundfont(cmds_short, out_prefix="YOUNG_GIRL_A_Sim")
    for k, p in sim_paths.items():
        print(f"  -> {k}: {p} ({p.stat().st_size / 1024.0:.1f} KB)", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
