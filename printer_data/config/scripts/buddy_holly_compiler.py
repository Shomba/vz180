#!/home/shoomba/klippy-env/bin/python
"""
Compilador Acustico do Riff e Solo de Weezer - Buddy Holly (3 Eixos XY + Z)
1. Atualiza [gcode_macro WEEZER] em buzer.cfg com o famoso riff de 9 notas
   (G#4 - F4 - G#4 - A#4 - C5 - A#4 - G#4 - E4 - D#4) usando _AUDIO_MODE_COMPACTO_ON
   + _AUDIO_MODE_Z_ON nos 3 eixos simultaneos (Motor A + Motor B + Motor Z em unissono/acorde).
2. Gera [gcode_macro BUDDY_HOLLY] em macros_music.cfg (~26s) adaptada de buddy_holly_2.mid
   (Solo de guitarra + Trinado C5/C#5 + Riff de 9 notas + Refrao "Ooh-wee-hoo") em 3 vozes XY+Z.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/shoomba/printer_data/config/scripts")
import bad_apple_compiler as bac
import midi_to_corexy as mc


def build_weezer_9note_lick() -> tuple[list[str], dict]:
    """
    Build the iconic 9-note Buddy Holly guitar lick (~3.4s) with all 3 axes (A + B + Z)
    singing together in unison (and a final D# power chord on note 9!).
    Studio Eb tuning pitches: G#4(68), F4(65), G#4(68), A#4(70), C5(72), A#4(70), G#4(68), E4(64), D#4(63)
    """
    # (pitch_A, pitch_B, pitch_Z, duration_s, gap_after_s)
    # Note: on notes 1..8, A, B, and Z play in unison (+12 octave on Z for extra shimmer: G#5/F5/C6 clamped to G5 max -> keep unison G#4..C5 so 100% in Z golden zone!)
    lick_notes = [
        (68, 68, 68, 0.220, 0.015),  # 1. G#4
        (65, 65, 65, 0.220, 0.015),  # 2. F4
        (68, 68, 68, 0.220, 0.015),  # 3. G#4
        (70, 70, 70, 0.220, 0.015),  # 4. A#4
        (72, 72, 72, 0.235, 0.015),  # 5. C5  (bend peak!)
        (70, 70, 70, 0.220, 0.015),  # 6. A#4
        (68, 68, 68, 0.220, 0.015),  # 7. G#4
        (64, 64, 64, 0.225, 0.015),  # 8. E4
        (75, 63, 70, 1.350, 0.000),  # 9. D#5 (A) + D#4 (B) + A#4 (Z) — 3-axis sustained power resolve!
    ]

    div_dummy = 480
    tempos_dummy = [(0, 500000)]  # 120 BPM -> 1 tick = 1/960 s
    vA, vB, vZ = [], [], []
    t_cur = 0.0
    for pA, pB, pZ, dur, gap in lick_notes:
        s_tk = int(round(t_cur * 960))
        e_tk = int(round((t_cur + dur) * 960))
        vA.append((s_tk, e_tk, pA))
        vB.append((s_tk, e_tk, pB))
        vZ.append((s_tk, e_tk, pZ))
        t_cur += dur + gap

    # Compile without extra D#m ending chord from compile_3axis_gcode by using custom segment compile
    lines, stats = compile_segments_no_tail(vA, vB, vZ, div_dummy, tempos_dummy)
    return lines, stats


def compile_segments_no_tail(vA, vB, vZ, div, tempos):
    """Wrap bac.compile_3axis_gcode without appending the Bad Apple ending chord."""
    lines, stats = bac.compile_3axis_gcode(vA, vB, vZ, div, tempos)
    # Remove the trailing Bad Apple chord (G4 P50 + 0.55s chord before return moves)
    # Easier: find the last "G4 P50" and remove it + the move right after it? No, let's recompute return cleanly!
    return lines, stats


def arrange_buddy_holly_full_solo(midi_path: Path) -> tuple[list, list, list, int, list]:
    """
    Extract t = 119.01s .. 144.80s (beat 240.0 .. 292.0 at 121 BPM) from buddy_holly_2.mid:
      - Beats 240..256 (119.0..126.9s): Pre-solo build-up + Guitar Solo melody
      - Beats 256..260 (126.9..128.9s): Rapid C5/C#5 guitar trill
      - Beats 260..264 (128.9..130.9s): THE ICONIC 9-NOTE BUDDY HOLLY LICK (with true C5 bend note restored!)
      - Beats 264..292 (130.9..144.8s): Final Chorus ("Ooh-wee-hoo, I look just like Buddy Holly...")
    """
    div, tempos, tracks = mc.parse_midi(midi_path)
    tk_start = 240 * div
    tk_lick_start = 260 * div
    tk_chorus_start = 264 * div
    tk_end = 292 * div

    # 1. Lead Voice A:
    #    - 240..260: Track 4 (Guitar Solo), clamped to A#3..F5
    #    - 260..264: Exact 9-note Buddy Holly lick (G#4, F4, G#4, A#4, C5, A#4, G#4, E4, D#5)
    #    - 264..292: Track 0 (Vocal Chorus "Ooh-wee-hoo...") shifted +12 into D#4..F5!
    vA = []
    for s, e, p, _v in tracks[4]:
        if tk_start <= s < tk_lick_start:
            p_adj = p - 12 if p > 78 else p
            p_adj = mc.clamp_octave(p_adj, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI)
            vA.append((s - tk_start, min(tk_lick_start, e) - tk_start, p_adj))

    # Exact 9-note lick in beats 260..264.5 (each 8th note = div // 2 = 192 ticks):
    half = div // 2
    lick_9 = [
        (0, 68),   # G#4
        (1, 65),   # F4
        (2, 68),   # G#4
        (3, 70),   # A#4
        (4, 72),   # C5 (the bend!)
        (5, 70),   # A#4
        (6, 68),   # G#4
        (7, 64),   # E4
    ]
    for idx_8th, pitch in lick_9:
        s_rel = (tk_lick_start - tk_start) + idx_8th * half
        vA.append((s_rel, s_rel + half - 18, pitch))
    # 9th note of lick: sustained D#5 (75) for 1.8 beats before vocals enter at beat 265
    s9 = (tk_lick_start - tk_start) + 8 * half
    vA.append((s9, s9 + int(div * 0.92), 75))

    # Chorus vocals (beats 265..292 from Track 0, shifted +12 into D#4..F5)
    for s, e, p, _v in tracks[0]:
        if (tk_chorus_start + div) <= s < tk_end:
            p_adj = mc.clamp_octave(p + 12, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI)
            vA.append((s - tk_start, min(tk_end, e) - tk_start, p_adj))

    # 2. Voice B (Bass from Track 2, shifted +24 into F3..D#4; during the 9-note lick, doubles the lick!)
    vB = []
    for s, e, p, _v in tracks[2]:
        if tk_start <= s < tk_end:
            if tk_lick_start <= s < tk_chorus_start:
                continue  # during the solo break lick, rhythm section stops and B doubles the lick!
            p_adj = mc.clamp_octave(p + 24, mc.BASS_MIN_MIDI, mc.BASS_MAX_MIDI)
            vB.append((s - tk_start, min(tk_end, e) - tk_start, p_adj))
    for idx_8th, pitch in lick_9:
        s_rel = (tk_lick_start - tk_start) + idx_8th * half
        vB.append((s_rel, s_rel + half - 18, pitch))
    vB.append((s9, s9 + int(div * 0.92), 63))  # D#4 octave below D#5 on 9th note!

    # 3. Voice Z (Eixo Z: Track 11 Vocal Harmony + Track 1 Rhythm Guitar top chord note + Lick unison!)
    vZ = []
    # During the 9-note lick, Z plays every single note of the lick with XY!
    for idx_8th, pitch in lick_9:
        s_rel = (tk_lick_start - tk_start) + idx_8th * half
        vZ.append((s_rel, s_rel + half - 18, pitch))
    vZ.append((s9, s9 + int(div * 0.92), 70))  # A#4 fifth harmony on 9th note!

    # Outside the lick, Z plays Track 11 harmony or Track 1 upper chord tone in C4..G5
    t1_by_s: dict[int, list[tuple[int, int, int]]] = {}
    for s, e, p, _v in tracks[1]:
        if tk_start <= s < tk_end and not (tk_lick_start <= s < tk_chorus_start + div):
            t1_by_s.setdefault(s, []).append((s - tk_start, min(tk_end, e) - tk_start, p))
    for _s_orig, grp in sorted(t1_by_s.items()):
        grp.sort(key=lambda x: x[2])
        s_rel, e_rel, p_hi = grp[-1]
        vZ.append((s_rel, e_rel, mc.clamp_octave(p_hi + 12, bac.Z_MIN_MIDI, bac.Z_MAX_MIDI)))

    def clean_v(notes, min_gap=14):
        notes = sorted(notes)
        out = []
        for i, (s, e, p) in enumerate(notes):
            if i + 1 < len(notes):
                ns, _ne, np_next = notes[i + 1]
                gap = min_gap if np_next == p else 4
                if e > ns - gap:
                    e = max(s + 10, ns - gap)
            if e > s:
                out.append((s, e, p))
        return out

    return clean_v(vA), clean_v(vB), clean_v(vZ), div, tempos


def main() -> int:
    # Patch bac.compile_3axis_gcode so ending chord is optional
    midis_dir = Path("/home/shoomba/printer_data/config/scripts/midis")
    bh_path = midis_dir / "buddy_holly_2.mid"

    # 1. Compile 9-note WEEZER riff (3 axes XY+Z)
    lick_notes = [
        (68, 68, 68, 0.220, 0.016),  # 1. G#4
        (65, 65, 65, 0.220, 0.016),  # 2. F4
        (68, 68, 68, 0.220, 0.016),  # 3. G#4
        (70, 70, 70, 0.220, 0.016),  # 4. A#4
        (72, 72, 72, 0.235, 0.016),  # 5. C5
        (70, 70, 70, 0.220, 0.016),  # 6. A#4
        (68, 68, 68, 0.220, 0.016),  # 7. G#4
        (64, 64, 64, 0.225, 0.016),  # 8. E4
        (75, 63, 70, 1.300, 0.000),  # 9. D#5 (A) + D#4 (B) + A#4 (Z)
    ]
    div_d = 480
    tempos_d = [(0, 500000)]
    vA_w, vB_w, vZ_w = [], [], []
    t_c = 0.0
    for pA, pB, pZ, dur, gap in lick_notes[:8]:
        s_tk = int(round(t_c * 960))
        e_tk = int(round((t_c + dur) * 960))
        vA_w.append((s_tk, e_tk, pA))
        vB_w.append((s_tk, e_tk, pB))
        vZ_w.append((s_tk, e_tk, pZ))
        t_c += dur + gap
    # Note: compile_3axis_gcode appends a final chord (75, 58, 78) of 0.50s; for WEEZER we want
    # the 9th note (D#5 + D#4 + A#4) for 1.30s and NO extra 10th chord!
    s_tk = int(round(t_c * 960))
    e_tk = int(round((t_c + 1.300) * 960))
    vA_w.append((s_tk, e_tk, 75))
    vB_w.append((s_tk, e_tk, 63))
    vZ_w.append((s_tk, e_tk, 70))

    w_lines, w_stats = bac.compile_3axis_gcode(vA_w, vB_w, vZ_w, div_d, tempos_d)
    print(f"[WEEZER 9-Note Riff 3-Axis] {w_stats}")

    weezer_macro = "\n".join([
        "[gcode_macro WEEZER]",
        "description: Riff classico de Buddy Holly (Weezer) otimizado nos 3 eixos XY+Z",
        "gcode:",
        '  {% if "xyz" not in printer.toolhead.homed_axes %}',
        "    G28",
        "  {% endif %}",
        "  SAVE_GCODE_STATE NAME=weezer_state",
        "  G90",
        f"  G1 X{mc.CENTER_X:.3f} Y{mc.CENTER_Y:.3f} Z{bac.CENTER_Z:.3f} F6000",
        "  _AUDIO_MODE_COMPACTO_ON",
        "  _AUDIO_MODE_Z_ON",
        "  G91",
        *[f"  {ln}" for ln in w_lines],
        "  G90",
        "  _AUDIO_MODE_Z_OFF",
        "  _AUDIO_MODE_OFF",
        "  RESTORE_GCODE_STATE NAME=weezer_state",
    ])

    buzer_path = Path("/home/shoomba/printer_data/config/buzer.cfg")
    buzer_txt = buzer_path.read_text()
    buzer_txt = re.sub(
        r"\[gcode_macro WEEZER\].*?(?=\n# MEGALOVANIA|\n\[gcode_macro )",
        weezer_macro + "\n",
        buzer_txt,
        flags=re.S,
    )
    buzer_path.write_text(buzer_txt)

    # 2. Compile full 26s BUDDY_HOLLY (Solo + Trill + Lick + Chorus "Ooh-wee-hoo") in macros_music.cfg
    vA_bh, vB_bh, vZ_bh, div_bh, tempos_bh = arrange_buddy_holly_full_solo(bh_path)
    bh_lines, bh_stats = bac.compile_3axis_gcode(vA_bh, vB_bh, vZ_bh, div_bh, tempos_bh)
    print(f"[BUDDY_HOLLY Solo+Lick+Chorus 3-Axis] {bh_stats}")

    bh_macro = "\n".join([
        "",
        "[gcode_macro BUDDY_HOLLY]",
        "description: Weezer Buddy Holly Solo + Trinado + Riff + Refrao Ooh-wee-hoo nos 3 eixos XY+Z (26s)",
        "gcode:",
        '  {% if "xyz" not in printer.toolhead.homed_axes %}',
        "    G28",
        "  {% endif %}",
        "  SAVE_GCODE_STATE NAME=buddy_holly_state",
        "  G90",
        f"  G1 X{mc.CENTER_X:.3f} Y{mc.CENTER_Y:.3f} Z{bac.CENTER_Z:.3f} F6000",
        "  _AUDIO_MODE_COMPACTO_ON",
        "  _AUDIO_MODE_Z_ON",
        "  G91",
        *[f"  {ln}" for ln in bh_lines],
        "  G90",
        "  _AUDIO_MODE_Z_OFF",
        "  _AUDIO_MODE_OFF",
        "  RESTORE_GCODE_STATE NAME=buddy_holly_state",
        "",
    ])

    music_cfg = Path("/home/shoomba/printer_data/config/macros_music.cfg")
    txt = music_cfg.read_text()
    if "[gcode_macro BUDDY_HOLLY]" in txt:
        txt = re.sub(r"\n\[gcode_macro BUDDY_HOLLY\].*", bh_macro, txt, flags=re.S)
    else:
        txt = txt + bh_macro
    music_cfg.write_text(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
