#!/home/shoomba/klippy-env/bin/python
"""
Compilador Acustico de Queen - Bohemian Rhapsody (3 Eixos Simultaneos XY + Z)
Adaptado de /home/shoomba/printer_data/config/scripts/midis/bohemian_1.mid:
  - Eixo Z (Motor Z, _AUDIO_MODE_Z_ON: 2ms_loud_2x, max_z_accel=1000):
      Toca o arpejo continuo do piano de Freddie Mercury (Track 1 A.PIANO 1) na faixa C4..G5.
  - Motor A (+45 deg CoreXY, _AUDIO_MODE_COMPACTO_ON: 4ms_punchy_4x):
      Toca a melodia vocal principal de Freddie Mercury (Track 4) + Solo de Guitarra de Brian May (Track 9)
      + Coral/Opera (Track 3/5) + Riff Rock (Track 13).
  - Motor B (-45 deg CoreXY, _AUDIO_MODE_COMPACTO_ON: 4ms_punchy_4x):
      Toca o baixo de John Deacon (Track 2 / mao esquerda do piano) + harmonias do coral/guitarra.

Gera:
  1. [gcode_macro BOHEMIAN_RHAPSODY] em macros_music.cfg (~71s):
       Intro de Piano -> "Mama, just killed a man..." -> "Mama, ooooh!" -> Solo Completo de Brian May!
  2. /home/shoomba/printer_data/gcodes/BOHEMIAN_RHAPSODY_FULL.gcode (~355s):
       Versao completa (Intro Coral -> Balada -> Solo -> Opera Galileo/Figaro -> Rock -> Outro).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/shoomba/printer_data/config/scripts")
import bad_apple_compiler as bac
import midi_to_corexy as mc


def extract_section_voices(
    tracks: list[list[tuple[int, int, int, int]]],
    div: int,
    tk_start: int,
    tk_end: int,
    tk_out_offset: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """
    Extract 3 voices (vA, vB, vZ) from bohemian_1.mid in [tk_start, tk_end),
    shifted to output timeline starting at tk_out_offset.
    """
    grid = div // 8  # 48 ticks (32nd-note quantization for human-played piano chords)

    def q_tick(tk: int) -> int:
        return int(round(tk / float(grid))) * grid

    # 1. Lead Voice A:
    #    Priority: Track 9 (Brian May Guitar Solo) > Track 4 (Freddie Lead Vocal) > Track 5/6 (Opera Choir Lead) > Track 13 (Rock Riff) > Track 3 (Intro Choir)
    vA: list[tuple[int, int, int]] = []

    def add_to_voice(target: list[tuple[int, int, int]], src_notes, pitch_fn, min_dur_ticks: int = 36):
        by_s: dict[int, list[tuple[int, int, int]]] = {}
        for s, e, p, _v in src_notes:
            if s < tk_start or s >= tk_end:
                continue
            qs = max(tk_start, q_tick(s))
            qe = min(tk_end, max(qs + min_dur_ticks, q_tick(e)))
            by_s.setdefault(qs, []).append((qs, qe, p))
        for qs, grp in sorted(by_s.items()):
            grp.sort(key=lambda x: x[2])
            s0, e0, p_ chosen = grp[-1]
            s_out = s0 - tk_start + tk_out_offset
            e_out = e0 - tk_start + tk_out_offset
            # Check if target already has an active note overlapping [s_out, e_out)
            if not any(ts < e_out - 24 and te > s_out + 24 for ts, te, _ in target):
                target.append((s_out, e_out, pitch_fn(s0, p_chosen, grp)))

    # Brian May Guitar Solo (Track 9):
    # Note: in beats 201..211 (t=164.2..173.1s), the solo climbs to G5..C6; shift that high sub-phrase -12 so contour stays intact inside A#3..F#5!
    def guitar_pitch_fn(tk_abs: int, p: int, _grp) -> int:
        beat = tk_abs / float(div)
        if 200.5 <= beat <= 211.8:
            return mc.clamp_octave(p - 12, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI)
        return mc.clamp_octave(p, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI)

    add_to_voice(vA, tracks[9], guitar_pitch_fn)
    add_to_voice(vA, tracks[4], lambda _tk, p, _g: mc.clamp_octave(p, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI))
    add_to_voice(vA, tracks[5], lambda _tk, p, _g: mc.clamp_octave(p, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI))
    add_to_voice(vA, tracks[6], lambda _tk, p, _g: mc.clamp_octave(p, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI))
    add_to_voice(vA, tracks[13], lambda _tk, p, _g: mc.clamp_octave(p, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI))
    add_to_voice(vA, tracks[3], lambda _tk, p, _g: mc.clamp_octave(p, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI))

    # 2. Voice Z (Eixo Z: Freddie's Rolling Piano Arpeggio from Track 1, + Track 3/14 when Piano rests)
    vZ: list[tuple[int, int, int]] = []
    p1_by_s: dict[int, list[tuple[int, int, int]]] = {}
    for s, e, p, _v in tracks[1]:
        if s < tk_start or s >= tk_end:
            continue
        qs = max(tk_start, q_tick(s))
        qe = min(tk_end, max(qs + 48, q_tick(e)))
        p1_by_s.setdefault(qs, []).append((qs, qe, p))

    piano_bass_notes: list[tuple[int, int, int, int]] = []
    for qs, grp in sorted(p1_by_s.items()):
        grp.sort(key=lambda x: x[2])
        # Separate left-hand bass notes (< F3=53) from right-hand arpeggio notes (>= F3=53)
        rh = [n for n in grp if n[2] >= 53]
        lh = [n for n in grp if n[2] < 53]
        if lh:
            piano_bass_notes.append((lh[0][0], lh[0][1], lh[0][2], 100))
        chosen = rh[-1] if rh else grp[-1]
        s_out = chosen[0] - tk_start + tk_out_offset
        e_out = min(chosen[1], chosen[0] + div) - tk_start + tk_out_offset
        # Map right-hand arpeggio into Z's crystal-clear C4..G5 (60..79) zone
        p_z = mc.clamp_octave(chosen[2], 60, 79)
        vZ.append((s_out, e_out, p_z))

    add_to_voice(vZ, tracks[3], lambda _tk, p, g: mc.clamp_octave(g[0][2], 60, 79))
    add_to_voice(vZ, tracks[14], lambda _tk, p, _g: mc.clamp_octave(p + 12, 60, 79))

    # 3. Voice B (John Deacon Bass Track 2 + Piano Left-Hand Bass + Track 15/3 Harmony)
    vB: list[tuple[int, int, int]] = []
    add_to_voice(vB, tracks[2], lambda _tk, p, _g: mc.clamp_octave(p + 24, mc.BASS_MIN_MIDI, mc.BASS_MAX_MIDI))
    add_to_voice(vB, piano_bass_notes, lambda _tk, p, _g: mc.clamp_octave(p + 24, mc.BASS_MIN_MIDI, mc.BASS_MAX_MIDI))
    add_to_voice(vB, tracks[15], lambda _tk, p, _g: mc.clamp_octave(p + 12, mc.BASS_MIN_MIDI, mc.BASS_MAX_MIDI))
    add_to_voice(vB, tracks[7], lambda _tk, p, _g: mc.clamp_octave(p + 12, mc.BASS_MIN_MIDI, mc.BASS_MAX_MIDI))

    return vA, vB, vZ


def clean_monophonic(notes: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    notes = sorted(notes)
    out: list[tuple[int, int, int]] = []
    for i, (s, e, p) in enumerate(notes):
        if i + 1 < len(notes):
            ns = notes[i + 1][0]
            if e > ns:
                e = ns
        if e > s + 12:
            out.append((s, e, p))
    return out


def main() -> int:
    midi_path = Path("/home/shoomba/printer_data/config/scripts/midis/bohemian_1.mid")
    div, tempos, tracks = mc.parse_midi(midi_path)

    # 1. Build [gcode_macro BOHEMIAN_RHAPSODY] (~71s):
    #    Part 1: Beats 65.0 .. 120.0 (55 beats: Piano Solo Intro -> "Mama, just killed a man..." -> "Mama, ooooh!")
    #    Part 2: Beats 190.0 .. 220.0 (30 beats: "...tomorrow!" -> Brian May's Complete Guitar Solo!)
    #    Notice the splice at beat 120.0 == beat 190.0 is beat-exact; use constant 72 BPM (us=833333) for smooth timing
    tk_p1_s = 65 * div
    tk_p1_e = 120 * div
    tk_p2_s = 190 * div
    tk_p2_e = 221 * div

    vA1, vB1, vZ1 = extract_section_voices(tracks, div, tk_p1_s, tk_p1_e, 0)
    # In the opening 8 beats (beats 65..73, "Mama" piano intro), let Z play the arpeggio solo for the first 4 beats, then B joins with left-hand bass!
    vB1 = [(s, e, p) for s, e, p in vB1 if s >= 4 * div]

    p1_len = tk_p1_e - tk_p1_s
    vA2, vB2, vZ2 = extract_section_voices(tracks, div, tk_p2_s, tk_p2_e, p1_len)

    vA_macro = clean_monophonic(vA1 + vA2)
    vB_macro = clean_monophonic(vB1 + vB2)
    vZ_macro = clean_monophonic(vZ1 + vZ2)

    tempos_72bpm = [(0, 833333)]  # 72.0 BPM
    macro_lines, macro_stats = bac.compile_3axis_gcode(
        vA_macro, vB_macro, vZ_macro, div, tempos_72bpm, add_ending_chord=False
    )
    print(f"[BOHEMIAN_RHAPSODY 71s Macro (Ballad + Chorus + Guitar Solo)] {macro_stats}")
    print(
        f"  Voices -> A(Freddie Vocal + Brian May Solo): {mc.midi_to_name(min(p for _,_,p in vA_macro))}..{mc.midi_to_name(max(p for _,_,p in vA_macro))} ({len(vA_macro)} notas) | "
        f"B(Deacon Bass + Harmonia): {mc.midi_to_name(min(p for _,_,p in vB_macro))}..{mc.midi_to_name(max(p for _,_,p in vB_macro))} ({len(vB_macro)} notas) | "
        f"Z(Arpejo de Piano): {mc.midi_to_name(min(p for _,_,p in vZ_macro))}..{mc.midi_to_name(max(p for _,_,p in vZ_macro))} ({len(vZ_macro)} notas)"
    )

    bh_macro_txt = "\n".join([
        "",
        "[gcode_macro BOHEMIAN_RHAPSODY]",
        "description: Queen Bohemian Rhapsody 3 eixos XY+Z (Arpejo Piano no Z + Vocal Mama + Solo Brian May, 71s)",
        "gcode:",
        '  {% if "xyz" not in printer.toolhead.homed_axes %}',
        "    G28",
        "  {% endif %}",
        "  SAVE_GCODE_STATE NAME=bohemian_state",
        "  G90",
        f"  G1 X{mc.CENTER_X:.3f} Y{mc.CENTER_Y:.3f} Z{bac.CENTER_Z:.3f} F6000",
        "  _AUDIO_MODE_COMPACTO_ON",
        "  _AUDIO_MODE_Z_ON",
        "  G91",
        *[f"  {ln}" for ln in macro_lines],
        "  G90",
        "  _AUDIO_MODE_Z_OFF",
        "  _AUDIO_MODE_OFF",
        "  RESTORE_GCODE_STATE NAME=bohemian_state",
        "",
    ])

    music_cfg = Path("/home/shoomba/printer_data/config/macros_music.cfg")
    txt = music_cfg.read_text()
    if "[gcode_macro BOHEMIAN_RHAPSODY]" in txt:
        txt = re.sub(r"\n\[gcode_macro BOHEMIAN_RHAPSODY\].*", bh_macro_txt, txt, flags=re.S)
    else:
        txt = txt + bh_macro_txt
    music_cfg.write_text(txt)

    # 2. Build full uncut BOHEMIAN_RHAPSODY_FULL.gcode (from Choir Intro beat 8 to Outro beat 440)
    vAf, vBf, vZf = extract_section_voices(tracks, div, 8 * div, 440 * div, 0)
    vAf, vBf, vZf = clean_monophonic(vAf), clean_monophonic(vBf), clean_monophonic(vZf)
    # Shift tempos by -8*div
    tempos_full = [(max(0, tk - 8 * div), us) for tk, us in tempos]
    full_lines, full_stats = bac.compile_3axis_gcode(
        vAf, vBf, vZf, div, tempos_full, add_ending_chord=False
    )
    print(f"[BOHEMIAN_RHAPSODY_FULL.gcode] {full_stats}")
    out_full = Path("/home/shoomba/printer_data/gcodes/BOHEMIAN_RHAPSODY_FULL.gcode")
    out_full.write_text("\n".join([
        "; Queen - Bohemian Rhapsody (Completa 3 Eixos XY+Z)",
        "; Eixo Z = Piano / Arpejo | Motor A = Vocal Freddie + Solo Brian May | Motor B = Baixo + Coral",
        "G90",
        "G28",
        f"G1 X{mc.CENTER_X:.3f} Y{mc.CENTER_Y:.3f} Z{bac.CENTER_Z:.3f} F6000",
        "_AUDIO_MODE_COMPACTO_ON",
        "_AUDIO_MODE_Z_ON",
        "G91",
        *full_lines,
        "G90",
        "_AUDIO_MODE_Z_OFF",
        "_AUDIO_MODE_OFF",
        "",
    ]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
