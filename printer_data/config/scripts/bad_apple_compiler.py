#!/home/shoomba/klippy-env/bin/python
"""
Compilador Acustico de Bad Apple!! usando o segundo arquivo enviado pelo usuario:
  /home/shoomba/printer_data/config/badapple!!midifull.mid (138.0 BPM, div=96)

Arquitetura Musical (3 Eixos XY + Z, com max_z_accel = 1000 mm/s^2):
  - Eixo Z (Motor Z, _AUDIO_MODE_Z_ON: 2ms_loud_2x, 0.95A):
      Toca a BATIDA DE FUNDO que comeca a musica no compasso 8 (Track 4 Synth2)
      e continua em todas as secoes que a utilizam (Track 4 Synth2 nos compassos
      8-48 e 64-104; Track 5 Synth1 nos compassos 48-64 e 104-120), com notas
      estendidas sem micro-buracos (gap=0) em B3..G#5.
  - Motor A (+45 deg CoreXY, _AUDIO_MODE_COMPACTO_ON: 4ms_punchy_4x, 1.7A):
      Entra no compasso 16 com o riff Synth1 (Track 5, compassos 16-24 e 64-72)
      e canta toda a Melodia Vocal reconstruida (Track 6 Vocals, compassos 24-64 e 72-120,
      unindo celulas de 24 ticks do FL Studio em colcheias/seminimas reais!).
  - Motor B (-45 deg CoreXY, _AUDIO_MODE_COMPACTO_ON: 4ms_punchy_4x, 1.7A):
      Entra no compasso 24 com a guitarra base sincopada (Track 7 Guitar, B3..E4).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/shoomba/printer_data/config/scripts")
import midi_to_corexy as mc

ORIG_MS_Z = 32
TARGET_MS_Z = 2
ROT_Z = 5.33333
SCALE_Z = ORIG_MS_Z / float(TARGET_MS_Z)  # 16.0
PULSES_PER_CMD_MM_Z = (200.0 * ORIG_MS_Z) / ROT_Z  # 1200.00075 pulses/cmd_mm
SAFE_Z_CMD_ACCEL = 60.0  # 60.0 * 16 = 960 mm/s^2 physical (< max_z_accel=1000)

CENTER_Z = 25.0
MIN_Z, MAX_Z = 15.0, 38.0

Z_MIN_MIDI = 59  # B3  (246.94 Hz)
Z_MAX_MIDI = 80  # G#5 (830.61 Hz)


def reconstruct_t6_vocals(raw_t6: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int]]:
    """
    Reconstruct true legato eighth (48 ticks) and quarter (96 ticks) vocal notes
    from FL Studio's 24-tick step cells in Track 6 of badapple!!midifull.mid.
    """
    notes = sorted({(s, e, p) for s, e, p, _v in raw_t6})
    out: list[tuple[int, int, int]] = []
    i = 0
    while i < len(notes):
        s, e, p = notes[i]
        # Case 1: Quarter-note bookend (s..s+24) + gap 48 + (s+72..s+96) of same pitch
        if (
            (e - s == 24)
            and (i + 1 < len(notes))
            and notes[i + 1][2] == p
            and notes[i + 1][0] == s + 72
            and notes[i + 1][1] == s + 96
        ):
            out.append((s, s + 96, p))
            i += 2
            continue
        # Case 2: Eighth-note pair (s..s+24) + (s+24..s+48) on eighth-note boundary (s % 48 == 0)
        if (
            (e - s == 24)
            and (s % 48 == 0)
            and (i + 1 < len(notes))
            and notes[i + 1][2] == p
            and notes[i + 1][0] == s + 24
            and notes[i + 1][1] == s + 48
        ):
            out.append((s, s + 48, p))
            i += 2
            continue
        out.append((s, e, p))
        i += 1
    return out


def make_beat_legato(notes: list[tuple[int, int, int]], max_bridge_ticks: int = 24) -> list[tuple[int, int, int]]:
    """
    Bridge 24-tick FL Studio gate gaps inside the background beat (Track 4/5)
    so the beat flows with gap=0 (48, 48, 24, 24, 24 ticks) without forcing XY stops.
    """
    notes = sorted(notes)
    out: list[tuple[int, int, int]] = []
    for i, (s, e, p) in enumerate(notes):
        if i + 1 < len(notes):
            ns = notes[i + 1][0]
            if 0 < (ns - e) <= max_bridge_ticks:
                e = ns
            elif e > ns:
                e = ns
        if e > s:
            out.append((s, e, p))
    return out


def arrange_bad_apple_v2(
    midi_path: Path,
    start_bar: int = 8,
    end_bar: int = 56,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]], int, list[tuple[int, int]]]:
    """
    Extract the 3 synchronized voices from /home/shoomba/printer_data/config/badapple!!midifull.mid:
      - Eixo Z (Voice Z): Background Beat starting the song solo in Bars 8-16 (Track 4 Synth2 +24)
        and continuing across all sections (Track 4 in Bars 16-48 & 64-104; Track 5 Synth1 -12 in Bars 48-64 & 104-120).
      - Motor A (Voice A): High Synth Hook in Bars 16-24 & 64-72 (Track 5 Synth1 -12 -> D#4..F#5)
        + Reconstructed Main Vocal Melody in Bars 24-64 & 72-120 (Track 6 Vocals -> B3..F#5)
        + Outro Guitar Lead in Bars 120-125 (Track 7).
      - Motor B (Voice B): Syncopated Rhythm Guitar Harmony in Bars 24-125 (Track 7 Guitar -> B3..E4).
    """
    div, tempos, tracks = mc.parse_midi(midi_path)
    bar_ticks = div * 4  # 384 ticks
    tk_start = start_bar * bar_ticks
    tk_end = end_bar * bar_ticks

    # 1. Voice Z: Background Beat (Track 4 Synth2 in bars 8..48 & 64..104; Track 5 Synth1 in bars 48..64 & 104..120)
    raw_Z: list[tuple[int, int, int]] = []
    for s, e, p, _v in tracks[4]:
        bar = s // bar_ticks
        if ((8 <= bar < 48) or (64 <= bar < 104)) and tk_start <= s < tk_end:
            p_adj = mc.clamp_octave(p + 24, Z_MIN_MIDI, Z_MAX_MIDI)
            raw_Z.append((s - tk_start, min(tk_end, e) - tk_start, p_adj))

    for s, e, p, _v in tracks[5]:
        bar = s // bar_ticks
        if ((48 <= bar < 64) or (104 <= bar < 120)) and tk_start <= s < tk_end:
            p_adj = mc.clamp_octave(p - 12, Z_MIN_MIDI, Z_MAX_MIDI)
            raw_Z.append((s - tk_start, min(tk_end, e) - tk_start, p_adj))

    voice_Z = make_beat_legato(raw_Z, max_bridge_ticks=24)

    # 2. Voice A: Synth1 Intro/Interlude (Bars 16..24 & 64..72) + Reconstructed Vocals (Bars 24..64 & 72..120)
    raw_A: list[tuple[int, int, int]] = []
    t5_intro: list[tuple[int, int, int]] = []
    for s, e, p, _v in tracks[5]:
        bar = s // bar_ticks
        if ((16 <= bar < 24) or (64 <= bar < 72)) and tk_start <= s < tk_end:
            p_adj = mc.clamp_octave(p - 12, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI)
            t5_intro.append((s - tk_start, min(tk_end, e) - tk_start, p_adj))
    raw_A.extend(make_beat_legato(t5_intro, max_bridge_ticks=24))

    vocals = reconstruct_t6_vocals(tracks[6])
    for s, e, p in vocals:
        bar = s // bar_ticks
        if ((24 <= bar < 64) or (72 <= bar < 120)) and tk_start <= s < tk_end:
            p_adj = mc.clamp_octave(p, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI)
            raw_A.append((s - tk_start, min(tk_end, e) - tk_start, p_adj))

    # 3. Voice B: Track 7 Guitar (Bars 24..125)
    raw_B: list[tuple[int, int, int]] = []
    t7_by_s: dict[int, list[tuple[int, int, int]]] = {}
    for s, e, p, _v in tracks[7]:
        bar = s // bar_ticks
        if 24 <= bar < 126 and tk_start <= s < tk_end:
            t7_by_s.setdefault(s, []).append((s - tk_start, min(tk_end, e) - tk_start, p))

    for s_orig, grp in sorted(t7_by_s.items()):
        grp.sort(key=lambda x: x[2])
        bar = s_orig // bar_ticks
        s_rel, e_rel, p_hi = grp[-1]
        if bar < 120:
            raw_B.append((s_rel, e_rel, mc.clamp_octave(p_hi, mc.BASS_MIN_MIDI, mc.BASS_MAX_MIDI)))
        else:
            raw_A.append((s_rel, e_rel, mc.clamp_octave(p_hi + 12, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI)))
            p_lo = grp[0][2]
            raw_B.append((s_rel, e_rel, mc.clamp_octave(p_lo + 12, mc.BASS_MIN_MIDI, mc.BASS_MAX_MIDI)))

    def clean_contiguous(notes: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        notes = sorted(notes)
        out: list[tuple[int, int, int]] = []
        for i, (s, e, p) in enumerate(notes):
            if i + 1 < len(notes):
                ns = notes[i + 1][0]
                if e > ns:
                    e = ns
            if e > s:
                out.append((s, e, p))
        return out

    return (
        clean_contiguous(raw_A),
        clean_contiguous(raw_B),
        clean_contiguous(voice_Z),
        div,
        tempos,
    )


def compile_3axis_gcode(
    voice_A: list[tuple[int, int, int]],
    voice_B: list[tuple[int, int, int]],
    voice_Z: list[tuple[int, int, int]],
    div_h: int,
    tempos_h: list[tuple[int, int]],
    orig_ms_xy: int = 16,
    rot_xy: float = 40.0,
    target_ms_xy: int = 4,
    add_ending_chord: bool = True,
) -> tuple[list[str], dict]:
    t2s = mc.make_tick_to_sec(div_h, tempos_h)
    scale_xy = orig_ms_xy / float(target_ms_xy)  # 4.0
    pulses_per_cmd_mm_xy = (200.0 * orig_ms_xy) / rot_xy  # 80.0

    ticks = set()
    for v in (voice_A, voice_B, voice_Z):
        for s, e, _ in v:
            ticks.add(s)
            ticks.add(e)
    sorted_ticks = sorted(ticks)

    starts_A = {s for s, _, _ in voice_A}
    starts_B = {s for s, _, _ in voice_B}
    starts_Z = {s for s, _, _ in voice_Z}

    segments = []
    iA = iB = iZ = 0
    for i in range(len(sorted_ticks) - 1):
        tk0, tk1 = sorted_ticks[i], sorted_ticks[i + 1]
        dt = t2s(tk1) - t2s(tk0)
        if dt < 0.005:
            continue
        t_mid = 0.5 * (tk0 + tk1)
        while iA < len(voice_A) and voice_A[iA][1] <= tk0:
            iA += 1
        while iB < len(voice_B) and voice_B[iB][1] <= tk0:
            iB += 1
        while iZ < len(voice_Z) and voice_Z[iZ][1] <= tk0:
            iZ += 1
        pA = voice_A[iA][2] if (iA < len(voice_A) and voice_A[iA][0] <= t_mid < voice_A[iA][1]) else None
        pB = voice_B[iB][2] if (iB < len(voice_B) and voice_B[iB][0] <= t_mid < voice_B[iB][1]) else None
        pZ = voice_Z[iZ][2] if (iZ < len(voice_Z) and voice_Z[iZ][0] <= t_mid < voice_Z[iZ][1]) else None
        segments.append((t2s(tk0), t2s(tk1), pA, pB, pZ, tk0 in starts_A, tk0 in starts_B, tk0 in starts_Z))

    if add_ending_chord and segments:
        t_end = segments[-1][1]
        segments.append((t_end, t_end + 0.04, None, None, None, False, False, False))
        segments.append((t_end + 0.04, t_end + 0.55, 75, 63, 75, True, True, True))

    x_phys, y_phys, z_phys = mc.CENTER_X, mc.CENTER_Y, CENTER_Z
    cum_dx_cmd = cum_dy_cmd = cum_dz_cmd = 0.0
    min_x = max_x = x_phys
    min_y = max_y = y_phys
    min_z = max_z = z_phys
    last_sA = last_sB = last_sZ = 1
    last_pA = last_pB = last_pZ = None
    cur_accel_bucket = -1

    lines: list[str] = []
    n_moves = n_rests = 0

    for t0, t1, pA, pB, pZ, onA, onB, onZ in segments:
        dt = t1 - t0
        if dt < 0.005:
            continue
        if pA is None and pB is None and pZ is None:
            ms = int(round(dt * 1000.0))
            if ms >= 5:
                lines.append(f"G4 P{ms}")
                n_rests += 1
            continue

        fA = mc.midi_to_hz(pA) if pA is not None else 0.0
        fB = mc.midi_to_hz(pB) if pB is not None else 0.0
        fZ = mc.midi_to_hz(pZ) if pZ is not None else 0.0

        mag_A_cmd = (fA / pulses_per_cmd_mm_xy) * dt
        mag_B_cmd = (fB / pulses_per_cmd_mm_xy) * dt
        mag_Z_cmd = (fZ / PULSES_PER_CMD_MM_Z) * dt

        cand_sA = [1, -1] if pA is not None else [0]
        cand_sB = [1, -1] if pB is not None else [0]
        best_xy = None
        best_xy_cost = 1e18
        for sA in cand_sA:
            for sB in cand_sB:
                dx_cmd = 0.5 * (sA * mag_A_cmd + sB * mag_B_cmd)
                dy_cmd = 0.5 * (sA * mag_A_cmd - sB * mag_B_cmd)
                nx = x_phys + dx_cmd * scale_xy
                ny = y_phys + dy_cmd * scale_xy
                if not (mc.MIN_X <= nx <= mc.MAX_X and mc.MIN_Y <= ny <= mc.MAX_Y):
                    continue
                cost = (nx - mc.CENTER_X) ** 2 + (ny - mc.CENTER_Y) ** 2
                if pA is not None and onA and pA == last_pA and sA == last_sA:
                    cost += 1200.0
                if pB is not None and onB and pB == last_pB and sB == last_sB:
                    cost += 1200.0
                if pA is not None and not onA and sA != last_sA:
                    cost += 2500.0
                if pB is not None and not onB and sB != last_sB:
                    cost += 2500.0
                if cost < best_xy_cost:
                    best_xy_cost = cost
                    best_xy = (sA, sB, dx_cmd, dy_cmd)

        cand_sZ = [1, -1] if pZ is not None else [0]
        best_sZ = 0
        best_z_cost = 1e18
        for sZ in cand_sZ:
            nz = z_phys + (sZ * mag_Z_cmd) * SCALE_Z
            if not (MIN_Z <= nz <= MAX_Z):
                continue
            cost = (nz - CENTER_Z) ** 2
            if pZ is not None and onZ and pZ == last_pZ and sZ == last_sZ:
                cost += 80.0
            if pZ is not None and not onZ and sZ != last_sZ:
                cost += 120.0
            if cost < best_z_cost:
                best_z_cost = cost
                best_sZ = sZ

        sA, sB, dx_cmd, dy_cmd = best_xy
        dz_cmd = best_sZ * mag_Z_cmd

        dx_q = round(dx_cmd, 5)
        dy_q = round(dy_cmd, 5)
        dz_q = round(dz_cmd, 5)
        L_cmd = math.sqrt(dx_q * dx_q + dy_q * dy_q + dz_q * dz_q)
        if L_cmd < 1e-6:
            continue

        if abs(dz_q) > 1e-6:
            raw_safe_accel = min(2500.0, SAFE_Z_CMD_ACCEL * (L_cmd / abs(dz_q)))
        else:
            raw_safe_accel = 2500.0
        for b in (60, 220, 700, 2500):
            if raw_safe_accel >= b:
                bucket = b
        if bucket != cur_accel_bucket:
            lines.append(f"SET_VELOCITY_LIMIT ACCEL={bucket}")
            cur_accel_bucket = bucket

        f_cmd = round((L_cmd / dt) * 60.0, 2)
        lines.append(f"G1 X{dx_q:.5f} Y{dy_q:.5f} Z{dz_q:.5f} F{f_cmd:.2f}")
        n_moves += 1

        cum_dx_cmd = round(cum_dx_cmd + dx_q, 5)
        cum_dy_cmd = round(cum_dy_cmd + dy_q, 5)
        cum_dz_cmd = round(cum_dz_cmd + dz_q, 5)
        x_phys = mc.CENTER_X + cum_dx_cmd * scale_xy
        y_phys = mc.CENTER_Y + cum_dy_cmd * scale_xy
        z_phys = CENTER_Z + cum_dz_cmd * SCALE_Z
        min_x, max_x = min(min_x, x_phys), max(max_x, x_phys)
        min_y, max_y = min(min_y, y_phys), max(max_y, y_phys)
        min_z, max_z = min(min_z, z_phys), max(max_z, z_phys)

        if pA is not None:
            last_sA, last_pA = sA, pA
        if pB is not None:
            last_sB, last_pB = sB, pB
        if pZ is not None:
            last_sZ, last_pZ = best_sZ, pZ

    ret_dx = round(-cum_dx_cmd, 5)
    ret_dy = round(-cum_dy_cmd, 5)
    ret_dz = round(-cum_dz_cmd, 5)
    if abs(ret_dz) > 1e-5:
        lines.append("SET_VELOCITY_LIMIT ACCEL=60")
        lines.append(f"G1 Z{ret_dz:.5f} F150.00")
    if abs(ret_dx) > 1e-5 or abs(ret_dy) > 1e-5:
        lines.append("SET_VELOCITY_LIMIT ACCEL=1500")
        lines.append(f"G1 X{ret_dx:.5f} Y{ret_dy:.5f} F1200.00")

    stats = {
        "moves": n_moves,
        "rests": n_rests,
        "duration_s": round(segments[-1][1] - segments[0][0], 1) if segments else 0.0,
        "x_phys_range": (round(min_x, 1), round(max_x, 1)),
        "y_phys_range": (round(min_y, 1), round(max_y, 1)),
        "z_phys_range": (round(min_z, 2), round(max_z, 2)),
        "return_cmd": (ret_dx, ret_dy, ret_dz),
    }
    return lines, stats


def main() -> int:
    user_midi = Path("/home/shoomba/printer_data/config/badapple!!midifull.mid")
    midis_dir = Path("/home/shoomba/printer_data/config/scripts/midis")
    out_cfg = Path("/home/shoomba/printer_data/config/macros_music.cfg")
    out_full_gcode = Path("/home/shoomba/printer_data/gcodes/BAD_APPLE_FULL.gcode")

    # 1. Keep MEGALOVANIA at top of macros_music.cfg
    div_m, tempos_m, trk_m = mc.parse_midi(midis_dir / "megalovania_mythic.mid")
    t2s_m = mc.make_tick_to_sec(div_m, tempos_m)
    lead_m, bass_m = mc.arrange_megalovania(trk_m, div_m, max_sec=64.0, t2s=t2s_m)
    segs_m = mc.build_events(lead_m, bass_m, t2s_m, add_ending_chord=True)
    lines_m, _ = mc.compile_corexy_gcode(segs_m)
    mc.write_klipper_macro(
        out_cfg,
        "MEGALOVANIA",
        "Undertale Megalovania 2 vozes CoreXY (4ms_punchy_4x otimizado, 64s tema principal)",
        lines_m,
        append=False,
    )

    # 2. BAD_APPLE (Bars 8..56 = 84.0s at 138 BPM:
    #    8 bars Z solo background beat -> 8 bars Z + A Synth1 hook -> 16 bars Z + A Vocal + B Guitar Verse -> 16 bars Z + A Vocal + B Guitar Chorus!)
    vA, vB, vZ, div_b, tempos_b = arrange_bad_apple_v2(user_midi, start_bar=8, end_bar=56)
    xyz_lines, xyz_stats = compile_3axis_gcode(vA, vB, vZ, div_b, tempos_b)
    print(f"[BAD_APPLE 84s Macro (from badapple!!midifull.mid)] {xyz_stats}")
    print(
        f"  Voices -> Z(Batida de Fundo T4/T5): {mc.midi_to_name(min(p for _,_,p in vZ))}..{mc.midi_to_name(max(p for _,_,p in vZ))} ({len(vZ)} notas) | "
        f"A(Synth1 + Vocal T6 reconstruido): {mc.midi_to_name(min(p for _,_,p in vA))}..{mc.midi_to_name(max(p for _,_,p in vA))} ({len(vA)} notas) | "
        f"B(Guitarra Base T7): {mc.midi_to_name(min(p for _,_,p in vB))}..{mc.midi_to_name(max(p for _,_,p in vB))} ({len(vB)} notas)"
    )

    macro_xyz = [
        "[gcode_macro BAD_APPLE]",
        "description: Bad Apple!! (de badapple!!midifull.mid) — Z na batida de fundo T4/T5 + A no Synth1/Vocal T6 + B na Guitarra T7 (84s)",
        "gcode:",
        '  {% if "xyz" not in printer.toolhead.homed_axes %}',
        "    G28",
        "  {% endif %}",
        "  SAVE_GCODE_STATE NAME=bad_apple_xyz_state",
        "  G90",
        f"  G1 X{mc.CENTER_X:.3f} Y{mc.CENTER_Y:.3f} Z{CENTER_Z:.3f} F6000",
        "  _AUDIO_MODE_COMPACTO_ON",
        "  _AUDIO_MODE_Z_ON",
        "  G91",
        *[f"  {ln}" for ln in xyz_lines],
        "  G90",
        "  _AUDIO_MODE_Z_OFF",
        "  _AUDIO_MODE_OFF",
        "  RESTORE_GCODE_STATE NAME=bad_apple_xyz_state",
        "",
    ]

    # 3. Full 204s BAD_APPLE_FULL.gcode (Bars 8..125, complete song from badapple!!midifull.mid)
    vA_f, vB_f, vZ_f, div_f, tempos_f = arrange_bad_apple_v2(user_midi, start_bar=8, end_bar=125)
    full_lines, full_stats = compile_3axis_gcode(vA_f, vB_f, vZ_f, div_f, tempos_f)
    print(f"[BAD_APPLE_FULL.gcode 204s] {full_stats}")
    full_gcode = [
        "; Bad Apple!! Completa (de /home/shoomba/printer_data/config/badapple!!midifull.mid)",
        "; Eixo Z = Batida de fundo (T4 Synth2 / T5 Synth1) | Motor A = Synth1 + Vocal T6 reconstruido | Motor B = Guitarra T7",
        "G90",
        "G28",
        f"G1 X{mc.CENTER_X:.3f} Y{mc.CENTER_Y:.3f} Z{CENTER_Z:.3f} F6000",
        "_AUDIO_MODE_COMPACTO_ON",
        "_AUDIO_MODE_Z_ON",
        "G91",
        *full_lines,
        "G90",
        "_AUDIO_MODE_Z_OFF",
        "_AUDIO_MODE_OFF",
        "",
    ]
    out_full_gcode.write_text("\n".join(full_gcode))

    out_cfg.write_text(out_cfg.read_text() + "\n" + "\n".join(macro_xyz))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
