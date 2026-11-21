#!/home/shoomba/klippy-env/bin/python
"""
Compilador Acustico de Bad Apple!! (Touhou) para VZ180 AWD CoreXY + Eixo Z
Gera duas macros otimizadas em /home/shoomba/printer_data/config/macros_music.cfg
(alem do arquivo completo /home/shoomba/printer_data/gcodes/BAD_APPLE_FULL.gcode):
  1. BAD_APPLE_Z : 100% no Eixo Z otimizado (_AUDIO_MODE_Z_ON: 2ms_loud_2x, 0.95A,
                   toff=4, hstrt=7, hend=6, tbl=2), na faixa de ouro C4..G5 (68-91% clareza).
  2. BAD_APPLE   : 3 Eixos Simultaneos (Motor A + Motor B + Motor Z, XY+Z polifonico de 3 vozes!),
                   abrindo com o Eixo Z solo na intro e juntando Motor A (Vocal), Motor B (Baixo)
                   e Motor Z (Synth / Contra-melodia em C5..G5) com limitador dinamico de
                   aceleracao fisica do Z (<= 450 mm/s^2) e retorno exato de homing XYZ.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/shoomba/printer_data/config/scripts")
import midi_to_corexy as mc

# Z physical parameters (printer.cfg: microsteps=32, rotation_distance=5.33333; _AUDIO_MODE_Z_ON: mres=7 -> 2ms)
ORIG_MS_Z = 32
TARGET_MS_Z = 2
ROT_Z = 5.33333
SCALE_Z = ORIG_MS_Z / float(TARGET_MS_Z)  # 16.0
PULSES_PER_CMD_MM_Z = (200.0 * ORIG_MS_Z) / ROT_Z  # 1200.00075 pulses/cmd_mm

CENTER_Z = 25.0
MIN_Z, MAX_Z = 16.0, 36.0

# Z Golden Zone in 2ms_loud_2x (C4=60 .. G5=79, with B3=59 also clean)
Z_MIN_MIDI = 59   # B3  (246.94 Hz)
Z_MAX_MIDI = 79   # G5  (783.99 Hz)
Z_HIGH_MIN = 68   # G#4 (415.30 Hz) — high register for 3-voice Z counter-melody
Z_HIGH_MAX = 79   # G5  (783.99 Hz) — 84-91% fundamental clarity on Z!


def arrange_bad_apple_z_solo(
    annejan_path: Path,
    start_beat: float = 51.0,
    end_beat: float = 195.0,
) -> list[tuple[float, float, int]]:
    """
    Extract clean monophonic Bad Apple melody from bad_apple_annejan.mid (138 BPM):
      - Beats 51..67  (4 bars): Synth intro hook (notes < B3 shifted +12 into B3..A#4)
      - Beats 67..131 (16 bars): Full Vocal Verse (D4..D#5 — 100% native inside Z golden zone!)
      - Beats 131..195 (16 bars): Full Soaring Chorus (C#4..F#5 — 100% native inside Z golden zone!)
    Returns [(t_start_s, t_end_s, midi_pitch)] shifted so t=0 at start_beat.
    """
    div, tempos, tracks = mc.parse_midi(annejan_path)
    t2s = mc.make_tick_to_sec(div, tempos)
    t_offset = t2s(int(round(start_beat * div)))

    raw = []
    for s, e, p, _v in tracks[0]:
        b_s = s / float(div)
        b_e = e / float(div)
        if b_s < start_beat or b_s >= end_beat:
            continue
        p_adj = mc.clamp_octave(p, Z_MIN_MIDI, Z_MAX_MIDI)
        raw.append((s, e, p_adj))

    raw.sort()
    out: list[tuple[float, float, int]] = []
    for i, (s, e, p) in enumerate(raw):
        if i + 1 < len(raw):
            ns, _ne, np_next = raw[i + 1]
            gap_ticks = 22 if np_next == p else 8
            if e > ns - gap_ticks:
                e = max(s + 20, ns - gap_ticks)
        t0 = max(0.0, t2s(s) - t_offset)
        t1 = max(t0 + 0.02, t2s(e) - t_offset)
        out.append((t0, t1, p))

    # Add clean final resolving note D#5 (MIDI 75) if ending at beat 195
    if out:
        t_last = out[-1][1]
        out.append((t_last + 0.04, t_last + 0.50, 75))
    return out


def compile_z_solo_gcode(notes: list[tuple[float, float, int]]) -> tuple[list[str], dict]:
    """Compile monophonic Z notes into relative G1 Z / G4 P G-code for _AUDIO_MODE_Z_ON."""
    z_phys = CENTER_Z
    cum_dz_cmd = 0.0
    min_z, max_z = z_phys, z_phys
    max_vz_phys = 0.0
    last_sZ = 1
    last_pZ: int | None = None

    lines: list[str] = []
    t_cur = 0.0
    n_moves = 0
    n_rests = 0

    for t0, t1, p in notes:
        if t0 - t_cur >= 0.006:
            ms = int(round((t0 - t_cur) * 1000.0))
            if ms >= 6:
                lines.append(f"G4 P{ms}")
                n_rests += 1
        dt = t1 - t0
        if dt < 0.01:
            t_cur = t1
            continue

        hz = mc.midi_to_hz(p)
        mag_dz_cmd = (hz / PULSES_PER_CMD_MM_Z) * dt

        # Pick direction (+1 or -1) to keep Z centered around 25 mm inside [MIN_Z, MAX_Z]
        best_sZ = 1
        best_cost = 1e18
        for sZ in (1, -1):
            nz = z_phys + (sZ * mag_dz_cmd) * SCALE_Z
            if not (MIN_Z <= nz <= MAX_Z):
                continue
            cost = (nz - CENTER_Z) ** 2
            # Flip on repeated note for crisp staccato; keep direction on scale runs when near center
            if p == last_pZ and sZ == last_sZ:
                cost += 25.0
            elif p != last_pZ and sZ != last_sZ and abs(z_phys - CENTER_Z) < 2.2:
                cost += 1.5
            if cost < best_cost:
                best_cost = cost
                best_sZ = sZ

        dz_q = round(best_sZ * mag_dz_cmd, 5)
        if abs(dz_q) < 1e-5:
            t_cur = t1
            continue
        vz_cmd = abs(dz_q) / dt
        f_cmd = round(vz_cmd * 60.0, 2)
        vz_phys = vz_cmd * SCALE_Z
        max_vz_phys = max(max_vz_phys, vz_phys)

        lines.append(f"G1 Z{dz_q:.5f} F{f_cmd:.2f}")
        n_moves += 1

        cum_dz_cmd = round(cum_dz_cmd + dz_q, 5)
        z_phys = CENTER_Z + cum_dz_cmd * SCALE_Z
        min_z, max_z = min(min_z, z_phys), max(max_z, z_phys)
        last_sZ = best_sZ
        last_pZ = p
        t_cur = t1

    ret_dz = round(-cum_dz_cmd, 5)
    if abs(ret_dz) > 1e-5:
        lines.append(f"G1 Z{ret_dz:.5f} F120.00")

    stats = {
        "moves": n_moves,
        "rests": n_rests,
        "duration_s": round(t_cur, 1),
        "z_phys_range": (round(min_z, 2), round(max_z, 2)),
        "max_vz_phys_mm_s": round(max_vz_phys, 2),
        "return_dz_cmd": ret_dz,
    }
    return lines, stats


def arrange_bad_apple_3axis(
    annejan_path: Path,
    handhule_path: Path,
    start_beat_h: float = 80.0,
    end_beat_h: float = 224.0,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], list[tuple[int, int, int]], int, list[tuple[int, int]]]:
    """
    Build 3 synchronized monophonic voices on the handhule tick grid (div=96, 138 BPM):
      - Beats 80..88  (2 bars, 3.5s):  Eixo Z SOLO plays the iconic Synth1 hook (D#4..F#5)!
      - Beats 88..96  (2 bars, 3.5s):  Eixo Z (Synth1 hook) + Motor B (Bass2 enters)!
      - Beats 96..224 (32 bars, 55.6s): Full 3-Voice Polyphony:
          * Motor A (Lead Vocal from annejan aligned at beat+29, D4..F#5)
          * Motor B (Bass from handhule Track 10 Bass2 / Track 7 Guitar, A#2..F4)
          * Motor Z (High Synth1 / Arp from handhule Track 5 / Track 8, C5..G5 where Z has 84-91% clarity!)
    """
    div_a, _tempos_a, trk_a = mc.parse_midi(annejan_path)
    div_h, tempos_h, trk_h = mc.parse_midi(handhule_path)

    tk_start = int(round(start_beat_h * div_h))
    tk_vocal = int(round(96.0 * div_h))
    tk_bass_in = int(round(88.0 * div_h))
    tk_end = int(round(end_beat_h * div_h))

    # 1. Voice A (Lead Vocal from annejan for beat_h >= 96.0)
    voice_A: list[tuple[int, int, int]] = []
    for s_a, e_a, p, _v in trk_a[0]:
        b_h_s = (s_a / float(div_a)) + 29.0
        b_h_e = (e_a / float(div_a)) + 29.0
        if b_h_s < 96.0 or b_h_s >= end_beat_h:
            continue
        s_h = int(round(b_h_s * div_h)) - tk_start
        e_h = min(tk_end, int(round(b_h_e * div_h))) - tk_start
        p_adj = mc.clamp_octave(p, mc.LEAD_MIN_MIDI, mc.LEAD_MAX_MIDI)
        if e_h > s_h:
            voice_A.append((s_h, e_h, p_adj))

    # 2. Voice B (Bass from handhule Track 10 Bass2, starting at beat 88.0 so beats 80..88 are 100% SOLO Z!)
    voice_B: list[tuple[int, int, int]] = []
    for s, e, p, _v in trk_h[10]:
        if s < tk_bass_in or s >= tk_end:
            continue
        s_rel = s - tk_start
        e_rel = min(tk_end, e) - tk_start
        # Shift D#2..G#3 (+12) -> D#3..G#4 clamped into BASS_MIN..BASS_MAX (A#2..F4)
        p_adj = mc.clamp_octave(p + 12, mc.BASS_MIN_MIDI, mc.BASS_MAX_MIDI)
        if e_rel > s_rel:
            voice_B.append((s_rel, e_rel, p_adj))

    # Fill any long bass gaps in beats 192..224 (bars 48..56 where Track 10 drops out) with Track 7 Guitar root
    t7_by_s: dict[int, list[tuple[int, int, int]]] = {}
    for s, e, p, _v in trk_h[7]:
        if tk_vocal <= s < tk_end:
            t7_by_s.setdefault(s, []).append((s - tk_start, min(tk_end, e) - tk_start, p))
    for _s_orig, grp in sorted(t7_by_s.items()):
        grp.sort(key=lambda x: x[2])
        s_rel, e_rel, p_low = grp[0]
        if not any(bs < e_rel and be > s_rel for bs, be, _ in voice_B):
            voice_B.append((s_rel, e_rel, mc.clamp_octave(p_low + 12, mc.BASS_MIN_MIDI, mc.BASS_MAX_MIDI)))

    # 3. Voice Z (Eixo Z: Track 5 Synth1 hook throughout beats 80..224!)
    #    - In beats 80..96 (Intro): Z plays in C#4..F#5 (its warm mid-high lead register)
    #    - In beats 96..224 (Verse+Chorus): Z plays Synth1 in G#4..G5 (where Z has 84-91% fundamental clarity!)
    voice_Z: list[tuple[int, int, int]] = []
    for s, e, p, _v in trk_h[5]:
        if s < tk_start or s >= tk_end:
            continue
        s_rel = s - tk_start
        e_rel = min(tk_end, e) - tk_start
        if s < tk_vocal:
            p_adj = mc.clamp_octave(p - 12, Z_MIN_MIDI, Z_MAX_MIDI)
        else:
            p_adj = mc.clamp_octave(p - 12, Z_HIGH_MIN, Z_HIGH_MAX)
        if e_rel > s_rel:
            voice_Z.append((s_rel, e_rel, p_adj))

    def clean_voice(notes: list[tuple[int, int, int]], min_gap: int = 3) -> list[tuple[int, int, int]]:
        notes = sorted(notes)
        out: list[tuple[int, int, int]] = []
        for i, (s, e, p) in enumerate(notes):
            if i + 1 < len(notes):
                ns, _ne, np_next = notes[i + 1]
                gap = min_gap if np_next == p else 1
                if e > ns - gap:
                    e = max(s + 3, ns - gap)
            if e > s:
                out.append((s, e, p))
        return out

    return (
        clean_voice(voice_A, min_gap=4),
        clean_voice(voice_B, min_gap=3),
        clean_voice(voice_Z, min_gap=3),
        div_h,
        tempos_h,
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
) -> tuple[list[str], dict]:
    """
    Compile 3-voice (Motor A + Motor B + Motor Z) timeline into G1 X Y Z / G4 G-code
    with dynamic safe Z-acceleration limiting and exact 3-axis homing return.
    """
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

    if segments:
        t_end = segments[-1][1]
        # Final 3-voice D# minor power resolve: D#5 (A) + A#3 (B) + F#5 (Z)
        segments.append((t_end, t_end + 0.05, None, None, None, False, False, False))
        segments.append((t_end + 0.05, t_end + 0.55, 75, 58, 78, True, True, True))

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

        # Optimize XY directions (sA, sB)
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
                    cost += 800.0
                if pB is not None and onB and pB == last_pB and sB == last_sB:
                    cost += 800.0
                if pA is not None and not onA and sA != last_sA:
                    cost += 500.0
                if pB is not None and not onB and sB != last_sB:
                    cost += 500.0
                if cost < best_xy_cost:
                    best_xy_cost = cost
                    best_xy = (sA, sB, dx_cmd, dy_cmd)

        # Optimize Z direction (sZ)
        cand_sZ = [1, -1] if pZ is not None else [0]
        best_sZ = 0
        best_z_cost = 1e18
        for sZ in cand_sZ:
            nz = z_phys + (sZ * mag_Z_cmd) * SCALE_Z
            if not (MIN_Z <= nz <= MAX_Z):
                continue
            cost = (nz - CENTER_Z) ** 2
            if pZ is not None and onZ and pZ == last_pZ and sZ == last_sZ:
                cost += 25.0
            if pZ is not None and not onZ and sZ != last_sZ:
                cost += 15.0
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

        # Dynamic safe Z acceleration limit: keeps physical Z accel <= 448 mm/s^2 (< max_z_accel=500)
        if abs(dz_q) > 1e-6:
            raw_safe_accel = min(2500.0, 28.0 * (L_cmd / abs(dz_q)))
        else:
            raw_safe_accel = 2500.0
        for b in (28, 70, 160, 350, 800, 2500):
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

    # Return Z and XY cleanly to exact starting coordinates before restoring normal modes
    ret_dx = round(-cum_dx_cmd, 5)
    ret_dy = round(-cum_dy_cmd, 5)
    ret_dz = round(-cum_dz_cmd, 5)
    if abs(ret_dz) > 1e-5:
        lines.append("SET_VELOCITY_LIMIT ACCEL=28")
        lines.append(f"G1 Z{ret_dz:.5f} F120.00")
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
    midis_dir = Path("/home/shoomba/printer_data/config/scripts/midis")
    annejan = midis_dir / "bad_apple_annejan.mid"
    handhule = midis_dir / "bad_apple_handhule.mid"
    out_cfg = Path("/home/shoomba/printer_data/config/macros_music.cfg")

    # 1. Regenerate MEGALOVANIA first so macros_music.cfg has MEGALOVANIA + BAD_APPLE_Z + BAD_APPLE
    div_m, tempos_m, trk_m = mc.parse_midi(midis_dir / "megalovania_mythic.mid")
    t2s_m = mc.make_tick_to_sec(div_m, tempos_m)
    lead_m, bass_m = mc.arrange_megalovania(trk_m, div_m, max_sec=64.0, t2s=t2s_m)
    segs_m = mc.build_events(lead_m, bass_m, t2s_m, add_ending_chord=True)
    lines_m, _stats_m = mc.compile_corexy_gcode(segs_m)
    mc.write_klipper_macro(
        out_cfg,
        "MEGALOVANIA",
        "Undertale Megalovania 2 vozes CoreXY (4ms_punchy_4x otimizado, 64s tema principal)",
        lines_m,
        append=False,
    )

    # 2. BAD_APPLE_Z (100% Eixo Z solo, 63s: Intro Synth + Verso + Refrao)
    z_notes = arrange_bad_apple_z_solo(annejan, start_beat=51.0, end_beat=195.0)
    z_lines, z_stats = compile_z_solo_gcode(z_notes)
    z_pitches = [p for _, _, p in z_notes]
    print(f"[BAD_APPLE_Z Solo Eixo Z] {z_stats}")
    print(f"  Z pitch range: {mc.midi_to_name(min(z_pitches))}..{mc.midi_to_name(max(z_pitches))} ({len(z_notes)} notas)")

    macro_z = [
        "[gcode_macro BAD_APPLE_Z]",
        "description: Touhou Bad Apple solo no Eixo Z otimizado (2ms_loud_2x, C4-F#5, 63s)",
        "gcode:",
        '  {% if "z" not in printer.toolhead.homed_axes %}',
        "    G28",
        "  {% endif %}",
        "  SAVE_GCODE_STATE NAME=bad_apple_z_state",
        "  G90",
        f"  G1 Z{CENTER_Z:.3f} F1500",
        "  _AUDIO_MODE_Z_ON",
        "  G91",
        *[f"  {ln}" for ln in z_lines],
        "  G90",
        "  _AUDIO_MODE_Z_OFF",
        "  RESTORE_GCODE_STATE NAME=bad_apple_z_state",
        "",
    ]

    # 3. BAD_APPLE (3 Eixos Simultaneos: Motor A + Motor B + Motor Z, 63s)
    vA, vB, vZ, div_h, tempos_h = arrange_bad_apple_3axis(annejan, handhule, start_beat_h=80.0, end_beat_h=224.0)
    xyz_lines, xyz_stats = compile_3axis_gcode(vA, vB, vZ, div_h, tempos_h)
    print(f"[BAD_APPLE 3 Eixos XY+Z] {xyz_stats}")
    print(
        f"  Ranges -> A(Vocal): {mc.midi_to_name(min(p for _,_,p in vA))}..{mc.midi_to_name(max(p for _,_,p in vA))} ({len(vA)}) | "
        f"B(Bass): {mc.midi_to_name(min(p for _,_,p in vB))}..{mc.midi_to_name(max(p for _,_,p in vB))} ({len(vB)}) | "
        f"Z(Synth): {mc.midi_to_name(min(p for _,_,p in vZ))}..{mc.midi_to_name(max(p for _,_,p in vZ))} ({len(vZ)})"
    )

    macro_xyz = [
        "[gcode_macro BAD_APPLE]",
        "description: Touhou Bad Apple 3 vozes simultaneas XY+Z (Motor A Vocal + Motor B Baixo + Motor Z Synth, 63s)",
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

    out_cfg.write_text(out_cfg.read_text() + "\n" + "\n".join(macro_z) + "\n" + "\n".join(macro_xyz))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
