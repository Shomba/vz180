#!/home/shoomba/klippy-env/bin/python
"""
MIDI -> Klipper CoreXY 2-Voice Acoustic Compiler
Otimizado especificamente para a VZ180 AWD CoreXY no perfil acustico vencedor
`4ms_punchy_4x` (mres=6, 1.7A, toff=5, hstrt=7, hend=10, tbl=2, multistep_filt=0, semin=0):
  - Faixa de ouro medida via microfone:
      Motor A (Lead):  A#3 (233 Hz) .. F#5 (740 Hz) — 100% fundamental, +-0.1..3 cents, -8.7..-9.6 dBFS
      Motor B (Bass):  A#2 (116 Hz) .. F#5 (740 Hz) — otimo corpo harmonico + 76-80% clareza em A4/D5
  - Polifonia real de 2 vozes independentes em CoreXY:
      dA_cmd = sA * (fA / pulses_per_cmd_mm) * dt
      dB_cmd = sB * (fB / pulses_per_cmd_mm) * dt
      dX_cmd = (dA_cmd + dB_cmd) / 2
      dY_cmd = (dA_cmd - dB_cmd) / 2
  - Controle inteligente de direcao (sA, sB in {-1, +1}):
      Mantem a cabeca centrada em (90, 87) dentro de [25..155] x [25..150] mm fisicos,
      inverte em notas repetidas para articulacao staccato nitida,
      e faz retorno exato (cum_dx_cmd = 0, cum_dy_cmd = 0) antes de _AUDIO_MODE_OFF
      para manter o homing do Klipper 100% preservado.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Acoustic sweet-spot bounds for 4ms_punchy_4x on VZ180 CoreXY AWD
LEAD_MIN_MIDI = 58  # A#3 (233.08 Hz)
LEAD_MAX_MIDI = 78  # F#5 (739.99 Hz) — above F#5, 4ms_punchy_4x breaks into 3/4 subharmonic
BASS_MIN_MIDI = 46  # A#2 (116.54 Hz) — gritty octave bass (2nd harm = A#3)
BASS_MAX_MIDI = 65  # F4  (349.23 Hz)

# Safe physical bed box (bed is 180 x 174 mm)
CENTER_X = 90.0
CENTER_Y = 87.0
MIN_X, MAX_X = 25.0, 155.0
MIN_Y, MAX_Y = 25.0, 150.0


def midi_to_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def midi_to_name(note: int) -> str:
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    val = 0
    while True:
        b = data[pos]
        pos += 1
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            return val, pos


def parse_midi(path: Path) -> tuple[int, list[tuple[int, int]], list[list[tuple[int, int, int, int]]]]:
    """Return (division, tempos[(tick, us_per_beat)], tracks[[(start_tick, end_tick, pitch, vel)]])."""
    data = path.read_bytes()
    if data[:4] != b"MThd":
        raise ValueError(f"Arquivo nao e MIDI valido: {path}")
    hdr_len, _fmt, ntrks, division = struct.unpack(">IHHH", data[4:14])
    pos = 8 + hdr_len
    tempos = [(0, 500000)]
    tracks: list[list[tuple[int, int, int, int]]] = []

    for _ in range(ntrks):
        if data[pos : pos + 4] != b"MTrk":
            raise ValueError("Cabecalho MTrk invalido")
        trk_len = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        end = pos + 8 + trk_len
        p = pos + 8
        tick = 0
        running = None
        active: dict[tuple[int, int], tuple[int, int]] = {}
        notes: list[tuple[int, int, int, int]] = []

        while p < end:
            delta, p = _read_vlq(data, p)
            tick += delta
            b = data[p]
            if b == 0xFF:
                mtype = data[p + 1]
                mlen, p = _read_vlq(data, p + 2)
                mdata = data[p : p + mlen]
                p += mlen
                if mtype == 0x51 and mlen == 3:
                    us = (mdata[0] << 16) | (mdata[1] << 8) | mdata[2]
                    tempos.append((tick, us))
            elif b in (0xF0, 0xF7):
                mlen, p = _read_vlq(data, p + 1)
                p += mlen
            else:
                if b & 0x80:
                    status = b
                    p += 1
                    running = status
                else:
                    status = running
                etype = status & 0xF0
                ch = status & 0x0F
                if etype in (0xC0, 0xD0):
                    p += 1
                else:
                    d1, d2 = data[p], data[p + 1]
                    p += 2
                    if ch == 9:
                        continue  # skip percussion channel
                    if etype == 0x90 and d2 > 0:
                        if (ch, d1) in active:
                            st, v = active.pop((ch, d1))
                            if tick > st:
                                notes.append((st, tick, d1, v))
                        active[(ch, d1)] = (tick, d2)
                    elif etype == 0x80 or (etype == 0x90 and d2 == 0):
                        if (ch, d1) in active:
                            st, v = active.pop((ch, d1))
                            if tick > st:
                                notes.append((st, tick, d1, v))
        notes.sort()
        tracks.append(notes)
        pos = end

    tempos.sort()
    return division, tempos, tracks


def make_tick_to_sec(division: int, tempos: list[tuple[int, int]], speed_mult: float = 1.0):
    def t2s(tk: int) -> float:
        sec = 0.0
        prev_tk = 0
        cur_us = 500000
        for tt, us in tempos:
            if tt >= tk:
                break
            if tt > prev_tk:
                sec += (tt - prev_tk) * cur_us / 1e6 / division
                prev_tk = tt
            cur_us = us
        sec += (tk - prev_tk) * cur_us / 1e6 / division
        return sec / speed_mult
    return t2s


def clamp_octave(pitch: int, lo: int, hi: int) -> int:
    while pitch > hi:
        pitch -= 12
    while pitch < lo:
        pitch += 12
    return pitch


def arrange_megalovania(
    tracks: list[list[tuple[int, int, int, int]]],
    division: int,
    max_sec: float | None = None,
    t2s=None,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """
    Phrase-aware octave arrangement of megalovania_mythic.mid into the exact
    measured sweet-spot of VZ180 4ms_punchy_4x:
      - Motor A (Lead):  A#3..F#5 (MIDI 58..78) — preserves melodic contour inside every 4-bar phrase
      - Motor B (Bass):  A#2..F4  (MIDI 46..65) — preserves pumping bass bounce + steals Lead dyads on rests
    Returns (lead_notes, bass_notes) each as [(start_tick, end_tick, midi_pitch)].
    """
    bar_ticks = division * 4  # 4/4 time
    raw_lead = tracks[0]
    raw_bass = tracks[1] if len(tracks) > 1 else []

    # Group simultaneous onsets on Track 0 to separate melody vs chord/octave doubles
    by_start: dict[int, list[tuple[int, int, int]]] = {}
    for s, e, p, _v in raw_lead:
        if max_sec is not None and t2s(s) >= max_sec:
            continue
        if max_sec is not None and t2s(e) > max_sec:
            # trim end cleanly
            pass
        by_start.setdefault(s, []).append((s, e, p))

    # Build Lead (monophonic melody) + Secondary Harmony (for rests in Bass)
    # Keep full chord candidates at each onset so we can pick the highest in-range tone after phrase shift
    lead_chords: list[tuple[int, list[tuple[int, int, int]]]] = []
    extra_harmony: list[tuple[int, int, int]] = []

    for s, group in sorted(by_start.items()):
        group.sort(key=lambda x: x[2])  # ascending pitch
        lead_chords.append((s, group))

    # Phrase-level octave shift per 2-bar sub-phrase so soaring melodies stay intact
    # without wrapping individual high peaks awkwardly mid-phrase
    lead_out: list[tuple[int, int, int]] = []
    sub_bars = 2
    max_bar = max((s // bar_ticks for s, _ in lead_chords), default=0) + 1
    phrase_shifts: dict[int, int] = {}
    for ph in range(0, max_bar, sub_bars):
        # Representative melody note per onset: lower if pure octave double inside range, else upper
        pnotes = []
        for s, grp in lead_chords:
            if not (ph <= (s // bar_ticks) < ph + sub_bars):
                continue
            if len(grp) > 1 and (grp[-1][2] - grp[0][2]) == 12 and LEAD_MIN_MIDI <= grp[0][2] <= LEAD_MAX_MIDI:
                pnotes.append(grp[0][2])
            else:
                pnotes.append(grp[-1][2])
        if not pnotes:
            phrase_shifts[ph] = 0
            continue
        best_shift = 0
        best_score = -1e9
        for sh in (-24, -12, 0, 12, 24):
            shifted = [p + sh for p in pnotes]
            in_range = sum(1 for p in shifted if LEAD_MIN_MIDI <= p <= LEAD_MAX_MIDI)
            center_pen = sum(abs(p - 68) for p in shifted) / len(shifted) * 0.05
            sc = in_range - center_pen - abs(sh) * 0.002
            if sc > best_score:
                best_score = sc
                best_shift = sh
        phrase_shifts[ph] = best_shift

    for s, grp in lead_chords:
        ph = ((s // bar_ticks) // sub_bars) * sub_bars
        sh = phrase_shifts.get(ph, 0)
        if len(grp) == 1:
            s0, e0, p0 = grp[0]
            lead_out.append((s0, e0, clamp_octave(p0 + sh, LEAD_MIN_MIDI, LEAD_MAX_MIDI)))
        else:
            lo_n, hi_n = grp[0], grp[-1]
            if (hi_n[2] - lo_n[2]) == 12 and LEAD_MIN_MIDI <= (lo_n[2] + sh) <= LEAD_MAX_MIDI:
                lead_out.append((lo_n[0], lo_n[1], lo_n[2] + sh))
            else:
                # Pick highest chord tone that fits inside [LEAD_MIN, LEAD_MAX] without octave wrap
                in_rng = [n for n in grp if LEAD_MIN_MIDI <= (n[2] + sh) <= LEAD_MAX_MIDI]
                chosen = in_rng[-1] if in_rng else hi_n
                lead_out.append((chosen[0], chosen[1], clamp_octave(chosen[2] + sh, LEAD_MIN_MIDI, LEAD_MAX_MIDI)))
                other = lo_n if chosen is hi_n else hi_n
                extra_harmony.append((other[0], other[1], other[2] + sh))

    # Arrange Bass (Track 1) cleanly per 4-bar phrase
    bass_by_start: dict[int, list[tuple[int, int, int]]] = {}
    for s, e, p, _v in raw_bass:
        if max_sec is not None and t2s(s) >= max_sec:
            continue
        bass_by_start.setdefault(s, []).append((s, e, p))

    bass_mono: list[tuple[int, int, int]] = []
    for s, group in sorted(bass_by_start.items()):
        group.sort(key=lambda x: x[2])
        bass_mono.append(group[-1])  # take cleaner upper bass note if doubled

    bass_phrase_shifts: dict[int, int] = {}
    for ph in range(0, max_bar, 4):
        bnotes = [n[2] for n in bass_mono if ph <= (n[0] // bar_ticks) < ph + 4]
        if not bnotes:
            bass_phrase_shifts[ph] = 0
            continue
        # Target crisp bass zone: A#2..D#4 (46..63), preferring A#3..D#4 (58..63) when single-octave
        best_shift = 12
        best_score = -1e9
        for sh in (0, 12, 24):
            shifted = [p + sh for p in bnotes]
            in_range = sum(1 for p in shifted if BASS_MIN_MIDI <= p <= BASS_MAX_MIDI)
            # Prefer high purity (>= C3=48, ideally A#3=58..D4=62) while staying below Lead
            clarity_bonus = sum(1.0 if p >= 58 else (0.5 if p >= 48 else -1.0) for p in shifted) / len(shifted)
            sc = in_range + clarity_bonus * 5.0
            if sc > best_score:
                best_score = sc
                best_shift = sh
        bass_phrase_shifts[ph] = best_shift

    bass_out: list[tuple[int, int, int]] = []
    for s, e, p in bass_mono:
        ph = ((s // bar_ticks) // 4) * 4
        p_adj = clamp_octave(p + bass_phrase_shifts.get(ph, 12), BASS_MIN_MIDI, BASS_MAX_MIDI)
        bass_out.append((s, e, p_adj))

    # Fill any gaps in Bass using extra_harmony (when Track 1 is silent during a Lead dyad)
    if extra_harmony:
        for s, e, p in extra_harmony:
            # Check if bass_out overlaps [s, e)
            overlaps = any(bs < e and be > s for bs, be, _ in bass_out)
            if not overlaps:
                p_adj = clamp_octave(p, LEAD_MIN_MIDI, LEAD_MAX_MIDI)
                bass_out.append((s, e, p_adj))
        bass_out.sort()

    # Ensure strictly non-overlapping notes within each monophonic voice
    def clean_mono(notes: list[tuple[int, int, int]], min_gap_ticks: int = 10) -> list[tuple[int, int, int]]:
        notes = sorted(notes)
        cleaned: list[tuple[int, int, int]] = []
        for i, (s, e, p) in enumerate(notes):
            if i + 1 < len(notes):
                next_s, _ne, next_p = notes[i + 1]
                # Leave crisp articulation gap on repeated pitch or overlap
                gap = min_gap_ticks if (next_p == p) else 4
                if e > next_s - gap:
                    e = max(s + 15, next_s - gap)
            if e > s:
                cleaned.append((s, e, p))
        return cleaned

    return clean_mono(lead_out, min_gap_ticks=14), clean_mono(bass_out, min_gap_ticks=14)


def build_events(
    lead_notes: list[tuple[int, int, int]],
    bass_notes: list[tuple[int, int, int]],
    t2s,
    add_ending_chord: bool = False,
) -> list[tuple[float, float, int | None, int | None, bool, bool]]:
    """
    Slice timeline into contiguous segments:
      (t_start, t_end, pitch_A, pitch_B, onset_A, onset_B)
    """
    ticks = set()
    for s, e, _ in lead_notes:
        ticks.add(s)
        ticks.add(e)
    for s, e, _ in bass_notes:
        ticks.add(s)
        ticks.add(e)
    sorted_ticks = sorted(ticks)

    lead_starts = {s for s, _, _ in lead_notes}
    bass_starts = {s for s, _, _ in bass_notes}

    # Build quick lookup by active interval
    def active_at(notes, t_mid):
        for s, e, p in notes:
            if s <= t_mid < e:
                return p
            if s > t_mid:
                break
        return None

    # Optimize lookup with pointer sweep
    segments: list[tuple[float, float, int | None, int | None, bool, bool]] = []
    li = 0
    bi = 0
    for i in range(len(sorted_ticks) - 1):
        tk0, tk1 = sorted_ticks[i], sorted_ticks[i + 1]
        dt = t2s(tk1) - t2s(tk0)
        if dt < 0.004:
            continue
        t_mid = 0.5 * (tk0 + tk1)
        while li < len(lead_notes) and lead_notes[li][1] <= tk0:
            li += 1
        while bi < len(bass_notes) and bass_notes[bi][1] <= tk0:
            bi += 1
        pA = lead_notes[li][2] if (li < len(lead_notes) and lead_notes[li][0] <= t_mid < lead_notes[li][1]) else None
        pB = bass_notes[bi][2] if (bi < len(bass_notes) and bass_notes[bi][0] <= t_mid < bass_notes[bi][1]) else None
        onsetA = tk0 in lead_starts and pA is not None
        onsetB = tk0 in bass_starts and pB is not None
        segments.append((t2s(tk0), t2s(tk1), pA, pB, onsetA, onsetB))

    if add_ending_chord and segments:
        t_end = segments[-1][1]
        # Crisp final D minor power resolve (D5 + D4)
        segments.append((t_end, t_end + 0.06, None, None, False, False))
        segments.append((t_end + 0.06, t_end + 0.52, 74, 62, True, True))  # D5 + D4

    return segments


def compile_corexy_gcode(
    segments: list[tuple[float, float, int | None, int | None, bool, bool]],
    orig_ms: int = 16,
    rot_dist: float = 40.0,
    target_ms: int = 4,
) -> tuple[list[str], dict]:
    """
    Compile 2-voice segments into relative G1/G4 G-code for _AUDIO_MODE_4MS_ON.
    Tracks exact physical & commanded coordinates so:
      - Zero bed-boundary violations
      - Zero cumulative homing drift (exact return to start before _AUDIO_MODE_OFF)
    """
    scale = orig_ms / float(target_ms)  # 16 / 4 = 4.0
    pulses_per_cmd_mm = (200.0 * orig_ms) / rot_dist  # 80.0 pulses/cmd_mm

    x_phys = CENTER_X
    y_phys = CENTER_Y
    cum_dx_cmd = 0.0
    cum_dy_cmd = 0.0
    last_sA = 1
    last_sB = 1
    last_pA: int | None = None
    last_pB: int | None = None

    min_x, max_x = x_phys, x_phys
    min_y, max_y = y_phys, y_phys
    max_v_phys = 0.0
    gcode_lines: list[str] = []
    n_moves = 0
    n_rests = 0

    for t0, t1, pA, pB, onsetA, onsetB in segments:
        dt = t1 - t0
        if dt < 0.005:
            continue

        if pA is None and pB is None:
            ms = int(round(dt * 1000.0))
            if ms >= 5:
                gcode_lines.append(f"G4 P{ms}")
                n_rests += 1
            continue

        fA = midi_to_hz(pA) if pA is not None else 0.0
        fB = midi_to_hz(pB) if pB is not None else 0.0

        # Split only if a single sustained note would ever exceed 55 mm belt travel
        max_belt_phys = max(fA, fB) * (rot_dist / (200.0 * target_ms)) * dt
        n_sub = max(1, int(math.ceil(max_belt_phys / 55.0)))
        sub_dt = dt / n_sub

        for sub_i in range(n_sub):
            mag_A_cmd = (fA / pulses_per_cmd_mm) * sub_dt
            mag_B_cmd = (fB / pulses_per_cmd_mm) * sub_dt

            cand_sA = [1, -1] if pA is not None else [0]
            cand_sB = [1, -1] if pB is not None else [0]

            best_choice = None
            best_cost = 1e18
            for sA in cand_sA:
                for sB in cand_sB:
                    dA_cmd = sA * mag_A_cmd
                    dB_cmd = sB * mag_B_cmd
                    dx_cmd = 0.5 * (dA_cmd + dB_cmd)
                    dy_cmd = 0.5 * (dA_cmd - dB_cmd)
                    nx = x_phys + dx_cmd * scale
                    ny = y_phys + dy_cmd * scale
                    # Hard bed limit check
                    if not (MIN_X <= nx <= MAX_X and MIN_Y <= ny <= MAX_Y):
                        continue
                    # Cost: stay close to center
                    dist_center = (nx - CENTER_X) ** 2 + (ny - CENTER_Y) ** 2
                    cost = dist_center
                    # On repeated pitch onset (e.g. D4 -> D4), prefer flipping direction for punchy articulation
                    if sub_i == 0:
                        if pA is not None and onsetA and pA == last_pA and sA == last_sA:
                            cost += 900.0
                        if pB is not None and onsetB and pB == last_pB and sB == last_sB:
                            cost += 900.0
                        # On legato/non-onset continuation, prefer keeping direction to avoid click
                        if pA is not None and not onsetA and sA != last_sA:
                            cost += 600.0
                        if pB is not None and not onsetB and sB != last_sB:
                            cost += 600.0
                    if cost < best_cost:
                        best_cost = cost
                        best_choice = (sA, sB, dx_cmd, dy_cmd, nx, ny)

            if best_choice is None:
                raise RuntimeError(
                    f"Movimento excedeu caixa segura em t={t0:.2f}s pos=({x_phys:.1f},{y_phys:.1f})"
                )

            sA, sB, dx_cmd, dy_cmd, _nx, _ny = best_choice
            # Quantize to 5 decimals BEFORE updating trackers so cumulative return is exact to 0.00001 mm
            dx_q = round(dx_cmd, 5)
            dy_q = round(dy_cmd, 5)
            L_cmd = math.hypot(dx_q, dy_q)
            if L_cmd < 1e-6:
                continue
            v_cmd = L_cmd / sub_dt
            f_cmd = round(v_cmd * 60.0, 2)
            v_phys = v_cmd * scale
            max_v_phys = max(max_v_phys, v_phys)

            gcode_lines.append(f"G1 X{dx_q:.5f} Y{dy_q:.5f} F{f_cmd:.2f}")
            n_moves += 1

            cum_dx_cmd = round(cum_dx_cmd + dx_q, 5)
            cum_dy_cmd = round(cum_dy_cmd + dy_q, 5)
            x_phys = CENTER_X + cum_dx_cmd * scale
            y_phys = CENTER_Y + cum_dy_cmd * scale
            min_x, max_x = min(min_x, x_phys), max(max_x, x_phys)
            min_y, max_y = min(min_y, y_phys), max(max_y, y_phys)

            if pA is not None:
                last_sA = sA
                last_pA = pA
            if pB is not None:
                last_sB = sB
                last_pB = pB

    # Exact return to starting coordinates BEFORE _AUDIO_MODE_OFF so Klipper logical == physical!
    ret_dx = round(-cum_dx_cmd, 5)
    ret_dy = round(-cum_dy_cmd, 5)
    if abs(ret_dx) > 1e-5 or abs(ret_dy) > 1e-5:
        # Move back swiftly (~80 mm/s physical = 20 mm/s cmd = F1200)
        gcode_lines.append(f"G1 X{ret_dx:.5f} Y{ret_dy:.5f} F1200.00")

    stats = {
        "moves": n_moves,
        "rests": n_rests,
        "duration_s": round(segments[-1][1] - segments[0][0], 1) if segments else 0.0,
        "x_phys_range": (round(min_x, 1), round(max_x, 1)),
        "y_phys_range": (round(min_y, 1), round(max_y, 1)),
        "max_v_phys_mm_s": round(max_v_phys, 1),
        "return_move_cmd": (ret_dx, ret_dy),
    }
    return gcode_lines, stats


def write_klipper_macro(
    out_cfg: Path,
    macro_name: str,
    description: str,
    body_lines: list[str],
    append: bool = False,
) -> None:
    lines = [
        f"[gcode_macro {macro_name}]",
        f"description: {description}",
        "gcode:",
        '  {% if "xy" not in printer.toolhead.homed_axes %}',
        "    G28 X Y",
        "  {% endif %}",
        f"  SAVE_GCODE_STATE NAME={macro_name.lower()}_state",
        "  G90",
        f"  G1 X{CENTER_X:.3f} Y{CENTER_Y:.3f} F9000",
        "  _AUDIO_MODE_COMPACTO_ON",
        "  G91",
    ]
    for ln in body_lines:
        lines.append(f"  {ln}")
    lines.extend([
        "  G90",
        "  _AUDIO_MODE_OFF",
        f"  RESTORE_GCODE_STATE NAME={macro_name.lower()}_state",
        "",
    ])
    text = "\n".join(lines)
    if append and out_cfg.exists():
        out_cfg.write_text(out_cfg.read_text() + "\n" + text)
    else:
        out_cfg.write_text(text)


def write_standalone_gcode(out_gcode: Path, header_comment: str, body_lines: list[str]) -> None:
    lines = [
        f"; {header_comment}",
        "; Perfil acustico: 4ms_punchy_4x (VZ180 AWD CoreXY)",
        "G90",
        "G28 X Y",
        f"G1 X{CENTER_X:.3f} Y{CENTER_Y:.3f} F9000",
        "_AUDIO_MODE_COMPACTO_ON",
        "G91",
        *body_lines,
        "G90",
        "_AUDIO_MODE_OFF",
        "",
    ]
    out_gcode.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="Compila MIDI para CoreXY 2 vozes (4ms_punchy_4x)")
    ap.add_argument("midi", type=Path, help="Arquivo .mid de entrada")
    ap.add_argument("--max-sec", type=float, default=64.0, help="Duracao maxima para a macro principal")
    ap.add_argument("--out-cfg", type=Path, default=Path("/home/shoomba/printer_data/config/macros_music.cfg"))
    ap.add_argument("--out-gcode", type=Path, default=Path("/home/shoomba/printer_data/gcodes/MEGALOVANIA_FULL.gcode"))
    args = ap.parse_args()

    division, tempos, tracks = parse_midi(args.midi)
    t2s = make_tick_to_sec(division, tempos)

    # 1. Iconic 64s arrangement for Klipper macro (Intro -> Bass drop -> Chorus -> B -> C -> Resolve)
    lead_64, bass_64 = arrange_megalovania(tracks, division, max_sec=args.max_sec, t2s=t2s)
    segs_64 = build_events(lead_64, bass_64, t2s, add_ending_chord=True)
    lines_64, stats_64 = compile_corexy_gcode(segs_64)

    pA_64 = [p for _, _, p in lead_64]
    pB_64 = [p for _, _, p in bass_64]
    print(f"[MEGALOVANIA 64s Macro] {stats_64}")
    print(f"  Lead range: {midi_to_name(min(pA_64))}..{midi_to_name(max(pA_64))} ({len(lead_64)} notas)")
    print(f"  Bass range: {midi_to_name(min(pB_64))}..{midi_to_name(max(pB_64))} ({len(bass_64)} notas)")

    write_klipper_macro(
        args.out_cfg,
        "MEGALOVANIA",
        "Undertale Megalovania 2 vozes CoreXY (4ms_punchy_4x otimizado, 64s tema principal)",
        lines_64,
        append=False,
    )

    # 2. Full 164s song as standalone G-code file in virtual_sdcard
    lead_full, bass_full = arrange_megalovania(tracks, division, max_sec=None, t2s=t2s)
    segs_full = build_events(lead_full, bass_full, t2s, add_ending_chord=True)
    lines_full, stats_full = compile_corexy_gcode(segs_full)
    print(f"[MEGALOVANIA FULL .gcode] {stats_full}")
    write_standalone_gcode(
        args.out_gcode,
        "Undertale - Megalovania (Completa 2 vozes CoreXY, adaptada de megalovania_mythic.mid)",
        lines_full,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
