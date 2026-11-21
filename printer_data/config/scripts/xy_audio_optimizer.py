#!/home/shoomba/klippy-env/bin/python
"""
Otimizador Acustico Automatico para Motores CoreXY (TMC2209 stepper_x/x1/y/y1)
usando o microfone da webcam USB.
Testa combinacoes de:
  - Microsteps (16, 8, 4, 2, 1/fullstep)
  - Chopper (toff, hstrt, hend, tbl, multistep_filt) + CoolStep OFF (semin=0)
  - Corrente (1.4A / 1.7A)
  - Modo harmonico (1x, 2x, 4x)
Ao final, roda a varredura de oitavas (C3-G6) nos motores A e B no perfil campeao.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/shoomba/printer_data/config/scripts")
import stepper_tuner as st
import z_audio_optimizer as zao

OUT_JSON = Path("/home/shoomba/printer_data/config/xy_audio_optimization.json")
MRES_MAP = {256: 0, 128: 1, 64: 2, 32: 3, 16: 4, 8: 5, 4: 6, 2: 7, 1: 8}
XY_STEPPERS = ("stepper_x", "stepper_x1", "stepper_y", "stepper_y1")


def apply_xy_config(
    orig_ms_xy: int,
    target_ms: int,
    current: float,
    toff: int,
    hstrt: int,
    hend: int,
    tbl: int,
    mfilt: int,
    intpol: int,
    semin: int,
) -> float:
    scale_xy = orig_ms_xy / float(target_ms)
    scaled_accel = max(50.0, round(12000.0 / scale_xy, 1))
    scaled_scv = max(0.2, round(5.0 / scale_xy, 3))
    mres = MRES_MAP[target_ms]
    cmds = [
        "M400",
        f"SET_VELOCITY_LIMIT ACCEL={scaled_accel} SQUARE_CORNER_VELOCITY={scaled_scv}",
    ]
    for stp in XY_STEPPERS:
        cmds.extend([
            f"SET_TMC_CURRENT STEPPER={stp} CURRENT={current:.2f}",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=en_spreadcycle VALUE=1",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=semin VALUE={semin}",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=multistep_filt VALUE={mfilt}",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=toff VALUE={toff}",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=hstrt VALUE={hstrt}",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=hend VALUE={hend}",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=tbl VALUE={tbl}",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=intpol VALUE={intpol}",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=mres VALUE={mres}",
        ])
    st.run_gcode("\n".join(cmds))
    return scale_xy


def restore_xy_config(cfg: dict) -> None:
    orig_ms_xy = int(cfg["stepper_x"]["microsteps"])
    orig_mres = MRES_MAP.get(orig_ms_xy, 4)
    cmds = ["G90", "M400"]
    for stp in XY_STEPPERS:
        tmc = cfg[f"tmc2209 {stp}"]
        orig_intpol = 1 if tmc["interpolate"] else 0
        cmds.extend([
            f"SET_TMC_FIELD STEPPER={stp} FIELD=mres VALUE={orig_mres}",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=intpol VALUE={orig_intpol}",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=toff VALUE=1",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=hstrt VALUE=0",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=hend VALUE=2",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=tbl VALUE=1",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=multistep_filt VALUE=1",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=semin VALUE=2",
            f"SET_TMC_FIELD STEPPER={stp} FIELD=en_spreadcycle VALUE=1",
            f"SET_TMC_CURRENT STEPPER={stp} CURRENT={tmc['run_current']}",
        ])
    cmds.append(
        f"SET_VELOCITY_LIMIT ACCEL={cfg['printer']['max_accel']} "
        f"SQUARE_CORNER_VELOCITY={cfg['printer']['square_corner_velocity']}"
    )
    st.run_gcode("\n".join(cmds))


def play_xy_note(
    motor: str,
    source: str,
    noise_freqs: np.ndarray,
    noise_spec: np.ndarray,
    target_hz: float,
    drive_div: float,
    steps_per_mm_xy: float,
    scale_xy: float,
    dur: float = 0.34,
) -> dict:
    """Spin ONLY Motor A (+45 deg) or Motor B (-45 deg) at speed for target_hz / drive_div."""
    cmd_hz = target_hz / drive_div
    gcode, rec_dur = st.build_corexy_isolated_move(
        motor, cmd_hz, dur, steps_per_mm_xy, scale_xy, direction=1
    )
    v_phys = (cmd_hz / steps_per_mm_xy) / math.sqrt(2.0)
    rec_holder: list[np.ndarray] = []
    t_rec = threading.Thread(target=lambda: rec_holder.append(st.record_audio(source, rec_dur)), daemon=True)
    t_rec.start()
    time.sleep(0.05)
    st.run_gcode(gcode)
    t_rec.join(timeout=rec_dur + 3.0)
    res = zao.measure_clarity(rec_holder[0], noise_freqs, noise_spec, target_hz)
    res["v_phys_mm_s"] = round(v_phys, 1)
    return res


def main() -> int:
    source = st.find_camera_mic()
    info = st.moonraker_get("/printer/info")["result"]
    if info["state"] != "ready":
        print(f"ERRO: Klipper state={info['state']}")
        return 1

    cfg = st.get_stepper_settings()
    orig_ms_xy = int(cfg["stepper_x"]["microsteps"])
    rot_xy = float(cfg["stepper_x"]["rotation_distance"])
    steps_per_mm_xy = 200.0 / rot_xy

    print(f"[Mic] {source}")
    noise_samples = st.record_audio(source, 0.8)
    noise_freqs, noise_spec = st.compute_spectrum(noise_samples, f_min=70.0, f_max=5000.0)
    noise_rms = 20.0 * math.log10(float(np.sqrt(np.mean(noise_samples ** 2))) + 1e-9)
    print(f"[Ruido Ambiente] {noise_rms:.1f} dBFS | XY: {orig_ms_xy}ms, {steps_per_mm_xy:.2f} steps/mm")

    homed = st.moonraker_get("/printer/objects/query?toolhead")["result"]["status"]["toolhead"]["homed_axes"]
    if "x" not in homed or "y" not in homed:
        print("[Klipper] Fazendo G28 X Y...")
        st.run_gcode("G28 X Y", timeout=60.0)

    st.run_gcode("G90\nG1 X90 Y87 F9000\nM400")

    # Format: (name, ms, current, toff, hstrt, hend, tbl, mfilt, intpol, semin, drive_div)
    candidates = [
        # Stock baseline
        ("stock_16ms_1x",         16, 1.40, 1, 0, 2, 1, 1, 0, 2, 1.0),
        # 8 microsteps
        ("8ms_loud_1x",            8, 1.70, 4, 7, 6, 2, 0, 0, 0, 1.0),
        # 4 microsteps (stock chopper vs crisp vs loud vs punchy, 1x vs 4x)
        ("4ms_stock_1x",           4, 1.70, 1, 0, 2, 1, 1, 0, 0, 1.0),
        ("4ms_crisp_1x",           4, 1.70, 3, 5, 4, 2, 0, 0, 0, 1.0),
        ("4ms_loud_1x",            4, 1.70, 4, 7, 6, 2, 0, 0, 0, 1.0),
        ("4ms_punchy_1x",          4, 1.70, 5, 7, 10, 2, 0, 0, 0, 1.0),
        ("4ms_loud_4x",            4, 1.70, 4, 7, 6, 2, 0, 0, 0, 4.0),
        ("4ms_punchy_4x",          4, 1.70, 5, 7, 10, 2, 0, 0, 0, 4.0),
        # 2 microsteps (stock chopper vs crisp vs loud vs punchy, 1x vs 2x)
        ("2ms_stock_1x",           2, 1.70, 1, 0, 2, 1, 1, 0, 0, 1.0),
        ("2ms_crisp_1x",           2, 1.70, 3, 5, 4, 2, 0, 0, 0, 1.0),
        ("2ms_loud_1x",            2, 1.70, 4, 7, 6, 2, 0, 0, 0, 1.0),
        ("2ms_punchy_1x",          2, 1.70, 5, 7, 10, 2, 0, 0, 0, 1.0),
        ("2ms_loud_2x",            2, 1.70, 4, 7, 6, 2, 0, 0, 0, 2.0),
        ("2ms_punchy_2x",          2, 1.70, 5, 7, 10, 2, 0, 0, 0, 2.0),
        # 1 microstep (FULL STEP on TMC2209!) at 1.4A and 1.7A with safe tbl=2, toff=4..5
        ("1ms_1.4A_loud_1x",       1, 1.40, 4, 7, 6, 2, 0, 0, 0, 1.0),
        ("1ms_1.7A_loud_1x",       1, 1.70, 4, 7, 6, 2, 0, 0, 0, 1.0),
        ("1ms_1.7A_punchy_1x",     1, 1.70, 5, 7, 10, 2, 0, 0, 0, 1.0),
        ("1ms_1.7A_crisp_1x",      1, 1.70, 3, 5, 4, 2, 0, 0, 0, 1.0),
    ]

    probe_notes = [("C4", 261.63), ("G4", 392.00), ("C5", 523.25)]
    bench_results = []

    print(f"\n{'Perfil':<20} | {'RMS(dB)':>7} | {'PeakHz':>7} | {'Cents':>7} | {'Fund%':>6} | {'Pure%':>6} | {'Score':>6} | Notas (Motor A)")
    print("-" * 102)

    try:
        for (name, ms, cur, toff, hstrt, hend, tbl, mfilt, intpol, semin, div) in candidates:
            st_now = st.moonraker_get("/printer/info")["result"]["state"]
            if st_now != "ready":
                print(f"ABORT: Klipper saiu de ready ({st_now}) antes de {name}")
                break
            try:
                scale_xy = apply_xy_config(orig_ms_xy, ms, cur, toff, hstrt, hend, tbl, mfilt, intpol, semin)
                time.sleep(0.06)
                note_res = []
                for nlabel, nhz in probe_notes:
                    r = play_xy_note("A", source, noise_freqs, noise_spec, nhz, div, steps_per_mm_xy, scale_xy, dur=0.32)
                    r["note"] = nlabel
                    note_res.append(r)
                    time.sleep(0.08)

                avg_rms = float(np.mean([r["rms_db"] for r in note_res]))
                avg_fund = float(np.mean([r["fund_clarity_pct"] for r in note_res]))
                avg_pure = float(np.mean([r["tonal_purity_pct"] for r in note_res]))
                avg_score = float(np.mean([r["score"] for r in note_res]))
                g4 = note_res[1]
                note_summary = " ".join(f"{r['note']}->{r['peak_note']}({r['rms_db']:.0f}dB)" for r in note_res)

                entry = {
                    "profile": name,
                    "params": {
                        "microsteps": ms, "mres": MRES_MAP[ms], "current": cur,
                        "toff": toff, "hstrt": hstrt, "hend": hend, "tbl": tbl,
                        "multistep_filt": mfilt, "intpol": intpol, "semin": semin,
                        "drive_div": div,
                    },
                    "avg_rms_db": round(avg_rms, 1),
                    "avg_fund_clarity_pct": round(avg_fund, 1),
                    "avg_tonal_purity_pct": round(avg_pure, 1),
                    "avg_score": round(avg_score, 1),
                    "notes": note_res,
                }
                bench_results.append(entry)
                print(
                    f"{name:<20} | {avg_rms:7.1f} | {g4['peak_hz']:7.1f} | {g4['cents']:+7.1f} | "
                    f"{avg_fund:5.1f}% | {avg_pure:5.1f}% | {avg_score:6.1f} | {note_summary}"
                )
            except Exception as exc:
                print(f"{name:<20} | ERRO: {exc}")
                break

        # Phase 2: Octave sweep on BOTH Motor A and Motor B using the top winner!
        winner_sweep = {"motor_A": [], "motor_B": []}
        if bench_results and st.moonraker_get("/printer/info")["result"]["state"] == "ready":
            winner = max(bench_results, key=lambda x: x["avg_score"])
            wp = winner["params"]
            print(f"\n=== VENCEDOR XY: {winner['profile']} (Score={winner['avg_score']}, RMS={winner['avg_rms_db']} dBFS, Clareza={winner['avg_fund_clarity_pct']}%) ===")
            print(f"Params: {wp}")
            scale_xy = apply_xy_config(
                orig_ms_xy, wp["microsteps"], wp["current"], wp["toff"], wp["hstrt"],
                wp["hend"], wp["tbl"], wp["multistep_filt"], wp["intpol"], wp["semin"]
            )
            sweep_notes = [
                ("C3", 130.81), ("G3", 196.00),
                ("C4", 261.63), ("E4", 329.63), ("G4", 392.00), ("A4", 440.00),
                ("C5", 523.25), ("E5", 659.25), ("G5", 783.99), ("A5", 880.00),
                ("C6", 1046.50), ("G6", 1567.98),
            ]
            for m in ("A", "B"):
                print(f"\n=== Varredura de Oitavas — Motor {m} ({winner['profile']}) ===")
                print("Nota | Vel(mm/s) | Alvo(Hz) | Medido(Hz) | NotaReal | Cents   | Fund% | Pure% | RMS(dB)")
                print("-" * 92)
                for nlabel, nhz in sweep_notes:
                    r = play_xy_note(m, source, noise_freqs, noise_spec, nhz, wp["drive_div"], steps_per_mm_xy, scale_xy, dur=0.30)
                    r["note"] = nlabel
                    winner_sweep[f"motor_{m}"].append(r)
                    print(
                        f"{nlabel:<4} | {r['v_phys_mm_s']:9.1f} | {nhz:8.2f} | {r['peak_hz']:10.2f} | "
                        f"{r['peak_note']:<8} | {r['cents']:+7.1f} | {r['fund_clarity_pct']:4.1f}% | "
                        f"{r['tonal_purity_pct']:4.1f}% | {r['rms_db']:7.1f}"
                    )
                    time.sleep(0.08)

        OUT_JSON.write_text(
            json.dumps(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "noise_floor_rms_db": round(noise_rms, 1),
                    "candidates": sorted(bench_results, key=lambda x: x["avg_score"], reverse=True),
                    "winner_octave_sweep": winner_sweep,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nResultados salvos em: {OUT_JSON}")
    finally:
        try:
            if st.moonraker_get("/printer/info")["result"]["state"] == "ready":
                restore_xy_config(cfg)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
