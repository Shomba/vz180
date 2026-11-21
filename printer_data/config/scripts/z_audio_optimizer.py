#!/home/shoomba/klippy-env/bin/python
"""
Otimizador Acustico Automatico para o Eixo Z (TMC2208) usando o microfone da webcam.
Testa combinacoes seguras de:
  - Microsteps (mres: 8, 4, 2, e 1/fullstep com corrente segura)
  - Parametros do chopper (toff, hstrt, hend, tbl, multistep_filt)
  - Corrente segura (0.85A, 1.0A, 1.15A com tbl=2/toff>=4)
  - Modo harmonico (1x fullstep, 2x microstep, 4x microstep)
Mede: Volume (RMS dBFS), Clareza/Pureza Tonal (dB acima do piso + % energia na nota alvo),
e Afinacao (desvio em cents).
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

OUT_JSON = Path("/home/shoomba/printer_data/config/z_audio_optimization.json")
MRES_MAP = {32: 3, 16: 4, 8: 5, 4: 6, 2: 7, 1: 8}


def measure_clarity(
    samples: np.ndarray,
    noise_freqs: np.ndarray,
    noise_spec: np.ndarray,
    target_hz: float,
) -> dict:
    """Measure volume, SNR, exact peak frequency, and tonal clarity (fraction of clean spectral energy at target note & its octaves)."""
    trim = int(len(samples) * 0.15)
    core = samples[trim : len(samples) - trim] if len(samples) > 1000 else samples
    freqs, spec = st.compute_spectrum(core, f_min=70.0, f_max=5000.0)
    noise_interp = np.interp(freqs, noise_freqs, noise_spec)
    clean_spec = np.maximum(spec - 1.5 * noise_interp, 0.0)

    power = clean_spec ** 2
    total_power = float(np.sum(power)) + 1e-12

    # Dominant peak
    peak_idx = int(np.argmax(clean_spec))
    peak_mag = float(clean_spec[peak_idx])
    median_floor = float(np.median(clean_spec) + 1e-9)
    snr_db = 20.0 * math.log10((peak_mag + 1e-9) / median_floor)

    if 0 < peak_idx < len(clean_spec) - 1:
        y0, y1, y2 = clean_spec[peak_idx - 1], clean_spec[peak_idx], clean_spec[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        peak_hz = float(freqs[peak_idx] + delta * (freqs[1] - freqs[0]))
    else:
        peak_hz = float(freqs[peak_idx])

    # Energy within +-3% of target_hz (exact fundamental clarity!)
    mask_fund = (freqs >= target_hz * 0.97) & (freqs <= target_hz * 1.03)
    fund_power = float(np.sum(power[mask_fund]))
    fund_clarity_pct = 100.0 * fund_power / total_power

    # Energy at target_hz and its integer harmonics (2x, 3x, 4x) -> musical tone purity vs broadband noise
    mask_harm = np.zeros_like(mask_fund)
    for h in (1, 2, 3, 4):
        fh = target_hz * h
        if fh <= freqs[-1]:
            mask_harm |= (freqs >= fh * 0.97) & (freqs <= fh * 1.03)
    tonal_purity_pct = 100.0 * float(np.sum(power[mask_harm])) / total_power

    # Peak specifically inside target_hz +-3%
    fund_peak_db = -120.0
    if np.any(mask_fund):
        f_max = float(np.max(clean_spec[mask_fund]))
        fund_peak_db = 20.0 * math.log10(f_max + 1e-9)

    rms_db = 20.0 * math.log10(float(np.sqrt(np.mean(core ** 2))) + 1e-9)
    note_name, _ = st.hz_to_note_name(peak_hz)
    cents = 1200.0 * math.log2(peak_hz / target_hz) if peak_hz > 0 and target_hz > 0 else 0.0

    # Composite score: rewards loudness (rms_db) + exact target note clarity + tonal purity
    # Penalizes if peak is not on target_hz (off-pitch or wrong octave)
    on_target_bonus = 15.0 if abs(cents) < 35.0 else (5.0 if abs(abs(cents) - 1200.0) < 35.0 else -10.0)
    score = (rms_db + 35.0) * 1.5 + (fund_clarity_pct * 0.4) + (tonal_purity_pct * 0.25) + on_target_bonus

    return {
        "target_hz": round(target_hz, 2),
        "peak_hz": round(peak_hz, 2),
        "peak_note": note_name,
        "cents": round(cents, 1),
        "rms_db": round(rms_db, 1),
        "snr_db": round(snr_db, 1),
        "fund_clarity_pct": round(fund_clarity_pct, 1),
        "tonal_purity_pct": round(tonal_purity_pct, 1),
        "score": round(score, 1),
    }


def apply_z_config(
    orig_ms_z: int,
    target_ms: int,
    current: float,
    toff: int,
    hstrt: int,
    hend: int,
    tbl: int,
    mfilt: int,
    intpol: int,
    en_spread: int,
    max_z_accel: float,
) -> float:
    scale_z = orig_ms_z / float(target_ms)
    scaled_accel = max(5.0, round(max_z_accel / scale_z, 2))
    mres = MRES_MAP[target_ms]
    cmds = [
        "M400",
        f"SET_VELOCITY_LIMIT ACCEL={scaled_accel}",
        f"SET_TMC_CURRENT STEPPER=stepper_z CURRENT={current:.2f}",
        f"SET_TMC_FIELD STEPPER=stepper_z FIELD=en_spreadcycle VALUE={en_spread}",
        f"SET_TMC_FIELD STEPPER=stepper_z FIELD=multistep_filt VALUE={mfilt}",
        f"SET_TMC_FIELD STEPPER=stepper_z FIELD=toff VALUE={toff}",
        f"SET_TMC_FIELD STEPPER=stepper_z FIELD=hstrt VALUE={hstrt}",
        f"SET_TMC_FIELD STEPPER=stepper_z FIELD=hend VALUE={hend}",
        f"SET_TMC_FIELD STEPPER=stepper_z FIELD=tbl VALUE={tbl}",
        f"SET_TMC_FIELD STEPPER=stepper_z FIELD=intpol VALUE={intpol}",
        f"SET_TMC_FIELD STEPPER=stepper_z FIELD=mres VALUE={mres}",
    ]
    st.run_gcode("\n".join(cmds))
    return scale_z


def restore_z_config(orig_ms_z: int, orig_current: float, orig_accel: float) -> None:
    orig_mres = MRES_MAP.get(orig_ms_z, 3)
    cmds = [
        "G90",
        "M400",
        f"SET_TMC_FIELD STEPPER=stepper_z FIELD=mres VALUE={orig_mres}",
        "SET_TMC_FIELD STEPPER=stepper_z FIELD=intpol VALUE=0",
        "SET_TMC_FIELD STEPPER=stepper_z FIELD=toff VALUE=1",
        "SET_TMC_FIELD STEPPER=stepper_z FIELD=hstrt VALUE=2",
        "SET_TMC_FIELD STEPPER=stepper_z FIELD=hend VALUE=3",
        "SET_TMC_FIELD STEPPER=stepper_z FIELD=tbl VALUE=1",
        "SET_TMC_FIELD STEPPER=stepper_z FIELD=multistep_filt VALUE=1",
        "SET_TMC_FIELD STEPPER=stepper_z FIELD=en_spreadcycle VALUE=1",
        f"SET_TMC_CURRENT STEPPER=stepper_z CURRENT={orig_current:.2f}",
        f"SET_VELOCITY_LIMIT ACCEL={orig_accel:.1f}",
    ]
    st.run_gcode("\n".join(cmds))


def play_z_note(
    source: str,
    noise_freqs: np.ndarray,
    noise_spec: np.ndarray,
    target_hz: float,
    drive_div: float,
    steps_per_mm_z: float,
    scale_z: float,
    dur: float = 0.38,
) -> dict:
    """Move Z up and down at physical speed corresponding to target_hz / drive_div."""
    cmd_hz = target_hz / drive_div
    v_phys = cmd_hz / steps_per_mm_z  # mm/s
    dist_phys = min(max(v_phys * dur, 1.5), 14.0)
    actual_dur = dist_phys / v_phys if v_phys > 0 else dur
    dist_cmd = dist_phys / scale_z
    f_cmd = (v_phys * 60.0) / scale_z

    gcode = (
        f"G91\n"
        f"G1 Z{dist_cmd:.5f} F{f_cmd:.2f}\n"
        f"G1 Z{-dist_cmd:.5f} F{f_cmd:.2f}\n"
        f"M400"
    )
    rec_dur = (actual_dur * 2.0) + 0.14
    rec_holder: list[np.ndarray] = []
    t_rec = threading.Thread(target=lambda: rec_holder.append(st.record_audio(source, rec_dur)), daemon=True)
    t_rec.start()
    time.sleep(0.05)
    st.run_gcode(gcode)
    t_rec.join(timeout=rec_dur + 3.0)
    res = measure_clarity(rec_holder[0], noise_freqs, noise_spec, target_hz)
    res["v_phys_mm_s"] = round(v_phys, 2)
    res["dist_phys_mm"] = round(dist_phys, 2)
    return res


def main() -> int:
    source = st.find_camera_mic()
    info = st.moonraker_get("/printer/info")["result"]
    if info["state"] != "ready":
        print(f"ERRO: Klipper state={info['state']}")
        return 1

    cfg = st.get_stepper_settings()
    orig_ms_z = int(cfg["stepper_z"]["microsteps"])
    rot_z = float(cfg["stepper_z"]["rotation_distance"])
    steps_per_mm_z = 200.0 / rot_z
    orig_current = float(cfg["tmc2208 stepper_z"]["run_current"])
    orig_accel = float(cfg["printer"]["max_accel"])
    max_z_accel = float(cfg["printer"]["max_z_accel"])

    print(f"[Mic] {source}")
    noise_samples = st.record_audio(source, 0.8)
    noise_freqs, noise_spec = st.compute_spectrum(noise_samples, f_min=70.0, f_max=5000.0)
    noise_rms = 20.0 * math.log10(float(np.sqrt(np.mean(noise_samples ** 2))) + 1e-9)
    print(f"[Ruido Ambiente] {noise_rms:.1f} dBFS | Z: {orig_ms_z}ms, {steps_per_mm_z:.2f} steps/mm, {orig_current}A")

    homed = st.moonraker_get("/printer/objects/query?toolhead")["result"]["status"]["toolhead"]["homed_axes"]
    if "z" not in homed:
        print("[Klipper] Fazendo G28 Z...")
        st.run_gcode("G28 Z", timeout=60.0)

    # Park Z safely at 40mm
    st.run_gcode("G90\nG1 Z40 F1800\nM400")

    # Candidate Z TMC2208 audio profiles (all safe currents <= 1.05A to avoid s2vsb!)
    # Format: (name, ms, current, toff, hstrt, hend, tbl, mfilt, intpol, en_spread, drive_div)
    candidates = [
        # Baseline stock autotune (32ms)
        ("stock_32ms_1x",          32, 1.00, 1, 2, 3, 1, 1, 0, 1, 1.0),
        # 8 microsteps with loud chopper
        ("8ms_loud_1x",             8, 1.00, 4, 7, 6, 2, 0, 0, 1, 1.0),
        ("8ms_loud_4x",             8, 1.00, 4, 7, 6, 2, 0, 0, 1, 4.0),
        # 4 microsteps: stock chopper vs loud chopper vs punchy chopper, 1x vs 4x drive!
        ("4ms_stock_1x",            4, 1.00, 1, 2, 3, 1, 1, 0, 1, 1.0),
        ("4ms_stock_4x",            4, 1.00, 1, 2, 3, 1, 1, 0, 1, 4.0),
        ("4ms_loud_1x",             4, 1.00, 4, 7, 6, 2, 0, 0, 1, 1.0),
        ("4ms_loud_2x",             4, 1.00, 4, 7, 6, 2, 0, 0, 1, 2.0),
        ("4ms_loud_4x",             4, 1.00, 4, 7, 6, 2, 0, 0, 1, 4.0),
        ("4ms_punchy_4x",           4, 1.00, 5, 7, 10, 2, 0, 0, 1, 4.0),
        ("4ms_toff5_h7_1x",         4, 1.00, 5, 7, 10, 2, 0, 0, 1, 1.0),
        # 2 microsteps (half-step!) at safe 0.95A and 1.00A with tbl=2, toff=4 (prevents s2vsb!)
        ("2ms_loud_1x",             2, 0.95, 4, 7, 6, 2, 0, 0, 1, 1.0),
        ("2ms_loud_2x",             2, 0.95, 4, 7, 6, 2, 0, 0, 1, 2.0),
        ("2ms_punchy_2x",           2, 0.95, 5, 7, 10, 2, 0, 0, 1, 2.0),
        ("2ms_1A_2x",               2, 1.00, 4, 7, 6, 2, 0, 0, 1, 2.0),
        # 1 microstep (fullstep!) at gentle 0.80A + tbl=2 + toff=5 so it does NOT trip s2vsb
        ("1ms_0.80A_1x",            1, 0.80, 5, 7, 6, 2, 0, 0, 1, 1.0),
        ("1ms_0.80A_2x",            1, 0.80, 5, 7, 6, 2, 0, 0, 1, 2.0),
    ]

    probe_notes = [("C4", 261.63), ("G4", 392.00), ("C5", 523.25)]
    bench_results = []

    print(f"\n{'Perfil':<18} | {'RMS(dB)':>7} | {'PeakHz':>7} | {'Cents':>7} | {'Fund%':>6} | {'Pure%':>6} | {'Score':>6} | Notas")
    print("-" * 98)

    try:
        for (name, ms, cur, toff, hstrt, hend, tbl, mfilt, intpol, en_spread, div) in candidates:
            # Verify Klipper is still ready before each candidate
            st_now = st.moonraker_get("/printer/info")["result"]["state"]
            if st_now != "ready":
                print(f"ABORT: Klipper saiu de ready ({st_now}) antes de {name}")
                break

            try:
                scale_z = apply_z_config(orig_ms_z, ms, cur, toff, hstrt, hend, tbl, mfilt, intpol, en_spread, max_z_accel)
                time.sleep(0.08)
                note_res = []
                for nlabel, nhz in probe_notes:
                    r = play_z_note(source, noise_freqs, noise_spec, nhz, div, steps_per_mm_z, scale_z, dur=0.35)
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
                        "multistep_filt": mfilt, "intpol": intpol, "en_spreadcycle": en_spread,
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
                    f"{name:<18} | {avg_rms:7.1f} | {g4['peak_hz']:7.1f} | {g4['cents']:+7.1f} | "
                    f"{avg_fund:5.1f}% | {avg_pure:5.1f}% | {avg_score:6.1f} | {note_summary}"
                )
            except Exception as exc:
                print(f"{name:<18} | ERRO: {exc}")
                break

        # Phase 2: Run full octave sweep on the TOP winner!
        winner_sweep = []
        if bench_results and st.moonraker_get("/printer/info")["result"]["state"] == "ready":
            winner = max(bench_results, key=lambda x: x["avg_score"])
            wp = winner["params"]
            print(f"\n=== VENCEDOR Z: {winner['profile']} (Score={winner['avg_score']}, RMS={winner['avg_rms_db']} dBFS, Clareza={winner['avg_fund_clarity_pct']}%) ===")
            print(f"Params: {wp}")
            print("\n=== Varredura de Oitavas no Perfil Vencedor ===")
            scale_z = apply_z_config(
                orig_ms_z, wp["microsteps"], wp["current"], wp["toff"], wp["hstrt"],
                wp["hend"], wp["tbl"], wp["multistep_filt"], wp["intpol"], wp["en_spreadcycle"], max_z_accel
            )
            sweep_notes = [
                ("C3", 130.81), ("E3", 164.81), ("G3", 196.00), ("A3", 220.00),
                ("C4", 261.63), ("E4", 329.63), ("G4", 392.00), ("A4", 440.00),
                ("C5", 523.25), ("E5", 659.25), ("G5", 783.99), ("A5", 880.00),
                ("C6", 1046.50), ("G6", 1567.98),
            ]
            print("Nota | Vel(mm/s) | Alvo(Hz) | Medido(Hz) | NotaReal | Cents   | Fund% | Pure% | RMS(dB)")
            print("-" * 92)
            for nlabel, nhz in sweep_notes:
                v_test = (nhz / wp["drive_div"]) / steps_per_mm_z
                if v_test > 95.0:
                    continue
                r = play_z_note(source, noise_freqs, noise_spec, nhz, wp["drive_div"], steps_per_mm_z, scale_z, dur=0.35)
                r["note"] = nlabel
                winner_sweep.append(r)
                print(
                    f"{nlabel:<4} | {r['v_phys_mm_s']:9.2f} | {nhz:8.2f} | {r['peak_hz']:10.2f} | "
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
                restore_z_config(orig_ms_z, orig_current, orig_accel)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
