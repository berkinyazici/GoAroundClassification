from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_train(data: str, config: str, outdir: str) -> dict:
    cmd = [
        "python",
        "-m",
        "goaround.train",
        "--data",
        data,
        "--config",
        config,
        "--outdir",
        outdir,
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(res.stdout)


def latest_run_dir(outdir: Path) -> Path:
    runs = sorted([p for p in outdir.iterdir() if p.is_dir()])
    return runs[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--report", default="reports/ablation_summary.md")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_train(args.data, "configs/adsb_only.yaml", args.outdir)
    adsb_dir = latest_run_dir(outdir)
    adsb_metrics = json.loads((adsb_dir / "metrics.json").read_text())

    run_train(args.data, "configs/adsb_plus_metar.yaml", args.outdir)
    metar_dir = latest_run_dir(outdir)
    metar_metrics = json.loads((metar_dir / "metrics.json").read_text())

    lines = [
        "# Ablation Summary (ADS-B vs ADS-B+METAR)",
        "",
        "| Metric | ADS-B only | ADS-B+METAR | Delta |",
        "|---|---:|---:|---:|",
    ]
    for k in sorted(adsb_metrics.keys()):
        a = float(adsb_metrics[k])
        b = float(metar_metrics[k])
        lines.append(f"| {k} | {a:.4f} | {b:.4f} | {b-a:+.4f} |")

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
