"""Merge the seven-model localization payload with TopoLoc-ConvAE."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from configs.config import sys_config
from configs.paths import OUTPUT_ROOT

CASE = sys_config["case_name"]
BASE = OUTPUT_ROOT / "detection5" / CASE / "localization_results.json"
TOPOLOC = OUTPUT_ROOT / "topoloc" / CASE / f"topoloc_convae_localization_{CASE}.json"
OUTPUT = OUTPUT_ROOT / "detection5_topoloc" / CASE / "localization_results.json"
FINAL_MODELS = (
    "ConvAE",
    "LSTMAE",
    "AT",
    "PIConvAE",
    "ReVGAE",
    "AT-GVAE",
    "TopoLoc-ConvAE",
)


def main():
    base = json.loads(BASE.read_text(encoding="utf-8"))
    topoloc = json.loads(TOPOLOC.read_text(encoding="utf-8"))
    keep = set(FINAL_MODELS[:-1])
    base["model_status"] = [
        row for row in base["model_status"] if row["model"] in keep
    ] + [topoloc["model_status"]]
    base["metrics"] = [row for row in base["metrics"] if row["model"] in keep]
    base["metrics"].extend(topoloc["metrics"])
    base["sample_metrics"] = [
        row for row in base["sample_metrics"] if row["model"] in keep
    ]
    base["sample_metrics"].extend(topoloc["sample_metrics"])

    topoloc_scores = {
        row["measurement_index"]: row["TopoLoc-ConvAE"]
        for row in topoloc["single_sample"]
    }
    standard_keys = {
        "measurement_index",
        "measurement_label",
        "true_perturbation",
        "is_significant",
        "attacked_buses",
    }
    for row in base["single_sample"]:
        for key in list(row):
            if key not in standard_keys and key not in keep:
                del row[key]
        row["TopoLoc-ConvAE"] = topoloc_scores[row["measurement_index"]]

    base["metadata"]["score_definition"] = (
        "last-frame normalized measurement score: absolute residual for "
        "ConvAE/LSTMAE/AT/PIConvAE; squared residual for ReVGAE; 0.7 generator "
        "+ 0.3 recognizer weighted squared residual for AT-GVAE; ConvAE absolute "
        "residual calibrated by the validation-selected node topology prior for "
        "TopoLoc-ConvAE"
    )
    base["metadata"]["topoloc_training"] = topoloc["metadata"]["training"]
    base["metadata"]["topoloc_fusion_weight"] = topoloc["metadata"][
        "selected_fusion_weight"
    ]
    base["topoloc_validation_fusion"] = topoloc["validation_fusion"]
    base["topoloc_main_ablation"] = topoloc["main_ablation"]
    extended = json.loads(
        (OUTPUT_ROOT / "topoloc" / CASE / f"topoloc_ablation_extended_{CASE}.json").read_text(encoding="utf-8")
    )
    base["topoloc_extended_ablation"] = extended["summary"]
    base["topoloc_extended_note"] = extended["note"]
    base["topoloc_gate_diagnostics"] = topoloc["gate_diagnostics"]

    order = {name: index for index, name in enumerate(FINAL_MODELS)}
    base["metrics"].sort(
        key=lambda row: (order[row["model"]], row["ang_no"], row["ang_str"])
    )
    base["sample_metrics"].sort(
        key=lambda row: (order[row["model"]], row["sample_index"])
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT}")
    print(
        f"models={len(base['model_status'])}, metrics={len(base['metrics'])}, "
        f"samples={len(base['sample_metrics'])}"
    )


if __name__ == "__main__":
    main()
