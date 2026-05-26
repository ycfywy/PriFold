from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_loader
from symfold.metrics import contact_metrics
from symfold.train import build_model, load_config, move_to_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test_sets", default="bprna-test,rnastralign-test,archiveii-test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = load_config(args.config) if args.config else ckpt["config"]
    if args.num_steps is not None:
        config.setdefault("sampling", {})["num_steps"] = args.num_steps

    class Args:
        pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = config["paths"].get("pretrained_lm_dir", str(ROOT / "model"))
    lm_args.model_scale = config["model"].get("mars_scale", "lx")
    extractor, tokenizer = get_extractor(lm_args)

    device = torch.device(config.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    model = build_model(config, extractor)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    results = {}
    for stage in [x.strip() for x in args.test_sets.split(",") if x.strip()]:
        loader = build_loader(stage, config, tokenizer, shuffle=False)
        merged = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0, "gt_pairs": 0.0, "pred_pairs": 0.0}
        n = 0
        with torch.no_grad():
            for batch in loader:
                batch = move_to_device(batch, device)
                pred, _ = model.sample(batch, num_steps=config.get("sampling", {}).get("num_steps", 10))
                m = contact_metrics(pred, batch["contact"], batch["length"])
                bs = m["n"]
                n += bs
                for key in merged:
                    merged[key] += m[key] * bs
        res = {key: val / max(n, 1) for key, val in merged.items()}
        res["N"] = n
        results[stage] = res
        print(f"[{stage}] N={n} F1={res['f1']:.4f} P={res['precision']:.4f} R={res['recall']:.4f} MCC={res['mcc']:.4f}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"ckpt": args.ckpt, "results": results}, f, indent=2)
        print(f"saved to {out}")


if __name__ == "__main__":
    main()
