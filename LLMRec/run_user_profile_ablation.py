import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


PROFILE_VARIANTS = ["pooled", "history_summary", "structured_profile"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the user-profile source ablation for LLMRec.")
    parser.add_argument("--dataset", default="steam", help="Dataset name passed to main.py.")
    parser.add_argument("--variants", nargs="+", default=PROFILE_VARIANTS, choices=PROFILE_VARIANTS, help="User profile embedding variants to compare.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2022, 2023, 2024], help="Seed list used for each profile variant.")
    parser.add_argument("--use-item-attribute", type=int, default=0, choices=[0, 1], help="Whether to keep item attribute augmentation enabled during profile-source comparison.")
    parser.add_argument("--output-dir", default="", help="Directory used to store per-run JSON files and the aggregated summary.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used to invoke main.py.")
    parser.add_argument("--main-path", default="", help="Optional explicit path to main.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without executing them.")
    args, extra_main_args = parser.parse_known_args()
    if extra_main_args and extra_main_args[0] == "--":
        extra_main_args = extra_main_args[1:]
    return args, extra_main_args


def aggregate_metric(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "var": float(arr.var(ddof=0)),
        "values": [float(v) for v in arr.tolist()],
    }


def aggregate_runs(run_results):
    ks = run_results[0]["ks"]
    summary = {"ks": ks, "recall": {}, "ndcg": {}}
    for metric_name in ("recall", "ndcg"):
        metric_matrix = np.asarray([run["best_metrics"][metric_name] for run in run_results], dtype=np.float64)
        for idx, k in enumerate(ks):
            summary[metric_name][str(k)] = aggregate_metric(metric_matrix[:, idx])
    return summary


def build_summary_markdown(summary):
    ks = summary["variants"][0]["aggregate"]["ks"]
    primary_k = str(ks[1]) if len(ks) > 1 else str(ks[0])
    lines = [
        f"# User Profile Source Summary ({summary['dataset']})",
        "",
        f"Seeds: {', '.join(str(seed) for seed in summary['seeds'])}",
        "",
        f"Item attribute enabled: {summary['use_item_attribute']}",
        "",
        "| Profile variant | Recall@{} mean+-std | NDCG@{} mean+-std |".format(primary_k, primary_k),
        "| --- | --- | --- |",
    ]
    for variant_result in summary["variants"]:
        recall_metric = variant_result["aggregate"]["recall"][primary_k]
        ndcg_metric = variant_result["aggregate"]["ndcg"][primary_k]
        lines.append(
            "| {} | {:.6f} +- {:.6f} | {:.6f} +- {:.6f} |".format(
                variant_result["variant"],
                recall_metric["mean"],
                recall_metric["std"],
                ndcg_metric["mean"],
                ndcg_metric["std"],
            )
        )
    return "\n".join(lines) + "\n"


def main():
    args, extra_main_args = parse_args()
    script_dir = Path(__file__).resolve().parent
    main_path = Path(args.main_path).resolve() if args.main_path else script_dir / "main.py"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else script_dir / "ablation_results" / f"profile_variants_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "variants_requested": args.variants,
        "use_item_attribute": bool(args.use_item_attribute),
        "main_path": str(main_path),
        "extra_main_args": extra_main_args,
        "variants": [],
    }

    for variant in args.variants:
        run_results = []
        for seed in args.seeds:
            experiment_name = f"text_plus_user_profile_{variant}"
            result_path = output_dir / f"{experiment_name}_seed{seed}.json"
            cmd = [
                args.python_bin,
                str(main_path),
                "--dataset",
                args.dataset,
                "--seed",
                str(seed),
                "--experiment_name",
                experiment_name,
                "--title",
                experiment_name,
                "--result_json_path",
                str(result_path),
                "--use_image_feat",
                "0",
                "--use_text_feat",
                "1",
                "--use_item_attribute",
                str(args.use_item_attribute),
                "--use_user_profile",
                "1",
                "--use_sample_augmentation",
                "0",
                "--user_profile_variant",
                variant,
            ]
            cmd.extend(extra_main_args)

            print("Running:", " ".join(cmd))
            if args.dry_run:
                continue

            completed = subprocess.run(
                cmd,
                cwd=str(script_dir),
                text=True,
                capture_output=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Command failed with code {}:\n{}\nSTDOUT:\n{}\nSTDERR:\n{}".format(
                        completed.returncode,
                        " ".join(cmd),
                        completed.stdout,
                        completed.stderr,
                    )
                )
            with result_path.open("r", encoding="utf-8") as handle:
                run_results.append(json.load(handle))

        if args.dry_run:
            continue

        summary["variants"].append(
            {
                "variant": variant,
                "runs": run_results,
                "aggregate": aggregate_runs(run_results),
            }
        )

    if args.dry_run:
        return

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    summary_md_path = output_dir / "summary.md"
    with summary_md_path.open("w", encoding="utf-8") as handle:
        handle.write(build_summary_markdown(summary))

    print(json.dumps({"summary_json": str(summary_path), "summary_md": str(summary_md_path)}, indent=2))


if __name__ == "__main__":
    main()
