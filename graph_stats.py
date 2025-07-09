import os
import json
import argparse
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(
        description="Analysis of graph metrics with flexible model formula via JSON configuration.",
        epilog="Example: python graph_stats.py --config config.json"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    input_file = config["input_file"]
    output_folder = config.get("output_folder", "results")
    model_template = config["model"]
    analysis_type = config.get("type", "global")

    # Load data
    df = pd.read_csv(input_file)
    df['subject'] = df['subject'].astype(str)
    if "timepoint" in df.columns:
        df['timepoint'] = df['timepoint'].astype("category")

    # Create output folders
    os.makedirs(os.path.join(output_folder, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "summaries"), exist_ok=True)

    # Determine metrics to analyze
    if analysis_type == "nodal":
        metrics = [col for col in df.columns if col not in ["subject", "timepoint", "node"]]
    else:
        metrics = [col for col in df.columns if col not in ["subject", "timepoint"]]

    # Ensure small_world is included if present
    if "small_world" in df.columns and "small_world" not in metrics:
        metrics.append("small_world")

    for metric in metrics:
        print(f"\n🔍 Analyzing {metric}...")

        # Insert model formula
        model_formula = model_template.format(metric=metric)
        try:
            model = smf.mixedlm(model_formula, df, groups="subject")
            result = model.fit()

            # Save summary
            summary_path = os.path.join(output_folder, "summaries", f"{metric}_summary.txt")
            with open(summary_path, "w") as f:
                f.write(result.summary().as_text())
            print(f"✔️ Summary saved to {summary_path}")

            # Determine significance stars
            stars = {}
            for level in result.params.index:
                if "C(timepoint)" in level:
                    p = result.pvalues[level]
                    if p < 0.001:
                        stars[level] = "***"
                    elif p < 0.01:
                        stars[level] = "**"
                    elif p < 0.05:
                        stars[level] = "*"

            # Plot with annotation
            plt.figure(figsize=(8, 5))
            sns.pointplot(data=df, x="timepoint", y=metric, errorbar=('ci', 95), capsize=0.1)
            plt.title(f"{metric.replace('_', ' ').title()} over Timepoints")
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xlabel("Timepoint")

            y_max = df[metric].max()
            offset = (df[metric].max() - df[metric].min()) * 0.05

            for i, level in enumerate(stars):
                if "[T." in level:
                    try:
                        tp = int(level.split("[T.")[1].rstrip("]"))
                        plt.text(tp - 1, y_max + offset, stars[level], ha='center', va='bottom', fontsize=12, color='red')
                    except Exception:
                        continue

            plt.tight_layout()
            plot_path = os.path.join(output_folder, "plots", f"{metric}_plot.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
            print(f"📊 Plot saved to {plot_path}")

        except Exception as e:
            print(f"❌ Model/Plot failed for {metric}: {e}")
            continue

if __name__ == "__main__":
    main()
