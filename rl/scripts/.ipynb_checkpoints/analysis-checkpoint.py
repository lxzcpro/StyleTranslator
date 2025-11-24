import json
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import pandas as pd

def find_latest_sample_files(output_dir="./outputs/benchmark"):
    """Find the latest pair of base and rl sample files."""
    base_files = glob.glob(os.path.join(output_dir, "base_samples_*.json"))
    rl_files = glob.glob(os.path.join(output_dir, "rl_samples_*.json"))
    
    if not base_files or not rl_files:
        raise FileNotFoundError(f"Missing sample files in {output_dir}. Did you run benchmark with --save?")
    
    # Sort by time
    latest_base = max(base_files, key=os.path.getmtime)
    latest_rl = max(rl_files, key=os.path.getmtime)
    
    print(f"Loaded Base Samples: {os.path.basename(latest_base)}")
    print(f"Loaded RL Samples:   {os.path.basename(latest_rl)}")
    
    return latest_base, latest_rl

def load_samples(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_metric(base_data, rl_data, metric_key, metric_name):
    """
    Run statistical tests for a specific metric.
    Returns dictionary with stats.
    """
    # Extract scores, default to 0.0 if missing
    base_scores = [item.get(metric_key, 0.0) for item in base_data]
    rl_scores = [item.get(metric_key, 0.0) for item in rl_data]
    
    if len(base_scores) != len(rl_scores):
        print(f"WARNING: Sample size mismatch for {metric_name}!")
        min_len = min(len(base_scores), len(rl_scores))
        base_scores = base_scores[:min_len]
        rl_scores = rl_scores[:min_len]

    n = len(base_scores)
    
    # Calculate means
    mean_base = np.mean(base_scores)
    mean_rl = np.mean(rl_scores)
    delta = mean_rl - mean_base
    
    # 1. Wilcoxon Signed-Rank Test (Non-parametric)
    # Robust against outliers and non-normal distributions (better for scores)
    try:
        w_stat, p_val_w = stats.wilcoxon(rl_scores, base_scores)
    except ValueError:
        # Happens if all differences are zero
        w_stat, p_val_w = 0.0, 1.0

    return {
        "metric": metric_name,
        "n": n,
        "mean_base": mean_base,
        "mean_rl": mean_rl,
        "delta": delta,
        "p_val_w": p_val_w,
        "significant": p_val_w < 0.05, # Using Wilcoxon as strict standard
        "base_scores": base_scores,
        "rl_scores": rl_scores
    }

def print_stats_table(results):
    """Print a nicely formatted table of results."""
    print("\n" + "="*85)
    print(f"{'METRIC':<20} | {'BASE':<10} | {'RL':<10} | {'DELTA':<10} | {'P-VALUE (Wilcoxon)':<20}")
    print("-" * 85)
    
    for res in results:
        sig_mark = "*" if res['significant'] else " "
        print(f"{res['metric']:<20} | {res['mean_base']:<10.4f} | {res['mean_rl']:<10.4f} | "
              f"{res['delta']:<+10.4f} | {res['p_val_w']:.4e} {sig_mark}")
    print("-" * 85)
    print("* p < 0.05 indicates statistically significant difference")
    print("="*85 + "\n")

def plot_all_metrics(results, save_path="benchmark_distribution.png"):
    """Plot violin plots for all metrics in a grid."""
    # We expect 6 metrics, so 2x3 grid is perfect
    n_metrics = len(results)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten()
    
    # Define colors
    palette = {'Original': '#1f77b4', 'RL Fine-tuned': '#ff7f0e'}
    
    for i, res in enumerate(results):
        ax = axes[i]
        metric_name = res['metric']
        
        # Prepare DataFrame for Seaborn
        data = []
        for s in res['base_scores']:
            data.append({'Model': 'Original', 'Score': s})
        for s in res['rl_scores']:
            data.append({'Model': 'RL Fine-tuned', 'Score': s})
            
        df = pd.DataFrame(data)
        
        # Violin Plot with inner boxplot
        sns.violinplot(
            data=df, x='Model', y='Score', 
            palette=palette, inner="box", ax=ax, alpha=0.6,
            linewidth=1.5
        )
        
        # Add strip plot (scatter) for individual points visibility
        # Sample if too many points to avoid clutter
        if len(df) > 200:
            df_sample = df.sample(200, random_state=42)
        else:
            df_sample = df
            
        sns.stripplot(
            data=df_sample, x='Model', y='Score', 
            color='black', alpha=0.3, size=3, jitter=True, ax=ax
        )
        
        # Titles and Labels
        p_val = res['p_val_w']
        sig_str = " (Significant)" if p_val < 0.05 else " (Not Sig.)"
        ax.set_title(f"{metric_name}\nDelta: {res['delta']:+.4f} | p={p_val:.1e}{sig_str}", fontsize=12, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Distribution of Performance Metrics (Original vs RL)", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust for suptitle
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Analyze significance of all metrics")
    parser.add_argument("--dir", default="./outputs/benchmark", help="Directory with json files")
    parser.add_argument("--base_file", default=None, help="Specific base sample json")
    parser.add_argument("--rl_file", default=None, help="Specific rl sample json")
    parser.add_argument("--output_img", default="benchmark_analysis_full.png", help="Output image path")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()
    
    try:
        # 1. Load Data
        if args.base_file and args.rl_file:
            base_path, rl_path = args.base_file, args.rl_file
        else:
            base_path, rl_path = find_latest_sample_files(args.dir)
            
        base_data = load_samples(base_path)
        rl_data = load_samples(rl_path)
        
        # 2. Analyze Metrics (Updated List)
        metrics_to_analyze = [
            ("style", "Style Score"),
            ("comet", "COMET Score"),
            ("bleu", "BLEU (Overall)"),
            ("bleu_1", "BLEU-1 (Vocab)"),
            ("bleu_4", "BLEU-4 (Fluency)"),
            ("format_score", "Format Score")
        ]
        
        results = []
        for key, name in metrics_to_analyze:
            res = analyze_metric(base_data, rl_data, key, name)
            results.append(res)
            
        # 3. Print Report
        print_stats_table(results)
        
        # 4. Plot
        if not args.no_plot:
            output_path = os.path.join(args.dir, args.output_img) if "/" not in args.output_img else args.output_img
            plot_all_metrics(results, output_path)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()