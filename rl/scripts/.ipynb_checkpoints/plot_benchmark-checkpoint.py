import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path
import argparse

def find_latest_benchmark_file(output_dir="./outputs/benchmark"):
    """Find the latest benchmark_metrics_*.json file."""
    # Search for files matching the pattern
    search_path = os.path.join(output_dir, "benchmark_metrics_*.json")
    files = glob.glob(search_path)
    
    if not files:
        raise FileNotFoundError(f"No benchmark metrics files found in {output_dir}")
    
    # Sort by modification time, newest first
    latest_file = max(files, key=os.path.getmtime)
    print(f"Loading latest benchmark file: {latest_file}")
    return latest_file

def load_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def plot_metrics(data, save_path=None):
    """Plot comparison metrics between Base and RL models."""
    if "base_model_metrics" not in data or "rl_model_metrics" not in data:
        print("Error: JSON must contain both 'base_model_metrics' and 'rl_model_metrics'")
        return

    base = data["base_model_metrics"]
    rl = data["rl_model_metrics"]

    # Metrics to compare (excluding BLEU for now as it has a different scale)
    metrics_0_1 = ["comet_score", "style_score", "format_score", "format_compliance_rate"]
    # Labels for the plot
    labels_0_1 = ["COMET", "Style", "Format Score", "Format Compliance"]
    
    base_values = [base.get(m, 0) for m in metrics_0_1]
    rl_values = [rl.get(m, 0) for m in metrics_0_1]

    # Setup the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Color palette
    color_base = '#1f77b4'  # Blue
    color_rl = '#ff7f0e'    # Orange
    
    # --- Plot 1: 0-1 Scale Metrics (COMET, Style, Format) ---
    x = np.arange(len(labels_0_1))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, base_values, width, label='Original Model', color=color_base, alpha=0.8)
    rects2 = ax1.bar(x + width/2, rl_values, width, label='RL Model', color=color_rl, alpha=0.8)
    
    ax1.set_ylabel('Score (0-1)')
    ax1.set_title('Quality & Style Metrics (Higher is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_0_1)
    ax1.set_ylim(0, 1.1)  # slightly above 1 for text
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # Add value labels
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1, ax1)
    autolabel(rects2, ax1)

    # --- Plot 2: BLEU Score (0-100 Scale) ---
    bleu_labels = ["BLEU", "BLEU-1", "BLEU-4"]
    bleu_keys = ["bleu_score", "bleu_1", "bleu_4"]
    
    base_bleu = [base.get(k, 0) for k in bleu_keys]
    rl_bleu = [rl.get(k, 0) for k in bleu_keys]
    
    x_bleu = np.arange(len(bleu_labels))
    
    rects3 = ax2.bar(x_bleu - width/2, base_bleu, width, label='Original Model', color=color_base, alpha=0.8)
    rects4 = ax2.bar(x_bleu + width/2, rl_bleu, width, label='RL Model', color=color_rl, alpha=0.8)
    
    ax2.set_ylabel('Score (0-100)')
    ax2.set_title('BLEU Scores (Higher is Better)')
    ax2.set_xticks(x_bleu)
    ax2.set_xticklabels(bleu_labels)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    autolabel(rects3, ax2)
    autolabel(rects4, ax2)

    # Final layout adjustments
    plt.suptitle(f'Benchmark Results Comparison\nTimestamp: {data.get("timestamp", "N/A")}', fontsize=14)
    plt.tight_layout()
    
    # Save or Show
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--dir", type=str, default="./outputs/benchmark", help="Directory containing benchmark json files")
    parser.add_argument("--file", type=str, default=None, help="Specific json file path (optional)")
    parser.add_argument("--output", type=str, default="benchmark_plot.png", help="Output image filename")
    
    args = parser.parse_args()
    
    try:
        if args.file:
            json_file = args.file
        else:
            json_file = find_latest_benchmark_file(args.dir)
            
        data = load_data(json_file)
        plot_metrics(data, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        