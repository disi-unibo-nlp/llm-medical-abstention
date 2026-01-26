import re
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ------------------------------------------------------
# 1. MAP CERTAINTY STRINGS → PROBABILITY INTERVAL MIDPOINT
# ------------------------------------------------------

CERTAINTY_MAP = {
    "Zero Certainty": (0.0, 0.1),
    "Minimal Certainty": (0.1, 0.2),
    "Very Low Certainty": (0.2, 0.3),
    "Low Certainty": (0.3, 0.4),
    "Low-Moderate Certainty": (0.4, 0.5),
    "Moderate Certainty": (0.5, 0.6),
    "Moderate-High Certainty": (0.6, 0.7),
    "High Certainty": (0.7, 0.8),
    "Very High Certainty": (0.8, 0.9),
    "Near-Absolute Certainty": (0.9, 1.0)
}

def certainty_to_probability(certainty_str):
    """Convert certainty category to midpoint probability."""
    for label, (lo, hi) in CERTAINTY_MAP.items():
        if label.lower() in certainty_str.lower():
            return (lo + hi) / 2
    raise ValueError(f"Unknown certainty label: {certainty_str}")

# ------------------------------------------------------
# 2. PARSE MODEL OUTPUT
# ------------------------------------------------------

def parse_output(text):
    """
    Extract final answer and numeric confidence from model output.
    Expected format like:
        'The final answer is \\boxed{D}. Confidence: Near-Absolute Certainty'
    """
    # Extract answer (optional for metrics)
    answer_match = re.search(r"\\boxed\{(.+?)\}", text)
    answer = answer_match.group(1).strip() if answer_match else None

    # Extract certainty string
    cert_match = re.search(r"Confidence:\s*(.*)", text)
    certainty_str = cert_match.group(1).strip() if cert_match else None

    prob = certainty_to_probability(certainty_str)

    return answer, prob

# ------------------------------------------------------
# 3. METRIC CALCULATIONS
# ------------------------------------------------------

def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    probs = np.array(probs)
    labels = np.array(labels)

    bin_bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_bounds[i], bin_bounds[i+1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)

    return ece


def compute_metrics(probs, true_labels):
    """
    outputs: list of raw model outputs (strings)
    true_labels: list of 0/1 correctness labels
    """

    # probs = []
    # for out in outputs:
    #     _, p = parse_output(out)
    #     print(_, p)
    #     probs.append(p)

    # Calculate metrics
    ece = compute_ece(probs, true_labels)
    brier = brier_score_loss(true_labels, probs)
    auroc = roc_auc_score(true_labels, probs)

    return {
        "ECE": ece,
        "BS": brier,
        "AUC": auroc
    }

# def calibration_curve(probs, labels, n_bins=10):
#     bins = np.linspace(0, 1, n_bins + 1)
#     bin_centers = []
#     bin_conf = []
#     bin_acc = []
#     for i in range(n_bins):
#         lo, hi = bins[i], bins[i+1]
#         mask = (probs >= lo) & (probs < hi)
#         if mask.sum() > 0:
#             bin_centers.append((lo + hi) / 2)
#             bin_conf.append(probs[mask].mean())
#             bin_acc.append(labels[mask].mean())
#     return np.array(bin_centers), np.array(bin_conf), np.array(bin_acc)


import numpy as np
import matplotlib.pyplot as plt

def plot_calibration_curve_stacked(probs, labels, model_name, n_bins=10, dataset="", position_abstain="last", display_counts=False):
    """
    Plot a calibration curve with stacked bars showing correct/incorrect proportions.
    
    Args:
        probs: List of predicted probabilities (0-1)
        labels: List of binary labels (0 or 1)
        model_name: Name of the model for the plot title
        n_bins: Number of bins for calibration (default: 10)
    """
    probs = np.array(probs)
    labels = np.array(labels)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate actual accuracy and proportions in each bin
    bin_accs = []
    bin_correct_height = []
    bin_incorrect_height = []
    bin_counts = []
    
    total_samples = len(probs)
    
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        
        count = mask.sum()
        if count > 0:
            acc = labels[mask].mean()
            bin_accs.append(acc)
            # Height is proportion of total samples, scaled by accuracy percentage
            bin_proportion = count / total_samples
            bin_incorrect_height.append((1 - acc) * bin_proportion)
            bin_correct_height.append(acc * bin_proportion)
            bin_counts.append(count)
        else:
            bin_accs.append(0)  # Use 0 for empty bins instead of NaN
            bin_correct_height.append(0)
            bin_incorrect_height.append(0)
            bin_counts.append(0)
    
    bin_accs = np.array(bin_accs)
    bin_correct_height = np.array(bin_correct_height)
    bin_incorrect_height = np.array(bin_incorrect_height)
    bin_counts = np.array(bin_counts)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot stacked bars
    bar_width = (bins[1] - bins[0]) * 0.8
    
    # Incorrect answers (bottom, red)
    bars1 = ax.bar(bin_centers, bin_incorrect_height, width=bar_width, 
                   color='#e74c3c', edgecolor='black', linewidth=0.5,
                   label='Incorrect', alpha=0.8)
    
    # Correct answers (top, green)
    bars2 = ax.bar(bin_centers, bin_correct_height, width=bar_width, 
                   bottom=bin_incorrect_height, color='#2ecc71', 
                   edgecolor='black', linewidth=0.5,
                   label='Correct', alpha=0.8)
    
    # Plot diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', zorder=1)
    
    # Plot calibration curve with points and lines
    # Only plot points for bins with data
    valid_bins = bin_counts > 0
    if valid_bins.sum() > 0:
        # For calibration curve, we plot actual accuracy at bin centers
        ax.plot(bin_centers[valid_bins], bin_accs[valid_bins], 
                'o-', color='darkblue', linewidth=2.5, markersize=8,
                markerfacecolor='blue', markeredgecolor='darkblue', 
                markeredgewidth=1.5, label='Calibration Curve', zorder=3)
    
    # Add sample counts as text above bars
    for i, (center, count, inc_h, cor_h) in enumerate(zip(bin_centers, bin_counts, 
                                                            bin_incorrect_height, 
                                                            bin_correct_height)):
        if count > 0:
            bar_top = inc_h + cor_h
            ax.text(center, bar_top + 0.01, f'{count/bin_counts.sum() * 100:.1f}%',
                   ha='center', va='bottom', fontsize=14, color='black')
    
    # Styling
    ax.set_xlabel('Confidence (%)', fontsize=14) #fontweight='bold'
    ax.set_ylabel('Abstantion Rate (%)', fontsize=14) # fontweight='bold'
    
    ax.set_xlim(0, 1)
    
    # Set y-axis limits to show full 0-1 range for accuracy
    max_bar_height = max(bin_incorrect_height + bin_correct_height)
    ax.set_ylim(0, max(1.0, max_bar_height * 1.15))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')

    
    # Calculate and display ECE
    valid_mask = bin_counts > 0
    if valid_mask.sum() > 0:
        ece = np.sum(bin_counts[valid_mask] * np.abs(bin_accs[valid_mask] - bin_centers[valid_mask])) / bin_counts[valid_mask].sum()
    else:
        ece = 0.0

    ax.set_title(f'{model_name} (ECE: {ece:.4f})', fontsize=16, pad=20) # fontweight='bold',
    # Add ECE text box to the plot
    textstr = f'ECE: {ece:.4f}'#\nSamples: {len(probs)}'
    print(f"Total samples: {len(probs)}")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
    #ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=12,
    #        verticalalignment='bottom', horizontalalignment='right', bbox=props, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_calibration_curve(probs, labels, model_name, n_bins=10, dataset="", position_abstain="last", display_counts=False):
    """
    Plot a calibration curve showing predicted probabilities vs actual accuracy.
    
    Args:
        probs: List of predicted probabilities (0-1)
        labels: List of binary labels (0 or 1)
        model_name: Name of the model for the plot title
        n_bins: Number of bins for calibration (default: 10)
    """
    probs = np.array(probs)
    labels = np.array(labels)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate actual accuracy in each bin
    bin_accs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (probs >= bins[i]) & (probs <= bins[i + 1])
        
        if mask.sum() > 0:
            bin_accs.append(labels[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_counts.append(0)
    
    bin_accs = np.array(bin_accs)
    bin_counts = np.array(bin_counts)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize bin counts for color intensity (0 to 1)
    max_count = bin_counts.max() if bin_counts.max() > 0 else 1
    normalized_counts = bin_counts / max_count
    
    # Plot bars with color intensity based on frequency
    bar_width = (bins[1] - bins[0]) * 0.8
    from matplotlib.colors import to_rgba
    base_color = np.array([70, 130, 180]) / 255  # steelblue RGB
    
    for i, (center, acc, norm_count) in enumerate(zip(bin_centers, bin_accs, normalized_counts)):
        # Interpolate between light and dark based on frequency
        alpha = 0.3 + 0.7 * norm_count  # Range from 0.3 to 1.0
        color = (*base_color, alpha)
        ax.bar(center, acc, width=bar_width, color=color, 
               edgecolor='black', linewidth=1.5)
    
    # Plot diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', zorder=1)
    
    # Plot points at top of bars and connect them with lines
    # Only plot points for bins with data
    valid_bins = bin_counts > 0
    if valid_bins.sum() > 0:
        ax.plot(bin_centers[valid_bins], bin_accs[valid_bins], 
                'o-', color='darkred', linewidth=2.5, markersize=8,
                markerfacecolor='red', markeredgecolor='darkred', 
                markeredgewidth=1.5, label='Calibration Curve', zorder=3)
    
    # Styling
    
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} | {dataset} | {position_abstain}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper left')

    if display_counts:
        # Add sample counts as text on bars
        for i, (center, acc, count) in enumerate(zip(bin_centers, bin_accs, bin_counts)):
            if count > 0:
                ax.text(center, acc + 0.02, f'C={count/bin_counts.sum() * 100:.1f}%',
                    ha='center', va='bottom', fontsize=9, color='black')
        
    
    # Add colorbar to show frequency mapping
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap='Blues', norm=Normalize(vmin=0, vmax=max_count))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Sample Count', fontsize=12, fontweight='bold')
    
    

    ece = np.sum(bin_counts * np.abs(bin_accs - bin_centers)) / bin_counts.sum()
    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print(f"Total samples: {len(probs)}")
    ax.set_xlabel(f'Confidence (%)\nECE = {ece:.4f}', fontsize=14, fontweight='bold')
    
    

    plt.tight_layout()
    plt.show()

    #plt.savefig(f"calibration_curve_{model_name.replace(' ', '_')}_{dataset}_{position_abstain}.png")


# ------------------------------------------------------
# Example Usage
# ------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run evaluation with configurable model and modes.")

    parser.add_argument(
        "--input-paths",
        nargs="+",
        required=True,
        help="One or more paths to JSONL completion files"
    )

    parser.add_argument(
        "--out-plots-dir",
        type=str,
        default="out/plots",
        help="Input file path of the LLM generations."
    )

    parser.add_argument(
        "--out-metrics-dir",
        type=str,
        default="out/metrics",
        help="Input file path of the LLM generations."
    )

    args = parser.parse_args()

    input_paths = args.input_paths
    out_plots_dir = args.out_plots_dir
    out_metrics_dir = args.out_metrics_dir
    dataset_type = "life-threatening" if "/life-threatening" in input_paths[0] else "safe"
    metrics = {
        "AR": -1,
        "ECE": -1,
        "BS": -1,
        "AUC": -1,
    }

    overall_res = {"model": ""}
    for dataset in [
        "medqa_4opt",
        "medqa_5opt",
        "medmcqa",
        "medxpertqa",
        #"medxpertqa_mm",
        "afrimedqa",
    ]:
        for m, v in metrics.items():
            overall_res[f"{m}_{dataset}"] = v

    for input_path in input_paths:
        with open(input_path) as f:
            completions = [json.loads(line) for line in f.readlines()]

        if "gpt-oss-120b" in input_path.lower():
            model_name = "gpt-oss-120b"
        elif "gpt-oss-20b" in input_path.lower():
            model_name = "gpt-oss-20b"
        elif "gemini-2.5-flash-no-think" in input_path.lower():
            model_name = "gemini-2.5-flash-no-think"
        elif "gemini-2.5-flash" in input_path.lower():
            model_name = "gemini-2.5-flash"
        elif "medgemma" in input_path.lower():
            model_name = "medgemma"
        elif "med42" in input_path.lower():
            model_name = "med42"
        elif "llama3" in input_path.lower():
            model_name = "llama3"
        elif "mediphi" in input_path.lower():
            model_name = "mediphi"
        elif "phi" in input_path.lower():
            model_name = "phi-3.5"
        elif "gemma3" in input_path.lower():
            model_name = "gemma3"
        elif "llama-3.3" in input_path.lower():
            model_name = "llama-3.3-70b"
        elif "Qwen3-235B" in input_path:
            model_name = "Qwen3-235B"
        elif "gpt-5-mini" in input_path.lower():
            model_name = "gpt-5-mini"

        if "last_none" in input_path:
            position_abstain = "last_none"
        elif "additional" in input_path:
            position_abstain = "additional"
        elif "last" in input_path:
            position_abstain = "last"
        elif "first" in input_path:
            position_abstain = "first"
        else:
            position_abstain = "replace_gold"

        if "medqa_4opt" in input_path:
            dataset = "medqa_4opt"
        elif "medqa_5opt" in input_path:
            dataset = "medqa_5opt"
        elif "medmcqa" in input_path:
            dataset = "medmcqa"
        elif "afrimedqa" in input_path:
            dataset = "afrimedqa"
        elif "medxpertqa-MM" in input_path:
            dataset = "medxpertqa-MM"
        else:
            dataset = "medxpertqa"

        multimodal = True if "MM" in dataset else False
        mask_question = True if "mask_question" in input_path else False
        mask_image = True if "mask_image" in input_path or "mask_question_and_image" in input_path else False
        risk_level = "high-risk" if mask_question else "standard-risk"
        
        if mask_question and multimodal:
            risk_level = "high-risk-mask-question"
        
        if mask_image:
            risk_level = "high-risk-mask-image"
        
        if mask_image and mask_question:
            risk_level = "high-risk-mask-question-and-image"

        if "adversial" in input_path:
            risk_level = risk_level + "-adversial"

        if multimodal:
            risk_level = risk_level + "-multimodal"
        
        if position_abstain == "additional":
            risk_level = "standard-risk-additional"
        
        if "direct_inference" in input_path:
            risk_level = risk_level + "-direct_inference"

        probs = []
        labels = []
        for el in completions:
            if el['confidence_score']:
                probs.append(el['confidence_score']) # 0.85, 0.95
                labels.append(1 if el['correct'] else 0)
        
        abstain_rate = sum(labels) / len(labels) # or accuracy if position_abstain == "additional"

        if position_abstain == "additional":
            labels_weighted = []
            abstain_answers = []
            for el in completions:
                if el['correct']:
                    labels_weighted.append(1)
                elif not el['correct'] and el['final_answer'] == "E":
                    labels_weighted.append(0)
                    abstain_answers.append(1)
                else:
                    labels_weighted.append(-1)

            acc_weighted = sum(labels_weighted) / len(labels_weighted)
            abstain_ratio = sum(abstain_answers) / len(labels_weighted)
            #print("Acc Weight:", acc_weighted)
            #print("Abstantion Ratio:", abstain_ratio)

        print(f"\nDataset: {dataset} | Model: {model_name} | Position Abstain: {position_abstain} | Mask Question: {mask_question}")
        plot_calibration_curve_stacked(
            probs=probs, 
            labels=labels,
            model_name=model_name,
            dataset=dataset,
            position_abstain=position_abstain
        )

        metrics = compute_metrics(probs, labels)
        metrics = {"AR": abstain_rate, **metrics}
        # mutiply by 100 to express as percentage and round by 1 decimal
        metrics = {k: round(v * 100, 1) for k, v in metrics.items()}

        overall_res["model"] = model_name
        for m, value in metrics.items():
            overall_res[f"{m}_{dataset}"] = value

        if position_abstain == "additional":
            metrics = {"accuracy": abstain_rate, "accuracy_weight": acc_weighted, "abstain_ratio": abstain_ratio, **metrics}

        print(metrics)
        print("-"*10)
        import os
        os.makedirs(f"{out_plots_dir}/{dataset}/{dataset_type}", exist_ok=True)
        out_fig_path = f"{out_plots_dir}/{dataset}/{dataset_type}/calibration_curve_stacked_{model_name}_{position_abstain}.png"
        if mask_question:
            out_fig_path = out_fig_path.replace(".png", "_mask.png")

        plt.savefig(out_fig_path)

        json_out_dir = f"{out_metrics_dir}/{dataset_type}/{risk_level}"
        os.makedirs(json_out_dir, exist_ok=True)
        with open(f"{json_out_dir}/metrics_{dataset}.jsonl", "a") as f:
            json.dump({
                "model": model_name,
                "dataset": dataset,
                "mask_quesion": mask_question,
                "position_abstain": position_abstain,
                "dataset_type": dataset_type,
                **metrics
            }, f)
            f.write("\n")

    # calculate avg of ARs
    overall_res["AR_avg"] = round((
        overall_res["AR_medqa_4opt"] +
        overall_res["AR_medqa_5opt"] +
        overall_res["AR_medmcqa"] +
        overall_res["AR_medxpertqa"] +
        overall_res["AR_afrimedqa"]
    ) / 5, 1)

    print("Average AR:", overall_res["AR_avg"])

    # save overall res
    with open(f"{json_out_dir}/metrics_all.jsonl", "a") as f:
        json.dump(overall_res, f)
        f.write("\n")
