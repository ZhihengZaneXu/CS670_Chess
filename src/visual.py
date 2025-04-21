import glob
import os
import re

import matplotlib

matplotlib.use("Agg")  # Use Agg backend for non-interactive environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Function to extract metrics from TensorBoard logs
def extract_metrics_from_tensorboard(log_dir):
    """
    Extract metrics related to move quality from TensorBoard logs.

    Parameters:
    log_dir (str): Path to TensorBoard log directory

    Returns:
    pandas.DataFrame: DataFrame containing extracted metrics
    """
    # Find all event files
    event_files = glob.glob(
        os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True
    )

    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {log_dir}")

    # Metrics to extract
    metrics = [
        "quality/perfect_move_rate",
        "quality/correct_piece_selection_rate",
        "quality/correct_move_selection_rate",
    ]

    # Initialize empty DataFrame
    all_data = []

    # Process each event file
    for event_file in event_files:
        print(f"Processing {event_file}")
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        # Check which metrics are available
        available_metrics = set(event_acc.Tags()["scalars"])
        metrics_to_extract = [m for m in metrics if m in available_metrics]

        if not metrics_to_extract:
            print(f"None of the required metrics found in {event_file}")
            continue

        # Extract metrics
        for metric in metrics_to_extract:
            events = event_acc.Scalars(metric)
            data = {
                "step": [e.step for e in events],
                "wall_time": [e.wall_time for e in events],
                "value": [e.value for e in events],
                "metric": [metric] * len(events),
            }

            all_data.append(pd.DataFrame(data))

    if not all_data:
        raise ValueError("No relevant metrics found in the TensorBoard logs")

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Pivot the data for easier plotting
    pivot_df = combined_df.pivot_table(
        index=["step", "wall_time"], columns="metric", values="value"
    ).reset_index()

    # Rename columns for clarity
    pivot_df = pivot_df.rename(
        columns={
            "quality/perfect_move_rate": "perfect_move_rate",
            "quality/correct_piece_selection_rate": "correct_piece_selection_rate",
            "quality/correct_move_selection_rate": "correct_move_selection_rate",
        }
    )

    return pivot_df


# Function to create histograms with specified bins
def visualize_histograms(df, bins=None):
    """
    Create histograms for each metric to show distribution.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the metrics
    bins (list): Custom bin edges (if None, uses default binning)
    """
    metrics = [
        "perfect_move_rate",
        "correct_piece_selection_rate",
        "correct_move_selection_rate",
    ]

    titles = {
        "perfect_move_rate": "Perfect Move Rate Distribution",
        "correct_piece_selection_rate": "Correct Piece Selection Rate Distribution",
        "correct_move_selection_rate": "Correct Move Selection Rate Distribution",
    }

    colors = {
        "perfect_move_rate": "green",
        "correct_piece_selection_rate": "blue",
        "correct_move_selection_rate": "orange",
    }

    # Use custom bins if specified, otherwise use default
    if bins is None:
        bins = [0, 0.25, 0.5, 0.8, 0.9, 1.0]

    for metric in metrics:
        if metric in df.columns:
            # Drop NaN values for accurate calculations
            data = df[metric].dropna()

            fig, ax = plt.subplots(figsize=(10, 6))

            # Create histogram
            n, bin_edges, patches = ax.hist(
                data, bins=bins, color=colors[metric], alpha=0.7, edgecolor="black"
            )

            # Calculate total count for percentage (using only non-NaN values)
            total_count = len(data)

            # Calculate exact percentage (ensure they sum to 100%)
            percentages = [count / total_count * 100 for count in n]
            total_pct = sum(percentages)

            # Print for debugging
            print(f"\n{metric} histogram statistics:")
            print(f"Total count: {total_count}")
            print(f"Counts per bin: {n}")
            print(f"Raw percentages: {percentages}")
            print(f"Total percentage: {total_pct}%")

            # Add percentage labels on each bar (correctly summing to 100%)
            for i in range(len(n)):
                # Calculate exact percentage
                pct = 100 * n[i] / total_count

                # Position label in the middle of the bar
                ax.text(
                    (bin_edges[i] + bin_edges[i + 1]) / 2,
                    n[i] + (max(n) * 0.02),  # Slight offset for visibility
                    f"{pct:.1f}%",
                    ha="center",
                    fontweight="bold",
                )

            ax.set_title(titles[metric])
            ax.set_xlabel("Rate Value (0-1)")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3, axis="y")

            # Add a more descriptive x-axis
            bin_labels = []
            for i in range(len(bins) - 1):
                bin_labels.append(f"{bins[i]:.2f}-{bins[i+1]:.2f}")

            plt.xticks(
                [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)],
                bin_labels,
                rotation=45,
            )

            # Add a text annotation with the total percentage
            plt.figtext(0.9, 0.01, f"Total: {total_pct:.1f}%", ha="right")

            plt.tight_layout()
            plt.savefig(f"{metric}_histogram.png")

            # Don't use plt.show() when using Agg backend
            plt.close(fig)

            print(f"Histogram saved to {metric}_histogram.png")
        else:
            print(f"Metric '{metric}' not found in the data")


# Main execution block
def main():
    # Specify the TensorBoard log directory
    log_dir = "./trained_models/self/tensorboard/"

    try:
        # Extract metrics
        print(f"Extracting metrics from {log_dir}...")
        df = extract_metrics_from_tensorboard(log_dir)

        # Display basic statistics
        print("\nBasic Statistics:")
        print(df.describe())

        # Save to CSV for further analysis
        df.to_csv("chess_quality_metrics.csv", index=False)
        print("Metrics saved to chess_quality_metrics.csv")

        # Create histograms with fixed percentage calculation
        print("\nCreating histograms with fixed percentages...")
        visualize_histograms(df, bins=[0, 0.25, 0.5, 0.8, 0.9, 1.0])

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
