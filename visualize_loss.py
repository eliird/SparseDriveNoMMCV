#!/usr/bin/env python3
"""
Script to visualize training loss from out.log
"""
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_loss_from_log(log_file):
    """Parse total loss values from log file"""
    losses = []
    iterations = []

    with open(log_file, 'r') as f:
        current_iter = 0
        for line in f:
            # Extract iteration number
            iter_match = re.search(r'\[Iter (\d+)/\d+\]', line)
            if iter_match:
                current_iter = int(iter_match.group(1))

            # Extract total loss
            loss_match = re.search(r'^Total Loss:\s+([\d.]+)', line)
            if loss_match:
                loss = float(loss_match.group(1))
                losses.append(loss)
                iterations.append(current_iter)

    return iterations, losses

def plot_loss(iterations, losses, save_path='loss_plot.png'):
    """Create and save loss visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Raw loss over iterations
    ax1.plot(iterations, losses, linewidth=0.8, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Total Loss', fontsize=12)
    ax1.set_title('Training Loss Over Iterations (Raw)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(losses) * 0.9, max(losses) * 1.1])

    # Add statistics to plot 1
    stats_text = f'Min: {min(losses):.2f}\nMax: {max(losses):.2f}\nMean: {np.mean(losses):.2f}\nStd: {np.std(losses):.2f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Smoothed loss with moving average
    window_size = min(20, len(losses) // 10) if len(losses) > 20 else 5
    if len(losses) >= window_size:
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        smoothed_iters = iterations[window_size-1:]

        ax2.plot(iterations, losses, linewidth=0.5, alpha=0.3, color='lightblue', label='Raw')
        ax2.plot(smoothed_iters, smoothed_losses, linewidth=2, color='darkred', label=f'Moving Avg (window={window_size})')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Total Loss', fontsize=12)
        ax2.set_title('Training Loss with Moving Average', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2.set_ylim([min(losses) * 0.9, max(losses) * 1.1])
    else:
        # If not enough data for smoothing, just plot raw
        ax2.plot(iterations, losses, linewidth=0.8, color='steelblue')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Total Loss', fontsize=12)
        ax2.set_title('Training Loss (Not enough data for smoothing)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Loss visualization saved to: {save_path}")

    # Print summary statistics
    print("\n" + "="*60)
    print("LOSS STATISTICS")
    print("="*60)
    print(f"Total iterations: {len(losses)}")
    print(f"Min loss: {min(losses):.4f} (iter {iterations[losses.index(min(losses))]})")
    print(f"Max loss: {max(losses):.4f} (iter {iterations[losses.index(max(losses))]})")
    print(f"Mean loss: {np.mean(losses):.4f}")
    print(f"Std dev: {np.std(losses):.4f}")
    print(f"First loss: {losses[0]:.4f}")
    print(f"Last loss: {losses[-1]:.4f}")
    print(f"Change: {losses[-1] - losses[0]:.4f} ({((losses[-1] - losses[0])/losses[0])*100:.2f}%)")
    print("="*60)

def main():
    log_file = 'out.log'
    output_file = 'loss_visualization.png'

    print(f"Parsing loss values from {log_file}...")
    iterations, losses = parse_loss_from_log(log_file)

    if not losses:
        print("No loss values found in log file!")
        return

    print(f"Found {len(losses)} loss values")
    plot_loss(iterations, losses, output_file)

if __name__ == '__main__':
    main()
