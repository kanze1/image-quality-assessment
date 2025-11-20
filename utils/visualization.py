"""
Visualization utilities for evaluation results.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_scatter(predictions, targets, save_path=None, title='Predictions vs Targets'):
    """
    Plot scatter plot of predictions vs targets with regression line.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(targets, predictions, alpha=0.5, s=30)
    
    # Regression line
    z = np.polyfit(targets, predictions, 1)
    p = np.poly1d(z)
    x_line = np.linspace(targets.min(), targets.max(), 100)
    plt.plot(x_line, p(x_line), "r--", linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Ideal line
    plt.plot(x_line, x_line, 'k-', linewidth=1, alpha=0.5, label='Ideal (y=x)')
    
    # Calculate metrics
    plcc, _ = stats.pearsonr(predictions, targets)
    srcc, _ = stats.spearmanr(predictions, targets)
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    plt.xlabel('Ground Truth', fontsize=12)
    plt.ylabel('Predictions', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add metrics text
    textstr = f'PLCC: {plcc:.4f}\nSRCC: {srcc:.4f}\nRMSE: {rmse:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_distribution(predictions, targets, save_path=None):
    """
    Plot error distribution histogram.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        save_path: Path to save the plot
    """
    errors = predictions - targets
    
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    
    # Add vertical line at 0
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    
    # Add mean and std
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.axvline(x=mean_error, color='green', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_error:.3f}')
    
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Error Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    textstr = f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}\nMedian: {np.median(errors):.4f}'
    plt.text(0.75, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(train_losses, val_losses, quality_plccs, identity_plccs, save_path=None):
    """
    Plot training curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        quality_plccs: List of quality PLCC values
        identity_plccs: List of identity PLCC values
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PLCC curves
    axes[1].plot(epochs, quality_plccs, 'g-', label='Quality PLCC', linewidth=2)
    axes[1].plot(epochs, identity_plccs, 'm-', label='Identity PLCC', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('PLCC', fontsize=12)
    axes[1].set_title('Validation PLCC', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
