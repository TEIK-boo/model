import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import argparse

def compare_results(experiment_name):
    """
    Load results from an experiment folder and generate comparison plots
    """
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    checkpoint_dir = base_dir / "src" / "checkpoints" / experiment_name
    
    if not checkpoint_dir.exists():
        print(f"Experiment directory not found: {checkpoint_dir}")
        return
    
    results = []
    
    # Iterate through model folders
    for model_dir in checkpoint_dir.iterdir():
        if model_dir.is_dir():
            result_file = model_dir / "results.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    results.append(data)
    
    if not results:
        print("No results found.")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('test_accuracy', ascending=False)
    
    print("\nModel Comparison:")
    print(df[['model_name', 'test_accuracy', 'best_val_accuracy', 'epochs_trained']])
    
    # Save CSV
    df.to_csv(checkpoint_dir / "summary.csv", index=False)
    
    # Plot Accuracy
    plt.figure(figsize=(12, 8))
    for res in results:
        plt.plot(res['history']['val_accuracy'], label=f"{res['model_name']} (Best: {res['best_val_accuracy']:.4f})")
    
    plt.title(f'Validation Accuracy Comparison - {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(checkpoint_dir / "accuracy_comparison.png")
    print(f"Saved plot to {checkpoint_dir / 'accuracy_comparison.png'}")
    
    # Plot Loss
    plt.figure(figsize=(12, 8))
    for res in results:
        plt.plot(res['history']['val_loss'], label=f"{res['model_name']}")
        
    plt.title(f'Validation Loss Comparison - {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(checkpoint_dir / "loss_comparison.png")
    print(f"Saved plot to {checkpoint_dir / 'loss_comparison.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_name', type=str, help='Name of the experiment folder in checkpoints')
    args = parser.parse_args()
    
    compare_results(args.experiment_name)
