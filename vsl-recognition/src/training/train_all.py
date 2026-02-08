import sys
from pathlib import Path
import datetime
import traceback

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from training.trainer import train_model
from models.model_registry import list_models

def run_all_experiments(epochs=5, quick_test=False):
    """
    Run training for all available models
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"comparison_{timestamp}"
    
    models = list_models()
    results = {}
    
    print(f"Starting experiment: {experiment_name}")
    print(f"Models to train: {models}")
    
    if quick_test:
        print("QUICK TEST MODE: 1 epoch only")
        epochs = 1
    
    for model_name in models:
        print(f"\n\n{'#'*80}")
        print(f"Processing Model: {model_name}")
        print(f"{'#'*80}")
        
        try:
            res = train_model(
                model_name=model_name,
                epochs=epochs,
                experiment_name=experiment_name
            )
            results[model_name] = res
            print(f"SUCCESS: {model_name} - Test Acc: {res['test_accuracy']:.4f}")
            
        except Exception as e:
            print(f"FAILED: {model_name}")
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
            
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"{'Model':<30} | {'Test Acc':<10} | {'Status'}")
    print("-" * 55)
    
    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<30} | {'N/A':<10} | FAILED")
        else:
            print(f"{name:<30} | {res['test_accuracy']:.4f}     | OK")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run quick test (1 epoch)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    
    args = parser.parse_args()
    
    run_all_experiments(epochs=args.epochs, quick_test=args.quick)
