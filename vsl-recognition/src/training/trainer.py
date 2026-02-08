"""
Main training script
"""
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from pathlib import Path
import sys
import json
import os

# Handle path for running from root or src
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.append(str(Path(__file__).parent.parent))

from training.data_loader import load_sequences, split_data, create_tf_dataset
from config import TRAINING_CONFIG, CHECKPOINT_DIR, LOGS_DIR
from models.model_registry import get_model

def configure_gpu():
    """Configure GPU settings for TensorFlow"""
    print("\n" + "="*70)
    print("GPU CONFIGURATION")
    print("="*70)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"Using GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU detected - training will use CPU")

def train_model(model_name='Baseline_LSTM', epochs=None, batch_size=None, experiment_name='default'):
    """
    Main training function
    
    Args:
        model_name: Name of the model to train (from model_registry)
        epochs: Override config epochs
        batch_size: Override config batch_size
        experiment_name: Name for the experiment (affects save paths)
    """
    
    # Configure GPU
    configure_gpu()
    
    # Use config values if overrides not provided
    if epochs is None:
        epochs = TRAINING_CONFIG['epochs']
    if batch_size is None:
        batch_size = TRAINING_CONFIG['batch_size']
        
    # Setup paths for this specific experiment
    exp_checkpoint_dir = CHECKPOINT_DIR / experiment_name / model_name
    exp_logs_dir = LOGS_DIR / experiment_name / model_name
    
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    exp_logs_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(f"TRAINING: {model_name}")
    print("="*70)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    X, y, action_names = load_sequences()
    num_classes = len(action_names)
    
    # 2. Split data
    print("\n[2/5] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_size=TRAINING_CONFIG['train_split'],
        val_size=TRAINING_CONFIG['val_split']
    )
    
    # 3. Create datasets
    print("\n[3/5] Creating datasets...")
    train_ds = create_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = create_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_ds = create_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)
    
    # 4. Build model
    print("\n[4/5] Building model...")
    sequence_length = X_train.shape[1]
    
    model = get_model(model_name, num_classes=num_classes, sequence_length=sequence_length)
    
    model.compile(
        optimizer=Adam(learning_rate=TRAINING_CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # 5. Setup callbacks
    print("\n[5/5] Setting up training...")
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(exp_checkpoint_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=TRAINING_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=TRAINING_CONFIG['reduce_lr_patience'],
            verbose=1
        ),
        TensorBoard(
            log_dir=str(exp_logs_dir),
            histogram_freq=1
        )
    ]
    
    # 6. Train
    print("\nSTARTING TRAINING")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # 7. Evaluate
    print("\nEVALUATING")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    # 8. Save final results
    model.save(str(exp_checkpoint_dir / 'final_model.h5'))
    
    # Save history
    history_dict = history.history
    # Convert float32 to float for JSON serialization
    for key in history_dict:
        history_dict[key] = [float(x) for x in history_dict[key]]
        
    results = {
        'model_name': model_name,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'history': history_dict,
        'epochs_trained': len(history_dict['loss']),
        'best_val_accuracy': float(max(history_dict['val_accuracy']))
    }
    
    with open(exp_checkpoint_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Save action mapping
    mapping = {i: name for i, name in enumerate(action_names)}
    with open(exp_checkpoint_dir / 'action_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
        
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Baseline_LSTM', help='Model to train')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--name', type=str, default='manual_run', help='Experiment name')
    
    args = parser.parse_args()
    
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        experiment_name=args.name
    )
