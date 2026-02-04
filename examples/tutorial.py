#!/usr/bin/env python3
"""
Example script demonstrating TransPolymer usage.

This script shows various ways to use TransPolymer for polymer property prediction.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transpolymer.core import PolymerPredictor
from transpolymer.utils import PathManager, load_config
import pandas as pd


def example_1_basic_prediction():
    """Example 1: Basic single prediction"""
    print("=" * 60)
    print("Example 1: Basic Single Prediction")
    print("=" * 60)
    
    # Initialize predictor with default paths
    predictor = PolymerPredictor(
        model_path="ckpt/PE_I_best_model.pt",
        config_path="transpolymer/configs/config_finetune.yaml"
    )
    
    # Make a prediction
    smiles = "CC[N+](C)(C)CC.O=S(=O)(F)[N-]S(=O)(=O)F"
    prediction = predictor.predict(smiles)
    
    print(f"SMILES: {smiles}")
    print(f"Predicted property value: {prediction:.4f}")
    print()


def example_2_batch_prediction():
    """Example 2: Batch prediction"""
    print("=" * 60)
    print("Example 2: Batch Prediction")
    print("=" * 60)
    
    predictor = PolymerPredictor(
        model_path="ckpt/PE_I_best_model.pt",
        config_path="transpolymer/configs/config_finetune.yaml"
    )
    
    # List of polymers to predict
    smiles_list = [
        "C=C",
        "CC",
        "CCC",
        "CC[N+](C)(C)CC.O=S(=O)(F)[N-]S(=O)(=O)F",
    ]
    
    predictions = predictor.predict_batch(smiles_list)
    
    print("Batch Predictions:")
    print("-" * 60)
    for smiles, pred in zip(smiles_list, predictions):
        print(f"{smiles:40} → {pred:.4f}")
    print()


def example_3_model_info():
    """Example 3: Get model information"""
    print("=" * 60)
    print("Example 3: Model Information")
    print("=" * 60)
    
    predictor = PolymerPredictor(
        model_path="ckpt/PE_I_best_model.pt",
        config_path="transpolymer/configs/config_finetune.yaml"
    )
    
    info = predictor.get_model_info()
    
    print("Model Information:")
    print("-" * 60)
    for key, value in info.items():
        if key != 'config':
            if isinstance(value, bool):
                print(f"{key:30}: {str(value)}")
            else:
                print(f"{key:30}: {value}")
    print()


def example_4_path_management():
    """Example 4: Path management utilities"""
    print("=" * 60)
    print("Example 4: Path Management")
    print("=" * 60)
    
    print("Package Root:", PathManager.get_package_root())
    print("Checkpoint Dir:", PathManager.get_checkpoint_dir())
    print("Config Dir:", PathManager.get_config_dir())
    print("Data Dir:", PathManager.get_data_dir())
    print()
    
    # Resolve relative paths
    model_path = PathManager.resolve_path("ckpt/PE_I_best_model.pt")
    config_path = PathManager.resolve_path("config_finetune.yaml", relative_to="config")
    
    print(f"Resolved model path: {model_path}")
    print(f"Resolved config path: {config_path}")
    print()


def example_5_config_loading():
    """Example 5: Configuration loading with automatic path resolution"""
    print("=" * 60)
    print("Example 5: Configuration Loading")
    print("=" * 60)
    
    config = load_config("transpolymer/configs/config_finetune.yaml")
    
    print("Configuration parameters:")
    print("-" * 60)
    for key, value in config.items():
        if key != 'config':
            if isinstance(value, str) and len(value) > 50:
                print(f"{key:30}: {value[:50]}...")
            else:
                print(f"{key:30}: {value}")
    print()


def example_6_csv_batch():
    """Example 6: Batch processing from CSV"""
    print("=" * 60)
    print("Example 6: Batch Processing from CSV")
    print("=" * 60)
    
    predictor = PolymerPredictor(
        model_path="ckpt/PE_I_best_model.pt",
        config_path="transpolymer/configs/config_finetune.yaml"
    )
    
    # Create sample data
    sample_data = pd.DataFrame({
        'smiles': [
            'C=C',
            'CC',
            'CCC',
            'CCCC',
            'CC[N+](C)(C)CC.O=S(=O)(F)[N-]S(=O)(=O)F',
        ],
        'name': ['ethene', 'ethane', 'propane', 'butane', 'polymer1']
    })
    
    # Make predictions
    print("Sample data:")
    print(sample_data)
    print("\nMaking predictions...")
    
    predictions = []
    for smiles in sample_data['smiles']:
        pred = predictor.predict(smiles)
        predictions.append(pred)
    
    sample_data['predicted_property'] = predictions
    
    print("\nResults with predictions:")
    print(sample_data)
    print()
    
    # Save to CSV (commented out to avoid file creation)
    # sample_data.to_csv('example_predictions.csv', index=False)
    # print("Results saved to: example_predictions.csv")


def example_7_custom_paths():
    """Example 7: Using custom paths"""
    print("=" * 60)
    print("Example 7: Custom Paths")
    print("=" * 60)
    
    # You can use absolute paths or paths relative to package root
    predictor = PolymerPredictor(
        model_path="ckpt/PE_I_best_model.pt",  # Relative to package root
        config_path="transpolymer/configs/config_finetune.yaml"  # Relative to package root
    )
    
    smiles = "C=C"
    prediction = predictor.predict(smiles)
    
    print(f"SMILES: {smiles}")
    print(f"Predicted value: {prediction:.4f}")
    print(f"Model used: {predictor.model_path}")
    print(f"Config used: {predictor.config_path}")
    print()


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "   TransPolymer Usage Examples".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    examples = [
        ("Basic Prediction", example_1_basic_prediction),
        ("Batch Prediction", example_2_batch_prediction),
        ("Model Information", example_3_model_info),
        ("Path Management", example_4_path_management),
        ("Configuration Loading", example_5_config_loading),
        ("CSV Batch Processing", example_6_csv_batch),
        ("Custom Paths", example_7_custom_paths),
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples) + 1}. Run all examples")
    print(f"  0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-{}): ".format(len(examples) + 1))
            choice = int(choice)
            
            if choice == 0:
                print("Goodbye!")
                break
            elif 1 <= choice <= len(examples):
                name, func = examples[choice - 1]
                try:
                    func()
                except Exception as e:
                    print(f"Error running example: {e}")
                    import traceback
                    traceback.print_exc()
            elif choice == len(examples) + 1:
                for name, func in examples:
                    try:
                        func()
                    except Exception as e:
                        print(f"Error in {name}: {e}")
                        import traceback
                        traceback.print_exc()
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
