#!/usr/bin/env python3
"""
TransPolymer Prediction Function
==============================

This script provides functions to predict polymer properties using the pretrained TransPolymer model.
It loads the pretrained model and tokenizer, and provides a simple interface for making predictions.

Usage:
    from transpolymer.core.predict import PolymerPredictor
    predictor = PolymerPredictor("ckpt/PE_I_best_model.pt")
    prediction = predictor.predict("CC[N+](C)(C)CC.O=S(=O)(F)[N-]S(=O)(=O)F")
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import yaml
import argparse
from pathlib import Path
from transformers import RobertaModel, RobertaConfig
from .PolymerSmilesTokenization import PolymerSmilesTokenizer
from .dataset import Downstream_Dataset
from ..utils import PathManager, load_config
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DownstreamRegression(nn.Module):
    """Regression model for downstream polymer property prediction"""
    
    def __init__(self, pretrained_model, drop_rate=0.1):
        super(DownstreamRegression, self).__init__()
        self.PretrainedModel = pretrained_model
        self.Regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.PretrainedModel.config.hidden_size, self.PretrainedModel.config.hidden_size),
            nn.SiLU(),
            nn.Linear(self.PretrainedModel.config.hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.PretrainedModel(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:, 0, :]  # CLS token representation
        output = self.Regressor(logits)
        return output


class PolymerPredictor:
    """Main class for polymer property prediction using TransPolymer"""
    
    def __init__(self, model_path, config_path="config_finetune.yaml", device=None):
        """
        Initialize the polymer predictor
        
        Args:
            model_path (str): Path to the trained model checkpoint (relative to package or absolute)
            config_path (str): Path to the configuration file (relative to package or absolute)
            device (str): Device to run on ('cuda' or 'cpu'). Auto-detects if None.
        """
        # Resolve paths using PathManager
        self.model_path = str(PathManager.resolve_path(model_path))
        self.config_path = str(PathManager.resolve_path(config_path))
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load configuration using ConfigManager
        self.config = load_config(self.config_path)
        
        # Load tokenizer
        self._load_tokenizer()
        
        # Load model
        self._load_model()
        
        # Load scaler if available
        self._load_scaler()
    
    def _load_tokenizer(self):
        """Load the polymer SMILES tokenizer"""
        print("Loading tokenizer...")
        
        # Resolve model path from config
        model_ckpt_path = str(PathManager.resolve_path(self.config['model_path']))
        
        # Load pretrained model for tokenizer
        self.pretrained_model = RobertaModel.from_pretrained(model_ckpt_path)
        
        # Initialize tokenizer
        self.tokenizer = PolymerSmilesTokenizer.from_pretrained(
            "roberta-base", 
            max_len=self.config['blocksize']
        )
        
        # Add supplementary vocabulary if specified
        if self.config.get('add_vocab_flag', False):
            vocab_sup_file = self.config.get('vocab_sup_file')
            if vocab_sup_file:
                vocab_sup_file = str(PathManager.resolve_path(vocab_sup_file))
                if Path(vocab_sup_file).exists():
                    print(f"Adding supplementary vocabulary from {vocab_sup_file}")
                    vocab_sup = pd.read_csv(vocab_sup_file, header=None).values.flatten().tolist()
                    self.tokenizer.add_tokens(vocab_sup)
                    self.pretrained_model.resize_token_embeddings(len(self.tokenizer))
        
        print(f"Tokenizer loaded with vocab size: {len(self.tokenizer)}")
    
    def _load_model(self):
        """Load the trained regression model"""
        print(f"Loading model from {self.model_path}...")
        
        # Create the regression model
        self.model = DownstreamRegression(
            pretrained_model=self.pretrained_model,
            drop_rate=self.config.get('drop_rate', 0.1)
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.double()
        self.model.eval()
        
        print("Model loaded successfully")
    
    def _load_scaler(self):
        """Load the scaler used for normalization"""
        scaler_path = Path(self.model_path).with_suffix(".scaler.pkl")
        if scaler_path.exists():
            print(f"Loading scaler from {scaler_path}")
            self.scaler = joblib.load(scaler_path)
        else:
            print("No scaler found. Predictions will be in normalized space.")
            self.scaler = None
    
    def predict(self, smiles_sequence, return_confidence=False):
        """
        Predict polymer property for a given SMILES sequence
        
        Args:
            smiles_sequence (str): SMILES representation of the polymer
            return_confidence (bool): Whether to return additional model outputs
            
        Returns:
            float or dict: Predicted property value (and confidence if requested)
        """
        # Tokenize the input
        encoding = self.tokenizer(
            str(smiles_sequence),
            add_special_tokens=True,
            max_length=self.config['blocksize'],
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_ids, attention_mask)
            prediction = prediction.cpu().numpy()
        
        # Inverse transform if scaler is available
        if self.scaler is not None:
            prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))
            prediction = prediction.flatten()[0]
        else:
            prediction = prediction.flatten()[0]
        
        if return_confidence:
            return {
                'prediction': prediction,
                'raw_output': prediction,
                'smiles': smiles_sequence
            }
        else:
            return prediction
    
    def predict_batch(self, smiles_sequences):
        """
        Predict polymer properties for multiple SMILES sequences
        
        Args:
            smiles_sequences (list): List of SMILES representations
            
        Returns:
            list: List of predicted property values
        """
        predictions = []
        for smiles in smiles_sequences:
            pred = self.predict(smiles)
            predictions.append(pred)
        return predictions
    
    def get_model_info(self):
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'vocab_size': len(self.tokenizer),
            'max_length': self.config['blocksize'],
            'has_scaler': self.scaler is not None,
            'config': self.config
        }


def main():
    """Command line interface for prediction"""
    parser = argparse.ArgumentParser(description='TransPolymer Property Prediction')
    parser.add_argument('--smiles', type=str, required=True, 
                       help='SMILES sequence to predict')
    parser.add_argument('--model_path', type=str, default='ckpt/PE_I_best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='transpolymer/configs/config_finetune.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = PolymerPredictor(
            model_path=args.model_path,
            config_path=args.config,
            device=args.device
        )
        
        if args.verbose:
            print("Model Information:")
            info = predictor.get_model_info()
            for key, value in info.items():
                if key != 'config':
                    print(f"  {key}: {value}")
            print()
        
        # Make prediction
        print(f"Predicting property for SMILES: {args.smiles}")
        prediction = predictor.predict(args.smiles, return_confidence=True)
        
        if isinstance(prediction, dict):
            print(f"Predicted property: {prediction['prediction']:.6f}")
            if predictor.scaler is not None:
                print("(Scaled back to original units)")
            else:
                print("(In normalized units)")
        else:
            print(f"Predicted property: {prediction:.6f}")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
