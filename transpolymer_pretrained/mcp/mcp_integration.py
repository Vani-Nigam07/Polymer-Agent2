#!/usr/bin/env python3
"""
MCP Integration for TransPolymer
===============================

This file provides MCP (Model Context Protocol) integration for TransPolymer predictions.
It can be used with your existing MCP setup.

Usage:
    python mcp_integration.py
"""

from mcp.server.fastmcp import FastMCP, Context
import sys
import os
import traceback
from pathlib import Path

# Add the current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our prediction module
from predict import PolymerPredictor

# Initialize MCP server
mcp = FastMCP("TransPolymer_mcp")

# Global variables for model and predictor
predictor = None

@mcp.tool()
async def predict_property(smiles: str, ctx: Context, property_name: str = "conductivity") -> str:
    """
    Predicts a specific physical or chemical property for a given polymer SMILES string.

    Args:
        smiles: The SMILES string representation of the polymer.
        property_name: The target property to predict (e.g., 'conductivity', 'bandgap').
    
    Returns:
        A string containing the predicted value for the property.
    """
    global predictor
    
    await ctx.info(f"Predicting {property_name} for {smiles}")
    
    try:
        if predictor is None:
            return "Error: Model not loaded. Please ensure the model is properly initialized."
        
        # Make prediction using our predictor
        prediction = predictor.predict(smiles)
        
        return f"The predicted {property_name} is: {prediction:.4f}"
        
    except Exception as e:
        error_msg = f"Error predicting {property_name}: {str(e)}"
        await ctx.error(error_msg)
        return error_msg

@mcp.tool()
async def predict_batch(smiles_list: list, ctx: Context, property_name: str = "conductivity") -> str:
    """
    Predicts properties for multiple polymer SMILES strings in batch.

    Args:
        smiles_list: List of SMILES string representations of polymers.
        property_name: The target property to predict.
    
    Returns:
        A string containing the predicted values for all polymers.
    """
    global predictor
    
    await ctx.info(f"Predicting {property_name} for {len(smiles_list)} polymers")
    
    try:
        if predictor is None:
            return "Error: Model not loaded. Please ensure the model is properly initialized."
        
        # Make batch predictions
        predictions = predictor.predict_batch(smiles_list)
        
        # Format results
        results = []
        for i, (smiles, pred) in enumerate(zip(smiles_list, predictions)):
            results.append(f"Polymer {i+1}: {pred:.4f}")
        
        return f"Batch predictions for {property_name}:\n" + "\n".join(results)
        
    except Exception as e:
        error_msg = f"Error in batch prediction: {str(e)}"
        await ctx.error(error_msg)
        return error_msg

@mcp.tool()
async def get_model_info(ctx: Context) -> str:
    """
    Get information about the loaded TransPolymer model.
    
    Returns:
        A string containing model information.
    """
    global predictor
    
    try:
        if predictor is None:
            return "Error: Model not loaded."
        
        info = predictor.get_model_info()
        
        info_str = "TransPolymer Model Information:\n"
        for key, value in info.items():
            if key != 'config':
                info_str += f"  {key}: {value}\n"
        
        return info_str
        
    except Exception as e:
        return f"Error getting model info: {str(e)}"

def initialize_model(model_path: str = "ckpt/PE_I_best_model.pt", 
                    config_path: str = "config_finetune.yaml",
                    device: str = None):
    """
    Initialize the TransPolymer model for MCP use.
    
    Args:
        model_path: Path to the trained model checkpoint
        config_path: Path to the configuration file
        device: Device to use ('cuda' or 'cpu')
    """
    global predictor
    
    try:
        print("Initializing TransPolymer model for MCP...")
        predictor = PolymerPredictor(
            model_path=model_path,
            config_path=config_path,
            device=device
        )
        print("✅ TransPolymer model loaded successfully for MCP!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading TransPolymer model: {e}")
        traceback.print_exc()
        return False

# Initialize model on startup
if __name__ == "__main__":
    # Default paths - adjust these to match your setup
    model_path = "ckpt/PE_I_best_model.pt"
    config_path = "config_finetune.yaml"
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the model checkpoint exists.")
        sys.exit(1)
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        print("Please ensure the configuration file exists.")
        sys.exit(1)
    
    # Initialize model
    if initialize_model(model_path, config_path):
        print("🚀 Starting TransPolymer MCP server...")
        mcp.run(transport="stdio")
    else:
        print("❌ Failed to initialize model. Exiting.")
        sys.exit(1)
