# main.py - TransPolymer Prediction Interface

This document explains how to use the `main.py` file with the `predict.py` module for TransPolymer predictions.

## Overview

The `main.py` file provides a comprehensive command-line interface for using the TransPolymer prediction functionality. It integrates with the `predict.py` module to offer multiple ways to make predictions:

- Single predictions
- Batch processing
- Interactive mode
- MCP (Model Context Protocol) integration

## Files Created

- `main.py` - Main command-line interface
- `mcp_integration.py` - MCP server integration
- `example_usage.py` - Usage examples
- `test_main.py` - Test script
- `MAIN_PY_README.md` - This documentation

## Quick Start

### 1. Single Prediction

```bash
# Basic prediction
python main.py --smiles "CC[N+](C)(C)CC.O=S(=O)(F)[N-]S(=O)(=O)F$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_Mw$NAN_Mn$1.0$1$S_5$NAN_Tm$NAN_Tg"

# With verbose output
python main.py --smiles "your_smiles_here" --verbose

# JSON output
python main.py --smiles "your_smiles_here" --json
```

### 2. Interactive Mode

```bash
python main.py --interactive
```

### 3. Batch Processing

```bash
# Process CSV file
python main.py --batch input.csv --output results.csv

# With custom column name
python main.py --batch input.csv --smiles_column "polymer_smiles"
```

### 4. MCP Integration

```bash
python mcp_integration.py
```

## Command Line Options

### Model Options
- `--model_path`: Path to trained model checkpoint (default: `ckpt/PE_I_best_model.pt`)
- `--config`: Path to configuration file (default: `config_finetune.yaml`)
- `--device`: Device to use (`cuda` or `cpu`, auto-detects if not specified)

### Input Options
- `--smiles`: SMILES sequence to predict
- `--property`: Property name for prediction (default: `conductivity`)
- `--interactive`: Run in interactive mode
- `--batch`: CSV file with SMILES sequences
- `--output`: Output file for batch predictions
- `--smiles_column`: Name of SMILES column in batch file (default: `smiles`)

### Output Options
- `--json`: Output results in JSON format
- `--verbose`: Show detailed output

## Usage Examples

### Example 1: Single Prediction

```bash
python main.py --smiles "CC[N+](C)(C)CC.O=S(=O)(F)[N-]S(=O)(=O)F$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_Mw$NAN_Mn$1.0$1$S_5$NAN_Tm$NAN_Tg" --property conductivity --verbose
```

**Output:**
```
Loading TransPolymer model from ckpt/PE_I_best_model.pt...
✅ Model loaded successfully!

Model Information:
  model_path: ckpt/PE_I_best_model.pt
  device: cuda
  vocab_size: 50265
  max_length: 411
  has_scaler: True

Predicted conductivity: -3.130768
```

### Example 2: JSON Output

```bash
python main.py --smiles "your_smiles_here" --json
```

**Output:**
```json
{
  "smiles": "your_smiles_here",
  "property": "conductivity",
  "prediction": -3.130768,
  "status": "success"
}
```

### Example 3: Interactive Mode

```bash
python main.py --interactive
```

**Interactive Session:**
```
TransPolymer Interactive Mode
========================================
Enter SMILES sequences (or 'quit' to exit):
Available commands:
  - Enter SMILES to predict
  - 'info' - Show model information
  - 'quit' - Exit
----------------------------------------

> CC[N+](C)(C)CC.O=S(=O)(F)[N-]S(=O)(=O)F$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_Mw$NAN_Mn$1.0$1$S_5$NAN_Tm$NAN_Tg
Predicted conductivity: -3.130768

> info
Model Information:
  model_path: ckpt/PE_I_best_model.pt
  device: cuda
  vocab_size: 50265
  max_length: 411
  has_scaler: True

> quit
Goodbye!
```

### Example 4: Batch Processing

**Input CSV (`input.csv`):**
```csv
smiles,id
CC[N+](C)(C)CC.O=S(=O)(F)[N-]S(=O)(=O)F$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_Mw$NAN_Mn$1.0$1$S_5$NAN_Tm$NAN_Tg,polymer1
CC[N+](C)(C)CC.O=S(=O)(F)[N-]S(=O)(=O)F$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_Mw$NAN_Mn$1.0$1$S_5$NAN_Tm$NAN_Tg,polymer2
```

**Command:**
```bash
python main.py --batch input.csv --output results.csv
```

**Output CSV (`results.csv`):**
```csv
smiles,id,conductivity_predicted
CC[N+](C)(C)CC.O=S(=O)(F)[N-]S(=O)(=O)F$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_Mw$NAN_Mn$1.0$1$S_5$NAN_Tm$NAN_Tg,polymer1,-3.130768
CC[N+](C)(C)CC.O=S(=O)(F)[N-]S(=O)(=O)F$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_ratio$NAN_Mw$NAN_Mn$1.0$1$S_5$NAN_Tm$NAN_Tg,polymer2,-3.130768
```

## MCP Integration

The `mcp_integration.py` file provides MCP (Model Context Protocol) integration:

### Available MCP Tools

1. **`predict_property`**: Predict a single property
   - Parameters: `smiles`, `property_name` (optional)
   - Returns: Prediction result string

2. **`predict_batch`**: Batch prediction
   - Parameters: `smiles_list`, `property_name` (optional)
   - Returns: Batch prediction results

3. **`get_model_info`**: Get model information
   - Returns: Model details and configuration

### MCP Usage

```bash
# Start MCP server
python mcp_integration.py
```

The MCP server will start and be ready to accept tool calls from MCP clients.

## Programmatic Usage

You can also use the `main.py` functionality programmatically:

```python
from main import TransPolymerInterface

# Initialize interface
interface = TransPolymerInterface(
    model_path="ckpt/PE_I_best_model.pt",
    config_path="config_finetune.yaml"
)

# Load model
interface.load_model()

# Single prediction
result = interface.predict_single("your_smiles_here", "conductivity")
print(f"Prediction: {result['prediction']}")

# Batch prediction
smiles_list = ["smiles1", "smiles2", "smiles3"]
results = interface.predict_batch(smiles_list, "conductivity")
for result in results:
    print(f"Prediction: {result['prediction']}")
```

## Error Handling

The interface includes comprehensive error handling:

- **Model loading errors**: Clear messages if model files are missing
- **Input validation**: Checks for valid SMILES sequences
- **Device errors**: Automatic fallback to CPU if GPU unavailable
- **File errors**: Validation of input/output files

## Performance Tips

1. **GPU Usage**: Automatically uses GPU if available
2. **Batch Processing**: Use `--batch` for multiple sequences
3. **Memory Management**: Large datasets are processed efficiently
4. **Caching**: Model is loaded once and reused for multiple predictions

## Troubleshooting

### Common Issues

1. **"Model not loaded" error**
   - Ensure model checkpoint exists at specified path
   - Check that all required files are present

2. **CUDA out of memory**
   - Use `--device cpu` to force CPU usage
   - Reduce batch size for large datasets

3. **Import errors**
   - Install required dependencies: `pip install torch transformers scikit-learn pandas numpy pyyaml joblib`
   - Check Python path and module locations

4. **Input format errors**
   - Ensure SMILES sequences follow the expected format
   - Check that sequences are not too long

### Debug Mode

Use `--verbose` flag for detailed output:

```bash
python main.py --smiles "your_smiles" --verbose
```

This will show:
- Model loading progress
- Device information
- Detailed error messages
- Model configuration

## Integration with Existing Code

The `main.py` interface is designed to work alongside the existing TransPolymer codebase:

1. **Training**: Use `Downstream.py` to train models
2. **Prediction**: Use `main.py` to make predictions
3. **Visualization**: Use `Attention_vis.py` for model interpretability
4. **MCP**: Use `mcp_integration.py` for MCP server integration

## File Structure

```
TransPolymer_pretrained/
├── main.py                 # Main command-line interface
├── predict.py              # Core prediction module
├── mcp_integration.py      # MCP server integration
├── example_usage.py        # Usage examples
├── test_main.py           # Test script
├── ckpt/                   # Model checkpoints
│   ├── PE_I_best_model.pt
│   └── PE_I_best_model.scaler.pkl
├── config_finetune.yaml   # Configuration
└── data/                   # Training data
```

## Next Steps

1. **Install Dependencies**: Ensure all required packages are installed
2. **Test Installation**: Run `python test_main.py` to verify setup
3. **Try Examples**: Use `python example_usage.py` to see usage patterns
4. **Start Predicting**: Use `python main.py --smiles "your_smiles"` for predictions
5. **MCP Integration**: Use `python mcp_integration.py` for MCP server

The `main.py` interface provides a complete solution for using TransPolymer predictions in various scenarios, from simple command-line usage to complex MCP integrations.


