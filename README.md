# Setup Guide - Finetuning, MCP, and Custom Environments

## Part 1: Finetuning & MCP Setup

### ✅ What You Need ONLY from TransPolymer_pretrained/

For **finetuning** and **MCP (Model Context Protocol)**, you ONLY need:

```
TransPolymer/
├── transpolymer_pretrained/              
│   ├── core/
│   ├── utils/
│   │   ├── path_utils.py
│   │   └── config.py
│   ├── configs/
│       └── config_*.yaml
│
├── ckpt/                      
├── data/  
|   ├── publish_data\
│   └── Property datasets for train and test         
│
├── Downstream.py              
├── Pretrain.py                
├── Attention_vis.py           
├── tSNE.py                                     
├── requirements.txt           
├── pyproject.toml                 
   
```

### Key Point

**For finetuning and MCP, you only need:**
1. The `transpolymer/` package
2. Your data in `data/`
3. Model checkpoints in `ckpt/`
4. The training scripts: `Downstream.py`, `Pretrain.py`
5. Package files: `setup.py`, `requirements.txt`

Everything else is **optional documentation and examples**.

---

## Part 2: Setting Up Conda/UV Virtual Environments in Custom Locations

### Option 1: Conda in Custom Location

#### Create environment in custom directory
```bash
# Create conda env in /custom/path/my_transpolymer_env
conda create --prefix /custom/path/my_transpolymer_env python=3.9

# Activate it
conda activate /custom/path/my_transpolymer_env

# Or use the activation command shown by conda
source activate /custom/path/my_transpolymer_env  # Linux/Mac
```

#### Install dependencies
```bash
# Install from requirements
pip install -r requirements.txt

# Or install package in development mode
pip install -e TransPolymer_pretrained/
```

#### Make it convenient (optional)
Add to `~/.bashrc` or `~/.zshrc`:
```bash
alias activate_tp="conda activate /custom/path/my_transpolymer_env"
```

Then just run:
```bash
activate_tp
```

---

### Option 2: UV (Faster than Conda)

UV is a fast Python package installer. Here's how to set it up:

#### Install UV first
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with conda
conda install -c conda-forge uv
```

#### Create UV virtual environment in custom location
```bash
# Create venv with specific Python version
uv venv /custom/path/my_transpolymer_env --python 3.9

# Activate it
source /custom/path/my_transpolymer_env/bin/activate  # Linux/Mac
.\custom\path\my_transpolymer_env\Scripts\activate   # Windows
```

#### Install dependencies with UV (much faster!)
```bash
# Install from requirements
uv pip install -r requirements.txt

# Or install package
uv pip install -e TransPolymer_pretrained/
```

---

### Option 3: Python venv (Built-in, No Extra Tools)

#### Create venv in custom location
```bash
# Create venv
python3.9 -m venv /custom/path/my_transpolymer_env

# Activate it
source /custom/path/my_transpolymer_env/bin/activate  # Linux/Mac
.\custom\path\my_transpolymer_env\Scripts\activate   # Windows
```

#### Install dependencies
```bash
pip install -r requirements.txt
pip install -e TransPolymer_pretrained/
```

---

## Part 3: Complete Setup Workflow

### Scenario: Set up UV environment in `/opt/envs/transpolymer`

```bash
# Step 1: Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Step 2: Create venv in custom location
uv venv /opt/envs/transpolymer --python 3.9

# Step 3: Activate environment
source /opt/envs/transpolymer/bin/activate

# Step 4: Install dependencies (super fast with UV!)
cd /path/to/TransPolymer_pretrained
uv pip install -r requirements.txt

# Step 5: Install package in development mode
uv pip install -e .

# Step 6: Verify installation
python -c "from transpolymer.core import PolymerPredictor; print('✓ Ready')"

# Step 7: Run finetuning
python Downstream.py

# Step 8: Run MCP (in another terminal)
source /opt/envs/transpolymer/bin/activate
python main.py --interactive
```

---

## Part 4: Managing Multiple Environments

### Keep environments organized
```
/opt/envs/
├── transpolymer_dev/        ← Development (with extra tools)
├── transpolymer_prod/       ← Production (minimal)
└── transpolymer_jupyter/    ← Jupyter notebooks
```

### Create each one
```bash
# Development environment (with jupyter, debugging tools)
uv venv /opt/envs/transpolymer_dev --python 3.9
source /opt/envs/transpolymer_dev/bin/activate
uv pip install -r requirements.txt
uv pip install jupyter ipython black flake8

# Production environment (minimal)
uv venv /opt/envs/transpolymer_prod --python 3.9
source /opt/envs/transpolymer_prod/bin/activate
uv pip install -r requirements.txt

# Jupyter environment
uv venv /opt/envs/transpolymer_jupyter --python 3.9
source /opt/envs/transpolymer_jupyter/bin/activate
uv pip install -r requirements.txt jupyter
```

---

## Part 5: Quick Reference Commands

### UV Environment Setup (Recommended - Fastest)
```bash
# Create
uv venv /opt/envs/transpolymer --python 3.9

# Activate
source /opt/envs/transpolymer/bin/activate

# Install
uv pip install -r requirements.txt
uv pip install -e .

# Run finetuning
python Downstream.py

# Run MCP
python main.py --interactive

# Deactivate
deactivate
```

### Conda Environment Setup
```bash
# Create
conda create --prefix /opt/envs/transpolymer python=3.9

# Activate
conda activate /opt/envs/transpolymer

# Install
pip install -r requirements.txt
pip install -e .

# Run (same as UV)
python Downstream.py
```

### Standard Venv Setup
```bash
# Create
python3.9 -m venv /opt/envs/transpolymer

# Activate
source /opt/envs/transpolymer/bin/activate

# Install
pip install -r requirements.txt
pip install -e .

# Run (same as UV)
python Downstream.py
```

---

## Part 6: Environment Files for Easy Activation

### Create activation helper script

**File: `/opt/envs/activate_transpolymer.sh`**
```bash
#!/bin/bash
# Activate TransPolymer environment

VENV_PATH="/opt/envs/transpolymer"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ TransPolymer environment activated"
    echo "Location: $VENV_PATH"
    python --version
else
    echo "✗ Environment not found at $VENV_PATH"
fi
```

### Make it executable and use it
```bash
chmod +x /opt/envs/activate_transpolymer.sh

# Use it
source /opt/envs/activate_transpolymer.sh
```

### Or add to shell profile

**Add to `~/.bashrc` or `~/.zshrc`:**
```bash
# TransPolymer environments
alias activate_tp="source /opt/envs/transpolymer/bin/activate"
alias activate_tp_jupyter="source /opt/envs/transpolymer_jupyter/bin/activate"
```

Then just run:
```bash
activate_tp
```

---

## Part 7: Finetuning Quick Start

Once environment is set up:

```bash
# Activate environment
source /opt/envs/transpolymer/bin/activate

# Go to project directory
cd /path/to/TransPolymer_pretrained

# Update config with your data paths
# Edit: transpolymer/configs/config_finetune.yaml
# Update:
#   - train_file: path/to/your/train.csv
#   - test_file: path/to/your/test.csv
#   - model_path: ckpt  (path to pretrained model)

# Run finetuning
python Downstream.py

# Monitor with tensorboard
tensorboard --logdir=runs
```

---

## Part 8: MCP Integration Quick Start

```bash
# Activate environment
source /opt/envs/transpolymer/bin/activate

# Terminal 1: Start MCP server
cd /path/to/TransPolymer_pretrained
python main.py --interactive

# Terminal 2 (optional): Use Python API
python -c "
from main import TransPolymerMCP
mcp = TransPolymerMCP()
result = mcp.predict_property('C=C')
print(result)
"
```

---

## Summary

### Minimal Files Needed
✅ Only the `transpolymer/` package folder  
✅ Your data in `data/`  
✅ Model checkpoints in `ckpt/`  
✅ Training scripts: `Downstream.py`, `Pretrain.py`  
✅ Package files: `setup.py`, `requirements.txt`  

### Best Environment Setup
**Use UV** (fastest):
```bash
uv venv /opt/envs/transpolymer --python 3.9
source /opt/envs/transpolymer/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

### For Finetuning
Just activate env and run:
```bash
python Downstream.py
```

### For MCP
Just activate env and run:
```bash
python main.py --interactive
```

---

**Everything stays in `TransPolymer_pretrained/` - no external dependencies needed!**
