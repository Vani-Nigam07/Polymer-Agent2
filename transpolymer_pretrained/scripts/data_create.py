import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import sys

# Set paths
DATA_FOLDER = Path("/home/vani/transpolymer/transpolymer_pretrained/data")
DATA_FOLDER.mkdir(exist_ok=True)

print("Libraries imported successfully!")
print(f"Data folder: {DATA_FOLDER.absolute()}")

# Configuration: Modify these values for your dataset
INPUT_FILE = '/home/vani/transpolymer/transpolymer_pretrained/data/Egb.csv' # Change this to your input file path
DATASET_NAME = "Egb"  # Name for output files (e.g., OPV, PE_I, etc.)
# for future ref:
#Libraries imported successfully!
#Data folder: /home/vani/transpolymer/../data
#✓ Successfully loaded: /home/vani/transpolymer/transpolymer_pretrained/data/OPV.csv
#  Shape: (1203, 2)

# Load the CSV file
try:
    df = pd.read_csv(INPUT_FILE)
    print(f" Successfully loaded: {INPUT_FILE}")
    print(f"  Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {INPUT_FILE}")
    print(f"  Available files in data folder:")
    for f in DATA_FOLDER.glob("*.csv"):
        print(f"    - {f.name}")

# stats:
# 4. Data Statistics:
#           PCE_ave
# count  1203.000000
# mean      4.220274
# std       2.342145
# min       0.010000
# 25%       2.385000
# 50%       4.200000
# 75%       5.975000
# max      10.500000

# Split configuration
RANDOM_STATE = 42  # For reproducibility
TRAIN_RATIO = 0.70

TEST_RATIO = 0.30

print("=" * 60)
print("DATA SPLITTING")
print("=" * 60)
print(f"\nSplit Ratios:")
print(f"  Train: {TRAIN_RATIO*100:.0f}%")

print(f"  Test: {TEST_RATIO*100:.0f}%")
print(f"  Random state: {RANDOM_STATE}")

# First split: separate test set (15%)
train_df, test_df = train_test_split(
    df,
    test_size=TEST_RATIO,
    random_state=RANDOM_STATE
)

print("=" * 60)
print("SAVING PROCESSED DATA")
print("=" * 60)


train_file = DATA_FOLDER / f"{DATASET_NAME}_train.csv"

test_file = DATA_FOLDER / f"{DATASET_NAME}_test.csv"

try:
    train_df.to_csv(train_file, index=False)
    print(f"\n Saved training set: {train_file.name}")
    print(f"  Size: {len(train_df)} samples")
    
    
    test_df.to_csv(test_file, index=False)
    print(f"\n Saved test set: {test_file.name}")
    print(f"  Size: {len(test_df)} samples")
    
    print(f"\n{'='*60}")
    print(f"All files saved successfully to: {DATA_FOLDER.absolute()}")
    print(f"{'='*60}")
    
except Exception as e:
    print(f"\n Error saving files: {e}")



print("=" * 60)
print("VERIFICATION")
print("=" * 60)

# Reload and verify each file
for file_path, set_name in [(train_file, "Training"),  (test_file, "Test")]:
    if file_path.exists():
        df_verify = pd.read_csv(file_path)
        print(f"\n {set_name} Set - {file_path.name}")
        print(f"  Rows: {len(df_verify)}")
        print(f"  Columns: {list(df_verify.columns)}")
        print(f"  Data types:\n{df_verify.dtypes}")
        print(f"  Sample row:")
        print(f"    {df_verify.iloc[0].to_dict()}")
    else:
        print(f"\n File not found: {file_path.name}")

print(f"\n{'='*60}")
print("Data preparation complete! ✓")
print(f"{'='*60}")
