

# from PolymerSmilesTokenization import PolymerSmilesTokenizer
# import os

# # tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base")

# save_dir = "TransPolymer/checkpoint"
# os.makedirs(save_dir, exist_ok=True)
# # ...existing code...
# # tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base")
# tokenizer = PolymerSmilesTokenizer.from_pretrained(
#     "/media/vani/ebc68877-0047-41f0-9256-3014e42ef8e1/vani/mcp/mcp2/TransPolymer_pretrained/ckpt",
#     local_files_only=True
# )
# # ✅ This automatically saves:
# # vocab.json, merges.txt, tokenizer_config.json, special_tokens_map.json, tokenizer.json
# tokenizer.save_pretrained(save_dir)
from transformers import RobertaTokenizer
import os

# Path where you want to store tokenizer files
save_dir = "/home/vani/transpolymer/transpolymer_pretrained/tokenizer"
os.makedirs(save_dir, exist_ok=True)

# 1️⃣ Download RoBERTa tokenizer from HF ONCE
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# 2️⃣ Save all tokenizer files locally
#    This will create vocab.json, merges.txt, tokenizer_config.json, special_tokens_map.json, tokenizer.json
tokenizer.save_pretrained(save_dir)

print(f"Tokenizer files saved to: {save_dir}")

