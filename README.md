# [Prediction module for Polymer-Agent](https://github.com/BaratiLab/Polymer-Agent) 
[DOI LINK](https://pubs.acs.org/doi/10.1021/acs.jcim.6c00343)
## Finetuning model for polymer property prediction

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


