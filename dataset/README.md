# Dataset

This folder is reserved for local dataset files if needed in the future.

## Current Setup

**The project currently uses the SNLI dataset from HuggingFace, which is downloaded automatically.**

- **Dataset**: SNLI (Stanford Natural Language Inference)
- **Source**: HuggingFace Datasets (`datasets.load_dataset('snli')`)
- **Cache Location**: `~/.cache/huggingface/` (or `HF_HOME`/`TRANSFORMERS_CACHE`)

The dataset is automatically downloaded and cached the first time any training/evaluation script runs. No manual download is required.

## Dataset Details

- **Training set**: ~550,000 examples
- **Validation set**: 9,842 examples  
- **Test set**: 9,824 examples
- **Task**: Natural Language Inference (NLI)
- **Labels**: 
  - 0: Entailment
  - 1: Neutral
  - 2: Contradiction

## Using Local Dataset Files

If you want to use local dataset files instead, you can:

1. Place JSON/JSONL files in this folder
2. Use the `--dataset` argument with `run.py` pointing to your local file:
   ```bash
   python train/run.py --dataset dataset/your_file.jsonl --task nli
   ```

The dataset format should be:
```json
{"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
```

