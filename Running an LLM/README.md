# Running an LLM - Portfolio Submission

This directory contains the code, graphs, and analysis for the "Running an LLM" assignment.

## Contents

### Code Files
- `llama_mmlu_eval.py` - Multi-model MMLU evaluation script
- `chat_agent.py` - Chat agent with context management
- `verify_setup.py` - Environment setup verification

### Documentation
- `VERIFICATION_CHECKLIST.md` - Complete checklist of implemented features
- `ANALYSIS.md` - Analysis and discussion of results

### Graphs (PDF format)
- `accuracy_by_subject.pdf` - Accuracy comparison across subjects
- `overall_accuracy.pdf` - Overall model performance
- `timing_comparison.pdf` - Performance timing analysis
- `error_analysis.pdf` - Error pattern analysis
- `accuracy_heatmap.pdf` - Visual heatmap of results

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Verify setup: `python verify_setup.py`
3. Run evaluation: `python llama_mmlu_eval.py --use-gpu`
4. Run chat agent: `python chat_agent.py`

## Important: Getting Complete Graphs

**Current graphs show only 2 subjects** because the existing JSON file has partial results.

To get **complete graphs with all 10 subjects**, run:

```bash
# Quick: 1 model on all 10 subjects (30-60 min)
python llama_mmlu_eval.py --use-gpu --models llama-3.2-1b

# Full: All 3 models on all 10 subjects (2-3 hours)
python llama_mmlu_eval.py --use-gpu
```

See `HOW_TO_GET_COMPLETE_GRAPHS.md` for detailed instructions.

## Features Implemented

✅ 3 tiny/small models evaluated  
✅ 10 MMLU subjects  
✅ Timing information (real, CPU, GPU time)  
✅ Verbose mode for question-level analysis  
✅ Graph generation  
✅ Chat agent with context management  
✅ History toggle option  
✅ Restartability with pickle  

See `VERIFICATION_CHECKLIST.md` for complete details.

