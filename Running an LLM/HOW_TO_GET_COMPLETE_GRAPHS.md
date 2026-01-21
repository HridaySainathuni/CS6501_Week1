# How to Get Complete Graphs with All 10 Subjects

## Current Situation

Your graphs currently show data for only **2 subjects** (astronomy, business_ethics) because the existing JSON result file only contains partial results.

## Solution: Run Full Evaluation

To get complete graphs with all 10 subjects, you need to run a full evaluation.

### Option 1: Quick Evaluation (Recommended for Testing)

Run **1 model** on **all 10 subjects** - this is faster:

```bash
# Quick evaluation with one model
python run_quick_evaluation.py

# Or manually:
python llama_mmlu_eval.py --use-gpu --models llama-3.2-1b
```

**Time estimate**: 30-60 minutes (depending on GPU/CPU)

### Option 2: Full Evaluation (All 3 Models)

Run **all 3 models** on **all 10 subjects** - this takes longer but gives complete comparison:

```bash
# Full evaluation with all models
python llama_mmlu_eval.py --use-gpu
```

**Time estimate**: 2-3 hours (depending on hardware)

### Option 3: CPU Evaluation (If No GPU)

```bash
# CPU evaluation (slower but works on any machine)
python llama_mmlu_eval.py --use-cpu --models llama-3.2-1b
```

**Time estimate**: 2-4 hours

## What Happens After Evaluation

Once the evaluation completes:
1. ✅ New JSON result file will be created with all 10 subjects
2. ✅ All 5 graphs will be automatically generated with complete data
3. ✅ Graphs will show all subjects on the x-axis with actual data

## Graph Files Generated

After running the evaluation, you'll get:

1. `accuracy_by_subject.png` - Shows all 10 subjects with model accuracy
2. `overall_accuracy.png` - Overall model performance
3. `timing_comparison.png` - Performance timing
4. `error_analysis.png` - Error rates for all subjects
5. `accuracy_heatmap.png` - Complete heatmap with all subjects

## Quick Test (2 Subjects Only)

If you want to test the system first with just 2 subjects:

1. Edit `llama_mmlu_eval.py`
2. Temporarily change `MMLU_SUBJECTS` to:
   ```python
   MMLU_SUBJECTS = [
       "astronomy",
       "business_ethics"
   ]
   ```
3. Run: `python llama_mmlu_eval.py --use-gpu --models llama-3.2-1b`
4. Change it back to all 10 subjects for full evaluation

## Regenerating Graphs

If you already have JSON files with more subjects, regenerate graphs:

```bash
python generate_graphs_from_json.py
```

## Notes

- **GPU is much faster**: If you have an NVIDIA GPU, use `--use-gpu`
- **Quantization saves memory**: Use `--quantization 4` if you run out of memory
- **Can run in background**: Start evaluation and let it run
- **Results are saved**: Even if interrupted, partial results are saved to JSON

## Expected Graph Appearance

After full evaluation, graphs will show:
- **X-axis**: All 10 subjects (astronomy, business_ethics, clinical_knowledge, etc.)
- **Y-axis**: Accuracy percentages or timing metrics
- **Data points**: Actual values for each subject
- **Multiple bars/lines**: One for each model evaluated

