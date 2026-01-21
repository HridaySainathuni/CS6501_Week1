# Note About Graphs

## Current Status

The graphs have been generated, but they currently show data from only **1 model (Llama 3.2-1B)** and **2 subjects (astronomy, business_ethics)** because that's what's in the existing JSON result file.

## To Get Complete Graphs

You need to run a full evaluation with all 3 models and all 10 subjects:

```bash
# Full evaluation with all models and all subjects
python llama_mmlu_eval.py --use-gpu

# This will:
# - Evaluate all 3 models (Llama 3.2-1B, Phi-2, TinyLlama-1.1B)
# - Run on all 10 MMLU subjects
# - Generate complete graphs with all data
```

## Quick Test (2 subjects, 1 model)

If you want to quickly test with 2 subjects first:

1. Temporarily modify `MMLU_SUBJECTS` in `llama_mmlu_eval.py` to have only 2 subjects
2. Run: `python llama_mmlu_eval.py --use-gpu --models llama-3.2-1b`
3. This will generate graphs with 2 subjects

## Regenerating Graphs

If you have existing JSON result files, you can regenerate graphs:

```bash
python generate_graphs_from_json.py
```

This script will:
- Find all JSON result files
- Load the results
- Generate all 5 graphs

## Graph Files

The following graphs are generated in the `results/` directory:

1. `accuracy_by_subject.png` - Shows accuracy for each model across subjects
2. `overall_accuracy.png` - Overall accuracy comparison between models
3. `timing_comparison.png` - Real time vs CPU time comparison
4. `error_analysis.png` - Error rates by subject
5. `accuracy_heatmap.png` - Visual heatmap of accuracy

## Converting to PDF

To convert graphs to PDF format:

```bash
cd "Running an LLM"
python convert_graphs_to_pdf.py
```

Or manually using any image-to-PDF converter.

