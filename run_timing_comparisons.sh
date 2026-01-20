#!/bin/bash
# Script to run timing comparisons for MMLU evaluation
# This script runs the evaluation with different configurations and times them

echo "=========================================="
echo "MMLU Evaluation Timing Comparisons"
echo "=========================================="
echo ""

# Create results directory
mkdir -p timing_results

# GPU with no quantization
echo "Running: GPU + No Quantization"
echo "----------------------------------------"
time python llama_mmlu_eval.py --use-gpu --quantization None > timing_results/gpu_no_quant.log 2>&1
echo ""

# GPU with 4-bit quantization (skip on Mac)
if command -v nvidia-smi &> /dev/null; then
    echo "Running: GPU + 4-bit Quantization"
    echo "----------------------------------------"
    time python llama_mmlu_eval.py --use-gpu --quantization 4 > timing_results/gpu_4bit.log 2>&1
    echo ""
    
    # GPU with 8-bit quantization
    echo "Running: GPU + 8-bit Quantization"
    echo "----------------------------------------"
    time python llama_mmlu_eval.py --use-gpu --quantization 8 > timing_results/gpu_8bit.log 2>&1
    echo ""
else
    echo "Skipping GPU quantization tests (not on NVIDIA GPU)"
    echo ""
fi

# CPU with no quantization
echo "Running: CPU + No Quantization"
echo "----------------------------------------"
time python llama_mmlu_eval.py --use-cpu --quantization None > timing_results/cpu_no_quant.log 2>&1
echo ""

# CPU with 4-bit quantization
echo "Running: CPU + 4-bit Quantization"
echo "----------------------------------------"
time python llama_mmlu_eval.py --use-cpu --quantization 4 > timing_results/cpu_4bit.log 2>&1
echo ""

echo "=========================================="
echo "All timing comparisons complete!"
echo "Results saved in timing_results/ directory"
echo "=========================================="

