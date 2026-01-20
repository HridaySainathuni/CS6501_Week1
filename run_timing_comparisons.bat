@echo off
REM Script to run timing comparisons for MMLU evaluation on Windows
REM This script runs the evaluation with different configurations and times them

echo ==========================================
echo MMLU Evaluation Timing Comparisons
echo ==========================================
echo.

REM Create results directory
if not exist timing_results mkdir timing_results

REM GPU with no quantization
echo Running: GPU + No Quantization
echo ----------------------------------------
python -m timeit -n 1 -r 1 "import subprocess; subprocess.run(['python', 'llama_mmlu_eval.py', '--use-gpu', '--quantization', 'None'], capture_output=True)" > timing_results\gpu_no_quant.log 2>&1
python llama_mmlu_eval.py --use-gpu --quantization None >> timing_results\gpu_no_quant.log 2>&1
echo.

REM Check for NVIDIA GPU
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    REM GPU with 4-bit quantization
    echo Running: GPU + 4-bit Quantization
    echo ----------------------------------------
    python llama_mmlu_eval.py --use-gpu --quantization 4 > timing_results\gpu_4bit.log 2>&1
    echo.
    
    REM GPU with 8-bit quantization
    echo Running: GPU + 8-bit Quantization
    echo ----------------------------------------
    python llama_mmlu_eval.py --use-gpu --quantization 8 > timing_results\gpu_8bit.log 2>&1
    echo.
) else (
    echo Skipping GPU quantization tests (not on NVIDIA GPU)
    echo.
)

REM CPU with no quantization
echo Running: CPU + No Quantization
echo ----------------------------------------
python llama_mmlu_eval.py --use-cpu --quantization None > timing_results\cpu_no_quant.log 2>&1
echo.

REM CPU with 4-bit quantization
echo Running: CPU + 4-bit Quantization
echo ----------------------------------------
python llama_mmlu_eval.py --use-cpu --quantization 4 > timing_results\cpu_4bit.log 2>&1
echo.

echo ==========================================
echo All timing comparisons complete!
echo Results saved in timing_results\ directory
echo ==========================================
pause

