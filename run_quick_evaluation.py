"""Quick evaluation script to run one model on all 10 subjects"""

import subprocess
import sys

def main():
    print("="*70)
    print("Quick Evaluation - Single Model on All 10 Subjects")
    print("="*70)
    print("\nThis will evaluate Llama 3.2-1B on all 10 MMLU subjects.")
    print("This is faster than running all 3 models.")
    print("\nTo run all 3 models, use: python llama_mmlu_eval.py --use-gpu")
    print("="*70)
    
    response = input("\nContinue with quick evaluation? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    print("\nStarting evaluation...")
    print("This may take 30-60 minutes depending on your hardware.\n")
    
    # Run evaluation with just one model
    cmd = [
        sys.executable,
        "llama_mmlu_eval.py",
        "--use-gpu" if input("Use GPU? (y/n): ").lower() == 'y' else "--use-cpu",
        "--models", "llama-3.2-1b"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print("Evaluation complete! Graphs have been generated.")
        print("="*70)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()

