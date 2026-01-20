"""Multi-Model MMLU Evaluation Script"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import json
from tqdm.auto import tqdm
import os
from datetime import datetime
import sys
import platform
import argparse
import time as time_module
import psutil
try:
    import resource
except ImportError:
    resource = None
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

MODELS = {
    "llama-3.2-1b": {
        "name": "meta-llama/Llama-3.2-1B-Instruct",
        "display_name": "Llama 3.2-1B"
    },
    "phi-2": {
        "name": "microsoft/phi-2",
        "display_name": "Phi-2"
    },
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "display_name": "TinyLlama-1.1B"
    }
}

MMLU_SUBJECTS = [
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics"
]

MAX_NEW_TOKENS = 1


class TimingTracker:
    def __init__(self, device):
        self.device = device
        self.real_start = None
        self.cpu_start = None
        self.gpu_start = None
        self.process = psutil.Process()
        
    def start(self):
        self.real_start = time_module.time()
        self.cpu_start = self.process.cpu_times()
        if self.device == "cuda" and torch.cuda.is_available():
            self.gpu_start = torch.cuda.Event(enable_timing=True)
            self.gpu_start.record()
        return self
    
    def stop(self):
        real_time = time_module.time() - self.real_start
        
        cpu_times = self.process.cpu_times()
        cpu_time = (cpu_times.user - self.cpu_start.user) + (cpu_times.system - self.cpu_start.system)
        
        gpu_time = None
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_end.record()
            torch.cuda.synchronize()
            gpu_time = self.gpu_start.elapsed_time(gpu_end) / 1000.0
        
        return {
            "real_time": real_time,
            "cpu_time": cpu_time,
            "gpu_time": gpu_time
        }


def detect_device(use_gpu):
    if not use_gpu:
        return "cpu"
    
    if torch.cuda.is_available():
        return "cuda"
    
    if torch.backends.mps.is_available():
        return "mps"
    
    return "cpu"


def check_environment(device, quantization_bits):
    print("="*70)
    print("Environment Check")
    print("="*70)
    
    try:
        import google.colab
        print("✓ Running in Google Colab")
        in_colab = True
    except:
        print("✓ Running locally (not in Colab)")
        in_colab = False
    
    print(f"✓ Platform: {platform.system()} ({platform.machine()})")
    if platform.system() == "Darwin":
        print(f"✓ Processor: {platform.processor()}")
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("✓ Apple Metal (MPS) Available")
    else:
        print("⚠ No GPU detected - running on CPU")
    
    if quantization_bits is not None:
        try:
            import bitsandbytes
            print(f"✓ bitsandbytes installed - {quantization_bits}-bit quantization available")
        except ImportError:
            print(f"✗ bitsandbytes NOT installed - cannot use quantization")
            sys.exit(1)
        if device == 'mps':
            print(f"⚠ Apple METAL is incompatible with quantization")
            print("✓ Quantization disabled - loading full precision model")
            quantization_bits = None
    else:
        print("✓ Quantization disabled - loading full precision model")
    
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ Hugging Face authenticated")
        else:
            print("⚠ No Hugging Face token found")
            print("Run: huggingface-cli login")
    except:
        print("⚠ Could not check Hugging Face authentication")
    
    print("="*70 + "\n")
    return in_colab, quantization_bits


def get_quantization_config(quantization_bits):
    if quantization_bits is None:
        return None
    
    if quantization_bits == 4:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif quantization_bits == 8:
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    else:
        raise ValueError(f"Invalid quantization_bits: {quantization_bits}. Use 4, 8, or None")
    
    return config


def load_model_and_tokenizer(model_key, device, quantization_bits):
    model_info = MODELS[model_key]
    model_name = model_info["name"]
    
    print(f"\nLoading model: {model_info['display_name']} ({model_name})...")
    print(f"Device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded")
        
        quant_config = get_quantization_config(quantization_bits)
        
        print("Loading model (this may take 2-3 minutes)...")
        
        if quant_config is not None:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            elif device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
        
        model.eval()
        print("✓ Model loaded successfully!")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        raise


def format_mmlu_prompt(question, choices, model_key):
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def get_model_prediction(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    answer = generated_text.strip()[:1].upper()
    
    if answer not in ["A", "B", "C", "D"]:
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
        else:
            answer = "A"
    
    return answer


def evaluate_subject(model, tokenizer, model_key, subject, device, verbose=False):
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'='*70}")
    
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"✗ Error loading subject {subject}: {e}")
        return None
    
    correct = 0
    total = 0
    question_details = []
    
    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]
        
        prompt = format_mmlu_prompt(question, choices, model_key)
        predicted_answer = get_model_prediction(model, tokenizer, prompt)
        
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1
        
        question_details.append({
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        })
        
        if verbose:
            print(f"\nQuestion: {question}")
            print(f"Choices: {choices}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Model Answer: {predicted_answer}")
            print(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            print("-" * 70)
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")
    
    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "question_details": question_details if verbose else None
    }


def evaluate_model(model_key, device, quantization_bits, verbose=False):
    model_info = MODELS[model_key]
    print(f"\n{'='*70}")
    print(f"Evaluating Model: {model_info['display_name']}")
    print(f"{'='*70}")
    
    model, tokenizer = load_model_and_tokenizer(model_key, device, quantization_bits)
    
    timing_tracker = TimingTracker(device)
    overall_timer = timing_tracker.start()
    
    results = []
    total_correct = 0
    total_questions = 0
    
    for i, subject in enumerate(MMLU_SUBJECTS, 1):
        print(f"\nProgress: {i}/{len(MMLU_SUBJECTS)} subjects")
        subject_timer = TimingTracker(device).start()
        
        result = evaluate_subject(model, tokenizer, model_key, subject, device, verbose)
        
        if result:
            subject_timing = subject_timer.stop()
            result["timing"] = subject_timing
            results.append(result)
            total_correct += result["correct"]
            total_questions += result["total"]
    
    overall_timing = overall_timer.stop()
    
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "model_key": model_key,
        "model_name": model_info["display_name"],
        "results": results,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "overall_accuracy": overall_accuracy,
        "timing": overall_timing
    }


def create_graphs(all_results, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    subject_list = MMLU_SUBJECTS
    x = np.arange(len(subject_list))
    width = 0.25
    
    for i, model_result in enumerate(all_results):
        model_name = model_result["model_name"]
        model_accuracies = [r["accuracy"] for r in model_result["results"]]
        ax.bar(x + i * width, model_accuracies, width, label=model_name)
    
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('MMLU Accuracy by Subject for Each Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(subject_list, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_by_subject.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/accuracy_by_subject.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = [r["model_name"] for r in all_results]
    overall_accs = [r["overall_accuracy"] for r in all_results]
    
    bars = ax.bar(model_names, overall_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Overall Accuracy (%)', fontsize=12)
    ax.set_title('Overall MMLU Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, overall_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_accuracy.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/overall_accuracy.png")
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = [r["model_name"] for r in all_results]
    real_times = [r["timing"]["real_time"] / 60 for r in all_results]
    cpu_times = [r["timing"]["cpu_time"] / 60 for r in all_results]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax.bar(x - width/2, real_times, width, label='Real Time', alpha=0.8)
    ax.bar(x + width/2, cpu_times, width, label='CPU Time', alpha=0.8)
    
    ax.set_ylabel('Time (minutes)', fontsize=12)
    ax.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/timing_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/timing_comparison.png")
    plt.close()
    
    error_analysis = defaultdict(lambda: {"total": 0, "errors": 0})
    
    for model_result in all_results:
        for subject_result in model_result["results"]:
            subject = subject_result["subject"]
            error_analysis[subject]["total"] += subject_result["total"]
            error_analysis[subject]["errors"] += (subject_result["total"] - subject_result["correct"])
    
    subjects_list = list(error_analysis.keys())
    error_rates = [error_analysis[s]["errors"] / error_analysis[s]["total"] * 100 
                   for s in subjects_list]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(subjects_list, error_rates, color='coral')
    ax.set_xlabel('Error Rate (%)', fontsize=12)
    ax.set_title('Error Rate by Subject (Combined Across All Models)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/error_analysis.png")
    plt.close()
    
    accuracy_matrix = []
    for model_result in all_results:
        model_accs = []
        for subject in MMLU_SUBJECTS:
            subject_result = next((r for r in model_result["results"] if r["subject"] == subject), None)
            model_accs.append(subject_result["accuracy"] if subject_result else 0)
        accuracy_matrix.append(model_accs)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(accuracy_matrix, 
                xticklabels=MMLU_SUBJECTS,
                yticklabels=[r["model_name"] for r in all_results],
                annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                cbar_kws={'label': 'Accuracy (%)'})
    ax.set_title('Accuracy Heatmap: Models vs Subjects', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/accuracy_heatmap.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate multiple models on MMLU benchmark')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--use-cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--quantization', type=str, default='None',
                        help='Quantization bits: 4, 8, or None (default: None)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print each question, answer, and correctness')
    parser.add_argument('--models', nargs='+', choices=list(MODELS.keys()),
                        default=list(MODELS.keys()),
                        help='Models to evaluate (default: all)')
    
    args = parser.parse_args()
    
    use_gpu = args.use_gpu and not args.use_cpu
    if args.quantization.lower() == 'none':
        quantization_bits = None
    elif args.quantization in ['4', '8']:
        quantization_bits = int(args.quantization)
    else:
        print(f"Error: Invalid quantization value: {args.quantization}. Use 4, 8, or None")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Multi-Model MMLU Evaluation")
    print("="*70 + "\n")
    
    device = detect_device(use_gpu)
    in_colab, quantization_bits = check_environment(device, quantization_bits)
    
    all_results = []
    for model_key in args.models:
        try:
            result = evaluate_model(model_key, device, quantization_bits, args.verbose)
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Error evaluating {MODELS[model_key]['display_name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    for result in all_results:
        print(f"\n{result['model_name']}:")
        print(f"  Overall Accuracy: {result['overall_accuracy']:.2f}%")
        print(f"  Total Questions: {result['total_questions']}")
        print(f"  Total Correct: {result['total_correct']}")
        print(f"  Real Time: {result['timing']['real_time']/60:.2f} minutes")
        print(f"  CPU Time: {result['timing']['cpu_time']/60:.2f} minutes")
        if result['timing']['gpu_time']:
            print(f"  GPU Time: {result['timing']['gpu_time']/60:.2f} minutes")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant_suffix = f"_{quantization_bits}bit" if quantization_bits else "_full"
    device_suffix = f"_{device}"
    output_file = f"mmlu_results{quant_suffix}{device_suffix}_{timestamp}.json"
    
    output_data = {
        "timestamp": timestamp,
        "device": str(device),
        "quantization_bits": quantization_bits,
        "models_evaluated": args.models,
        "results": all_results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*70)
    print("Generating Graphs")
    print("="*70)
    create_graphs(all_results)
    
    print("\n✓ Evaluation complete!")
    return output_file


if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\n⚠ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
