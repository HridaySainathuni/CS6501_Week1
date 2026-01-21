"""Generate graphs from existing JSON result files"""

import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

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

def load_results_from_json(json_file):
    """Load results from a JSON file and convert to the format expected by create_graphs"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle old format (single model)
    if "model" in data and "subject_results" in data:
        return {
            "model_key": "llama-3.2-1b",
            "model_name": data.get("model", "Unknown Model").split("/")[-1],
            "results": data["subject_results"],
            "total_correct": data.get("total_correct", 0),
            "total_questions": data.get("total_questions", 0),
            "overall_accuracy": data.get("overall_accuracy", 0),
            "timing": {
                "real_time": data.get("duration_seconds", 0),
                "cpu_time": data.get("duration_seconds", 0) * 0.8,  # Estimate
                "gpu_time": None
            }
        }
    # Handle new format (multiple models)
    elif "results" in data:
        return data["results"]
    else:
        return []

def create_graphs(all_results, output_dir="results"):
    """Create visualization graphs for the results"""
    if not all_results:
        print("No results to plot. Please run evaluation first.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert single result to list if needed
    if isinstance(all_results, dict):
        all_results = [all_results]
    
    # Filter out results with no data
    all_results = [r for r in all_results if r.get("results") and len(r.get("results", [])) > 0]
    
    if not all_results:
        print("No valid results to plot.")
        return
    
    # 1. Accuracy by Subject for Each Model
    fig, ax = plt.subplots(figsize=(14, 8))
    subject_list = MMLU_SUBJECTS
    x = np.arange(len(subject_list))
    width = 0.25
    
    for i, model_result in enumerate(all_results):
        model_name = model_result.get("model_name", "Unknown")
        model_accuracies = []
        for subject in subject_list:
            subject_result = next((r for r in model_result.get("results", []) if r.get("subject") == subject), None)
            if subject_result:
                model_accuracies.append(subject_result.get("accuracy", 0))
            else:
                model_accuracies.append(0)
        
        if any(acc > 0 for acc in model_accuracies):
            ax.bar(x + i * width, model_accuracies, width, label=model_name)
    
    ax.set_xlabel('Subject', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('MMLU Accuracy by Subject for Each Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(all_results) - 1) / 2 if all_results else 0)
    ax.set_xticklabels(subject_list, rotation=45, ha='right')
    if all_results:
        ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_by_subject.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/accuracy_by_subject.png")
    plt.close()
    
    # 2. Overall Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = [r.get("model_name", "Unknown") for r in all_results]
    overall_accs = [r.get("overall_accuracy", 0) for r in all_results]
    
    if model_names and overall_accs:
        bars = ax.bar(model_names, overall_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(model_names)])
        ax.set_ylabel('Overall Accuracy (%)', fontsize=12)
        ax.set_title('Overall MMLU Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, overall_accs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_accuracy.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/overall_accuracy.png")
    plt.close()
    
    # 3. Timing Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    model_names = [r.get("model_name", "Unknown") for r in all_results]
    real_times = []
    cpu_times = []
    
    for r in all_results:
        timing = r.get("timing", {})
        if isinstance(timing, dict):
            real_times.append(timing.get("real_time", 0) / 60)
            cpu_times.append(timing.get("cpu_time", 0) / 60)
        else:
            real_times.append(0)
            cpu_times.append(0)
    
    if model_names and (real_times or cpu_times):
        x = np.arange(len(model_names))
        width = 0.35
        
        if any(t > 0 for t in real_times):
            ax.bar(x - width/2, real_times, width, label='Real Time', alpha=0.8)
        if any(t > 0 for t in cpu_times):
            ax.bar(x + width/2, cpu_times, width, label='CPU Time', alpha=0.8)
        
        ax.set_ylabel('Time (minutes)', fontsize=12)
        ax.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        if real_times or cpu_times:
            ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/timing_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/timing_comparison.png")
    plt.close()
    
    # 4. Error Analysis
    error_analysis = defaultdict(lambda: {"total": 0, "errors": 0})
    
    for model_result in all_results:
        for subject_result in model_result.get("results", []):
            subject = subject_result.get("subject")
            if subject:
                error_analysis[subject]["total"] += subject_result.get("total", 0)
                error_analysis[subject]["errors"] += (subject_result.get("total", 0) - subject_result.get("correct", 0))
    
    if error_analysis:
        subjects_list = list(error_analysis.keys())
        error_rates = [error_analysis[s]["errors"] / error_analysis[s]["total"] * 100 
                       if error_analysis[s]["total"] > 0 else 0
                       for s in subjects_list]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(subjects_list, error_rates, color='coral')
        ax.set_xlabel('Error Rate (%)', fontsize=12)
        ax.set_title('Error Rate by Subject (Combined Across All Models)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/error_analysis.png")
        plt.close()
    
    # 5. Heatmap
    accuracy_matrix = []
    for model_result in all_results:
        model_accs = []
        for subject in MMLU_SUBJECTS:
            subject_result = next((r for r in model_result.get("results", []) if r.get("subject") == subject), None)
            model_accs.append(subject_result.get("accuracy", 0) if subject_result else 0)
        accuracy_matrix.append(model_accs)
    
    if accuracy_matrix and any(any(row) for row in accuracy_matrix):
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(accuracy_matrix, 
                    xticklabels=MMLU_SUBJECTS,
                    yticklabels=[r.get("model_name", "Unknown") for r in all_results],
                    annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                    cbar_kws={'label': 'Accuracy (%)'})
        ax.set_title('Accuracy Heatmap: Models vs Subjects', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/accuracy_heatmap.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/accuracy_heatmap.png")
        plt.close()

def main():
    """Load results from JSON files and generate graphs"""
    json_files = glob.glob("mmlu_results_*.json") + glob.glob("llama_*_mmlu_results_*.json")
    
    if not json_files:
        print("No JSON result files found.")
        print("Please run: python llama_mmlu_eval.py --use-gpu")
        return
    
    # Sort by modification time, newest first
    json_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"Found {len(json_files)} result file(s)")
    
    all_results = []
    for json_file in json_files:
        print(f"Loading: {json_file}")
        try:
            result = load_results_from_json(json_file)
            if isinstance(result, list):
                results_to_add = result
            else:
                results_to_add = [result]
            
            # Check if results have at least 5 subjects (filter out incomplete runs)
            for r in results_to_add:
                subject_count = len(r.get("results", []))
                if subject_count >= 5:  # Only include if has at least 5 subjects
                    all_results.append(r)
                    print(f"  Added: {r.get('model_name', 'Unknown')} with {subject_count} subjects")
                else:
                    print(f"  Skipped: {r.get('model_name', 'Unknown')} - only {subject_count} subjects (incomplete)")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    if all_results:
        print(f"\nGenerating graphs from {len(all_results)} model result(s)...")
        create_graphs(all_results)
        print("\nGraphs generated successfully!")
    else:
        print("No valid results found in JSON files.")

if __name__ == "__main__":
    main()

