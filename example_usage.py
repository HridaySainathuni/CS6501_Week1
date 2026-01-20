"""
Example Usage Script

This script demonstrates how to use the evaluation and chat agent
functionality programmatically.
"""

import json
from llama_mmlu_eval import (
    detect_device, check_environment, load_model_and_tokenizer,
    evaluate_subject, MODELS, MMLU_SUBJECTS, TimingTracker
)
from chat_agent import ChatAgent

def example_evaluation():
    """Example of running a single model evaluation"""
    print("="*70)
    print("Example: Single Model Evaluation")
    print("="*70)
    
    # Configuration
    model_key = "llama-3.2-1b"
    use_gpu = True
    quantization_bits = None
    verbose = False
    
    # Setup
    device = detect_device(use_gpu)
    in_colab, quantization_bits = check_environment(device, quantization_bits)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_key, device, quantization_bits)
    
    # Evaluate on a single subject
    subject = MMLU_SUBJECTS[0]  # astronomy
    print(f"\nEvaluating on subject: {subject}")
    
    timer = TimingTracker(device).start()
    result = evaluate_subject(model, tokenizer, model_key, subject, device, verbose)
    timing = timer.stop()
    
    if result:
        print(f"\nResults:")
        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  Correct: {result['correct']}/{result['total']}")
        print(f"  Real Time: {timing['real_time']:.2f} seconds")
        print(f"  CPU Time: {timing['cpu_time']:.2f} seconds")
        if timing['gpu_time']:
            print(f"  GPU Time: {timing['gpu_time']:.2f} seconds")
    
    # Cleanup
    del model, tokenizer
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()

def example_chat():
    """Example of using the chat agent programmatically"""
    print("\n" + "="*70)
    print("Example: Chat Agent Usage")
    print("="*70)
    
    # Create chat agent
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    agent = ChatAgent(model_name, use_history=True)
    
    # Have a conversation
    questions = [
        "What is machine learning?",
        "Can you give me a simple example?",
        "How does it differ from traditional programming?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        response = agent.generate_response(question)
        print(f"Assistant: {response}")
    
    # Show conversation history
    print(f"\nConversation History: {len(agent.conversation_history)} turns")
    for i, turn in enumerate(agent.conversation_history, 1):
        print(f"  Turn {i}:")
        print(f"    User: {turn['user'][:50]}...")
        print(f"    Assistant: {turn['assistant'][:50]}...")
    
    # Test without history
    print("\n" + "="*70)
    print("Testing without history:")
    print("="*70)
    agent_no_history = ChatAgent(model_name, use_history=False)
    
    for question in questions[:2]:
        print(f"\nUser: {question}")
        response = agent_no_history.generate_response(question)
        print(f"Assistant: {response}")
        print(f"History length: {len(agent_no_history.conversation_history)}")

def example_load_results():
    """Example of loading and analyzing results"""
    print("\n" + "="*70)
    print("Example: Loading Results")
    print("="*70)
    
    # Find the most recent results file
    import glob
    import os
    
    result_files = glob.glob("mmlu_results_*.json")
    if not result_files:
        print("No results files found. Run evaluation first.")
        return
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"Loading: {latest_file}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nEvaluation Summary:")
    print(f"  Device: {data['device']}")
    print(f"  Quantization: {data['quantization_bits']}")
    print(f"  Models Evaluated: {len(data['results'])}")
    
    for model_result in data['results']:
        print(f"\n  {model_result['model_name']}:")
        print(f"    Overall Accuracy: {model_result['overall_accuracy']:.2f}%")
        print(f"    Real Time: {model_result['timing']['real_time']/60:.2f} minutes")
        
        # Find best and worst subjects
        subject_results = model_result['results']
        best = max(subject_results, key=lambda x: x['accuracy'])
        worst = min(subject_results, key=lambda x: x['accuracy'])
        print(f"    Best Subject: {best['subject']} ({best['accuracy']:.2f}%)")
        print(f"    Worst Subject: {worst['subject']} ({worst['accuracy']:.2f}%)")

if __name__ == "__main__":
    print("Example Usage Script")
    print("="*70)
    print("\nThis script demonstrates programmatic usage of the evaluation")
    print("and chat agent functionality.")
    print("\nNote: This requires models to be loaded, which may take time.")
    print("Uncomment the examples you want to run.\n")
    
    # Uncomment to run examples:
    # example_evaluation()
    # example_chat()
    # example_load_results()
    
    print("\nTo run examples, uncomment them in the script.")

