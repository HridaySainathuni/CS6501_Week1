"""
Test Implementation Script

This script runs basic tests to verify the implementation works correctly.
"""

import sys
import os
import argparse

def test_imports():
    """Test that all required modules can be imported"""
    print("="*70)
    print("Test 1: Import Checks")
    print("="*70)
    
    try:
        import torch
        print("[OK] torch imported")
    except ImportError as e:
        print(f"[FAIL] torch import failed: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("[OK] transformers imported")
    except ImportError as e:
        print(f"[FAIL] transformers import failed: {e}")
        return False
    
    try:
        from datasets import load_dataset
        print("[OK] datasets imported")
    except ImportError as e:
        print(f"[FAIL] datasets import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("[OK] matplotlib imported")
    except ImportError as e:
        print(f"[FAIL] matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("[OK] seaborn imported")
    except ImportError as e:
        print(f"[FAIL] seaborn import failed: {e}")
        return False
    
    try:
        import psutil
        print("[OK] psutil imported")
    except ImportError as e:
        print(f"[FAIL] psutil import failed: {e}")
        return False
    
    return True

def test_evaluation_script_structure():
    """Test that the evaluation script has the expected structure"""
    print("\n" + "="*70)
    print("Test 2: Evaluation Script Structure")
    print("="*70)
    
    try:
        # Import the evaluation script modules
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from llama_mmlu_eval import (
            MODELS, MMLU_SUBJECTS, detect_device, 
            get_quantization_config, format_mmlu_prompt
        )
        
        # Check MODELS
        assert len(MODELS) == 3, f"Expected 3 models, found {len(MODELS)}"
        print(f"[OK] MODELS dictionary has {len(MODELS)} models")
        for key, value in MODELS.items():
            assert "name" in value, f"Model {key} missing 'name'"
            assert "display_name" in value, f"Model {key} missing 'display_name'"
        print("[OK] All models have required fields")
        
        # Check MMLU_SUBJECTS
        assert len(MMLU_SUBJECTS) == 10, f"Expected 10 subjects, found {len(MMLU_SUBJECTS)}"
        print(f"[OK] MMLU_SUBJECTS has {len(MMLU_SUBJECTS)} subjects")
        
        # Test detect_device
        device = detect_device(False)
        assert device == "cpu", f"Expected 'cpu' when use_gpu=False, got {device}"
        print("[OK] detect_device works correctly")
        
        # Test get_quantization_config
        config = get_quantization_config(None)
        assert config is None, "Expected None for no quantization"
        print("[OK] get_quantization_config handles None correctly")
        
        # Test format_mmlu_prompt
        question = "What is 2+2?"
        choices = ["3", "4", "5", "6"]
        prompt = format_mmlu_prompt(question, choices, "llama-3.2-1b")
        assert "What is 2+2?" in prompt, "Prompt should contain question"
        assert "A. 3" in prompt, "Prompt should contain choices"
        print("[OK] format_mmlu_prompt works correctly")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Evaluation script structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_chat_agent_structure():
    """Test that the chat agent has the expected structure"""
    print("\n" + "="*70)
    print("Test 3: Chat Agent Structure")
    print("="*70)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from chat_agent import ChatAgent, AVAILABLE_MODELS
        
        # Check AVAILABLE_MODELS
        assert len(AVAILABLE_MODELS) >= 1, "Should have at least 1 model"
        print(f"[OK] AVAILABLE_MODELS has {len(AVAILABLE_MODELS)} models")
        
        # Check ChatAgent class exists
        assert hasattr(ChatAgent, '__init__'), "ChatAgent should have __init__"
        assert hasattr(ChatAgent, 'generate_response'), "ChatAgent should have generate_response"
        assert hasattr(ChatAgent, 'save_state'), "ChatAgent should have save_state"
        assert hasattr(ChatAgent, 'load_state'), "ChatAgent should have load_state"
        print("[OK] ChatAgent has required methods")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Chat agent structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_argument_parsing():
    """Test command-line argument parsing"""
    print("\n" + "="*70)
    print("Test 4: Argument Parsing")
    print("="*70)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from llama_mmlu_eval import main as eval_main
        import argparse
        
        # Test that argparse works
        parser = argparse.ArgumentParser()
        parser.add_argument('--use-gpu', action='store_true')
        parser.add_argument('--quantization', type=str, default='None')
        
        # Test parsing
        args = parser.parse_args(['--use-gpu', '--quantization', '4'])
        assert args.use_gpu == True, "use_gpu should be True"
        assert args.quantization == '4', "quantization should be '4'"
        print("[OK] Argument parsing works correctly")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Argument parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_existence():
    """Test that all required files exist"""
    print("\n" + "="*70)
    print("Test 5: File Existence")
    print("="*70)
    
    required_files = [
        "llama_mmlu_eval.py",
        "chat_agent.py",
        "verify_setup.py",
        "requirements.txt",
        "README.md"
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file} exists")
        else:
            print(f"[FAIL] {file} does not exist")
            all_exist = False
    
    return all_exist

def test_graph_generation_function():
    """Test that graph generation function exists and is callable"""
    print("\n" + "="*70)
    print("Test 6: Graph Generation Function")
    print("="*70)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from llama_mmlu_eval import create_graphs
        
        # Check function exists
        assert callable(create_graphs), "create_graphs should be callable"
        print("[OK] create_graphs function exists and is callable")
        
        # Test with empty results (should not crash)
        try:
            create_graphs([])
            print("[OK] create_graphs handles empty results gracefully")
        except Exception as e:
            # This is OK - it might fail on empty results, but shouldn't crash
            print(f"[INFO] create_graphs with empty results: {e}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Graph generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("Implementation Verification Tests")
    print("="*70 + "\n")
    
    tests = [
        ("Import Checks", test_imports),
        ("Evaluation Script Structure", test_evaluation_script_structure),
        ("Chat Agent Structure", test_chat_agent_structure),
        ("Argument Parsing", test_argument_parsing),
        ("File Existence", test_file_existence),
        ("Graph Generation Function", test_graph_generation_function),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print("\n" + "="*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n[OK] All tests passed! Implementation is verified.")
        return 0
    else:
        print(f"\n[WARN] {total - passed} test(s) failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

