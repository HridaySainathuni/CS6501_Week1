"""Chat Agent with Context Management"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import os
import pickle
from datetime import datetime
import sys
import platform

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

AVAILABLE_MODELS = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "phi-2": "microsoft/phi-2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

MAX_CONTEXT_LENGTH = 2048
MAX_HISTORY_TURNS = 10


class ChatAgent:
    def __init__(self, model_name, use_history=True, device=None):
        self.model_name = model_name
        self.use_history = use_history
        self.conversation_history = []
        self.device = device or self._detect_device()
        
        print(f"Loading model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✓ Model loaded successfully!")
    
    def _detect_device(self):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _format_prompt(self, user_input, history=None):
        if not self.use_history or history is None:
            if "llama" in self.model_name.lower():
                prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif "phi" in self.model_name.lower():
                prompt = f"Instruct: {user_input}\nOutput:"
            elif "tinyllama" in self.model_name.lower():
                prompt = f"<|user|>\n{user_input}<|assistant|>\n"
            else:
                prompt = f"User: {user_input}\nAssistant:"
            return prompt
        
        if "llama" in self.model_name.lower():
            prompt = "<|begin_of_text|>"
            for turn in history:
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{turn['user']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{turn['assistant']}<|eot_id|>"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "phi" in self.model_name.lower():
            prompt = ""
            for turn in history:
                prompt += f"Instruct: {turn['user']}\nOutput: {turn['assistant']}\n\n"
            prompt += f"Instruct: {user_input}\nOutput:"
        elif "tinyllama" in self.model_name.lower():
            prompt = ""
            for turn in history:
                prompt += f"<|user|>\n{turn['user']}<|assistant|>\n{turn['assistant']}<|endoftext|>\n"
            prompt += f"<|user|>\n{user_input}<|assistant|>\n"
        else:
            prompt = ""
            for turn in history:
                prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"
            prompt += f"User: {user_input}\nAssistant:"
        
        return prompt
    
    def _manage_context(self):
        if not self.use_history:
            return []
        
        recent_history = self.conversation_history[-MAX_HISTORY_TURNS:]
        
        total_tokens = 0
        trimmed_history = []
        
        for turn in reversed(recent_history):
            turn_tokens = len(self._format_prompt(turn['user'], [turn])) // 4
            if total_tokens + turn_tokens > MAX_CONTEXT_LENGTH:
                break
            trimmed_history.insert(0, turn)
            total_tokens += turn_tokens
        
        return trimmed_history
    
    def generate_response(self, user_input):
        history = self._manage_context()
        prompt = self._format_prompt(user_input, history)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        response = generated_text.strip()
        
        if self.use_history:
            self.conversation_history.append({
                "user": user_input,
                "assistant": response
            })
        
        return response
    
    def clear_history(self):
        self.conversation_history = []
        print("✓ Conversation history cleared")
    
    def save_state(self, filepath):
        state = {
            "model_name": self.model_name,
            "conversation_history": self.conversation_history,
            "use_history": self.use_history,
            "timestamp": datetime.now().isoformat()
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        print(f"✓ State saved to {filepath}")
    
    @classmethod
    def load_state(cls, filepath, device=None):
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        
        print(f"Loading saved state from {filepath}")
        print(f"State timestamp: {state['timestamp']}")
        print(f"Conversation turns: {len(state['conversation_history'])}")
        
        agent = cls(state["model_name"], state["use_history"], device)
        agent.conversation_history = state["conversation_history"]
        
        print("✓ State loaded successfully!")
        return agent


def main():
    parser = argparse.ArgumentParser(description='Chat Agent with Context Management')
    parser.add_argument('--model', choices=list(AVAILABLE_MODELS.keys()),
                        default="llama-3.2-1b",
                        help='Model to use for chat')
    parser.add_argument('--no-history', action='store_true',
                        help='Disable conversation history')
    parser.add_argument('--save-state', action='store_true',
                        help='Enable state saving/loading')
    parser.add_argument('--load-state', type=str,
                        help='Load state from file')
    
    args = parser.parse_args()
    
    model_name = AVAILABLE_MODELS[args.model]
    use_history = not args.no_history
    save_state = args.save_state
    
    print("="*70)
    print("Chat Agent")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"History: {'Enabled' if use_history else 'Disabled'}")
    print(f"State saving: {'Enabled' if save_state else 'Disabled'}")
    print("="*70 + "\n")
    
    state_file = "chat_agent_state.pkl"
    
    if args.load_state:
        try:
            agent = ChatAgent.load_state(args.load_state)
        except Exception as e:
            print(f"Error loading state: {e}")
            print("Creating new agent...")
            agent = ChatAgent(model_name, use_history)
    else:
        if save_state and os.path.exists(state_file):
            response = input(f"Found existing state file ({state_file}). Load it? (y/n): ")
            if response.lower() == 'y':
                try:
                    agent = ChatAgent.load_state(state_file)
                except Exception as e:
                    print(f"Error loading state: {e}")
                    print("Creating new agent...")
                    agent = ChatAgent(model_name, use_history)
            else:
                agent = ChatAgent(model_name, use_history)
        else:
            agent = ChatAgent(model_name, use_history)
    
    print("\n" + "="*70)
    print("Chat started! Type 'quit' or 'exit' to end, 'clear' to clear history")
    print("="*70 + "\n")
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.lower() == 'clear':
                agent.clear_history()
                continue
            
            print("Assistant: ", end="", flush=True)
            response = agent.generate_response(user_input)
            print(response)
            print()
            
            if save_state:
                agent.save_state(state_file)
    
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user")
    
    finally:
        if save_state:
            agent.save_state(state_file)
            print(f"\n✓ Final state saved to {state_file}")
        
        print("\nGoodbye!")


if __name__ == "__main__":
    main()

