from enum import Enum, auto as enum_auto
from transformers import pipeline
import torch
import json
import os

# Function to determine the best available device
def get_device():
    if torch.cuda.is_available():
        # Check available GPUs and select the one with most free memory
        device_count = torch.cuda.device_count()
        
        if device_count > 1:
            # If multiple GPUs are available, prioritize GPU 1 and check available memory
            try:
                # Get available memory for each GPU
                gpu0_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0) - torch.cuda.memory_reserved(0)
                gpu1_free = torch.cuda.get_device_properties(1).total_memory - torch.cuda.memory_allocated(1) - torch.cuda.memory_reserved(1)
                
                # Calculate memory usage percentage
                gpu0_total = torch.cuda.get_device_properties(0).total_memory
                gpu1_total = torch.cuda.get_device_properties(1).total_memory
                gpu0_usage_percent = (gpu0_total - gpu0_free) / gpu0_total * 100
                gpu1_usage_percent = (gpu1_total - gpu1_free) / gpu1_total * 100
                
                print(f"GPU 0 memory: {gpu0_free / 1024**2:.1f}MB free of {gpu0_total / 1024**2:.1f}MB ({gpu0_usage_percent:.1f}% used)")
                print(f"GPU 1 memory: {gpu1_free / 1024**2:.1f}MB free of {gpu1_total / 1024**2:.1f}MB ({gpu1_usage_percent:.1f}% used)")
                
                # Prioritize GPU 1 if it has at least 500MB free memory
                if gpu1_free > 500 * 1024 * 1024:  # 500MB threshold
                    return "cuda:1"
                # Fall back to GPU 0 if it has sufficient memory
                elif gpu0_free > 1000 * 1024 * 1024:  # 1GB threshold
                    return "cuda:0"
                # If both GPUs are low on memory, use the one with more free memory
                elif gpu1_free > gpu0_free:
                    return "cuda:1"
                else:
                    return "cuda:0"
            except Exception as e:
                print(f"Error checking GPU memory: {e}")
                # If memory check fails, default to GPU 1 if available
                return "cuda:1"
        else:
            return "cuda:0"
    # Check for TPU
    elif 'COLAB_TPU_ADDR' in os.environ and hasattr(torch, 'tpu') and torch.tpu.is_available():
        return "tpu"  # Google TPU
    # Check for Intel NPU
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        return "npu"  # Intel Neural Processing Units
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"  # Intel GPUs
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        # Check if MPS (Metal Performance Shaders) is actually working properly
        try:
            # Create a small test tensor to verify MPS is working
            test_tensor = torch.ones(1, device="mps")
            # If we get here, MPS is working properly
            return "mps"  # Apple Silicon
        except Exception:
            # If MPS initialization fails, fall back to CPU
            print("Warning: MPS (Metal) acceleration available but failed initialization. Falling back to CPU.")
            return "cpu"
    else:
        return "cpu"

# Function to optimize memory usage for GPU
def optimize_memory():
    # Set PyTorch CUDA memory allocation strategy with expandable segments to avoid fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:16"
    
    # Set environment variable to enable garbage collection
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    
    # Force garbage collection before getting device
    import gc
    gc.collect()
    
    device = get_device()
    print(f"Selected device: {device}")
    
    if device.startswith("cuda"):
        # Enable memory optimization for CUDA
        torch.cuda.empty_cache()
        
        # Get device index
        device_idx = int(device.split(":")[1]) if ":" in device else 0
        
        # Set memory allocation strategy
        if hasattr(torch.cuda, 'memory_stats'):
            # Use 70% of available memory for better stability (reduced from 85%)
            torch.cuda.set_per_process_memory_fraction(0.7, device_idx)
            
        # Enable memory efficient optimizations
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
        # Additional memory optimizations
        torch.cuda.empty_cache()
        gc.collect()
        
        # Print memory usage after optimization
        if hasattr(torch.cuda, 'memory_allocated'):
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
            print(f"CUDA Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved on {device}")
    elif device == "tpu":
        # Enable memory optimization for TPU if available
        if hasattr(torch, 'tpu') and hasattr(torch.tpu, 'empty_cache'):
            torch.tpu.empty_cache()
        # Set TPU-specific optimizations if available
        if hasattr(torch, 'tpu') and hasattr(torch.tpu, 'experimental'):
            # Enable any available TPU optimizations
            try:
                torch.tpu.experimental.embedding_checkpoint = True
            except:
                pass
    elif device == "npu":
        # Enable memory optimization for Intel NPU
        if hasattr(torch, 'npu') and hasattr(torch.npu, 'empty_cache'):
            torch.npu.empty_cache()
        # Set memory allocation strategy if available
        if hasattr(torch.npu, 'memory_stats'):
            torch.npu.set_per_process_memory_fraction(0.85)  # Use up to 85% of available memory for better stability
        # Enable NPU-specific optimizations if available
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'npu'):
            try:
                torch.backends.npu.matmul.allow_tf32 = True  # Allow TF32 for faster computation if available
            except:
                pass
    elif device == "xpu":
        # Enable memory optimization for Intel XPU
        if hasattr(torch, 'xpu') and hasattr(torch.xpu, 'empty_cache'):
            torch.xpu.empty_cache()
        # Enable XPU-specific optimizations if available
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'xpu'):
            try:
                # Enable any available XPU optimizations
                if hasattr(torch.backends.xpu, 'matmul'):
                    torch.backends.xpu.matmul.allow_tf32 = True
            except:
                pass
    elif device == "mps":
        # Enable memory optimization for Apple Silicon MPS
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        # Set MPS-specific optimizations
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps'):
            try:
                # Enable any available MPS optimizations
                torch.backends.mps.enable_procedural_optimization = True
            except:
                pass
    return device

# Set environment variables to optimize performance
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Enable optimizations for transformer models
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

class Available(Enum):
    # Initialize device and optimize memory
    device = optimize_memory()
    # Use mixed precision for faster inference on GPU
    dtype = torch.float16 if device != "cpu" and not device.startswith("cpu") else torch.float32
    
    # Define model configuration with memory optimizations
    model_config = {
        "low_cpu_mem_usage": True,
        "device_map": "auto",  # Let transformers decide optimal device mapping
        "offload_folder": "offload_folder",  # Enable disk offloading if needed
        "offload_state_dict": True,  # Enable state dict offloading
        "torch_dtype": dtype,  # Include dtype in model_kwargs instead of as separate parameter
    }
    
    # When using device_map="auto" in model_kwargs, we should not specify device parameter
    # Also, don't use torch_dtype as a separate parameter when it's in model_kwargs
    llamai_3_2 = pipeline(
        "text-generation", 
        model="meta-llama/Llama-3.2-3B-Instruct",
        model_kwargs=model_config,
        # Use BetterTransformer for faster inference when available
        use_fast=True
    )
    llamai_3_2_think = llamai_3_2

def produce_output_llamai_3_2(prompt, conversations=None):
    if conversations is None:
        conversations = []
    content = conversations
    content.append({'role': 'user', 'content': prompt})
    generator = Available.llamai_3_2.value
    return generator(content, max_length=1500, do_sample=True, temperature=0.5)[0]["generated_text"][-1]['content']

def produce_output_llamai_3_2_think(prompt, conversations=None, instructions = ''):
    generator = Available.llamai_3_2_think.value
    if conversations is None:
        conversations = []
    content = conversations
    content.append([{'role': 'user', 'content': f"How would you complete the following request / answer this prompt in up to 2 steps: ``{prompt}``? Please number them, and add `----` between each step. Please also detail on each step."}])
    if instructions:
        content.insert(0, {'role': 'system', 'content': instructions})

    raw_steps = generator(content, max_length=500, do_sample=True, temperature=0.5)[0]
    
    steps = raw_steps[0]["generated_text"][-1]['content']
    stepsi = steps.split('----')

    final_message = ''
    for step in stepsi:
        for line in step.split('\n'):
            final_message += '| ' + line
        final_message += '\n\n'
    final_message += '\n'

    for step in stepsi:
        if step == '':
            continue
        content = conversations
        content.append([{'role': 'user', 'content': f"Please complete the following step: ``{step}`` to answer this prompt: ``{prompt}``. The other steps are: ``{steps}``. Your current response is: ``{final_message}``"}])
        if instructions:
            content.insert(0, {'role':'system', 'content': instructions})

        final_message += generator(content, max_length=500, do_sample=True, temperature=0.4)[0][0]["generated_text"][-1]['content']

    return final_message

def produce_output(model: Available, prompt: str):
    if model == Available.llamai_3_2:
        return produce_output_llamai_3_2(prompt)
    elif model == Available.llamai_3_2_think:
        return produce_output_llamai_3_2_think(prompt)

if __name__ == "__main__":
    device = Available.device.value
    print(f"\n=== Hardware Acceleration Status ===")
    print(f"Using device: {device}")
    
    if device.startswith("cuda"):
        # Extract device index from the device string (cuda:0 or cuda:1)
        device_idx = int(device.split(":")[1]) if ":" in device else 0
        print(f"GPU {device_idx}: {torch.cuda.get_device_name(device_idx)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(device_idx)/1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(device_idx)/1024**2:.2f} MB")
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated(device_idx)/1024**2:.2f} MB")
        print(f"Using mixed precision: {Available.dtype.value == torch.float16}")
    elif device == "tpu":
        print(f"Google TPU acceleration active")
        print(f"Using mixed precision: {Available.dtype.value == torch.float16}")
    elif device == "npu":
        print(f"Intel NPU acceleration active")
        if hasattr(torch.npu, 'get_device_name'):
            print(f"NPU: {torch.npu.get_device_name(0)}")
        print(f"Using mixed precision: {Available.dtype.value == torch.float16}")
    elif device == "xpu":
        print(f"Intel GPU acceleration active")
        print(f"Using mixed precision: {Available.dtype.value == torch.float16}")
    elif device == "mps":
        print(f"Apple Silicon acceleration active")
        print(f"Using mixed precision: {Available.dtype.value == torch.float16}")
    else:
        print("No GPU/TPU/NPU detected. Running on CPU.")
        print("Performance will be slower without hardware acceleration.")
    
    print("\n=== Model loaded and ready ===\n")
    
    while True:
        prompt = input("You: ")
        print("AI: " + produce_output(Available.llamai_3_2_think, prompt))
