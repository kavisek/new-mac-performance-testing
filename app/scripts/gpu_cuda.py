import torch

# Check if CUDA (GPU support) is available in PyTorch
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# Print the name of the available GPU (if any)
if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU.")
