import torch

def get_device():
    """
    Returns the best available device for computation.
    Prioritizes MPS (for Mac Silicon), then CUDA, then CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        return torch.device("cpu")

def get_free_memory_gb(device=None):
    """
    Returns the amount of free memory in GB for the given device.
    For MPS, this is an approximation since PyTorch doesn't provide direct memory info.
    """
    if device is None:
        device = get_device()
    
    if device.type == "cuda":
        memory_stats = torch.cuda.memory_stats(device)
        bytes_active = memory_stats['active_bytes.all.current']
        bytes_reserved = memory_stats['reserved_bytes.all.current']
        bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
        bytes_inactive_reserved = bytes_reserved - bytes_active
        bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
        return bytes_total_available / (1024 ** 3)
    elif device.type == "mps":
        # MPS doesn't have a direct way to query memory
        # Return a conservative estimate (4GB for M1, adjust as needed)
        return 4.0
    else:
        # For CPU, we don't track memory usage
        return float('inf')

def empty_cache(device=None):
    """
    Empties the cache for the given device.
    """
    if device is None:
        device = get_device()
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        # MPS doesn't have a direct equivalent to empty_cache
        # but we can force garbage collection
        import gc
        gc.collect()
