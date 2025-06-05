# MPS Support for Mac Silicon

This codebase has been adapted to support Metal Performance Shaders (MPS) on Mac Silicon chips (M1/M2/M3). This allows you to leverage GPU acceleration on Apple Silicon without requiring CUDA.

## Changes Made

1. Added device detection that automatically selects the best available device:
   - MPS for Mac Silicon
   - CUDA for NVIDIA GPUs
   - CPU as fallback

2. Modified memory management to work with MPS:
   - Replaced CUDA-specific memory functions with device-agnostic alternatives
   - Added conservative memory estimation for MPS (since PyTorch doesn't provide direct memory info for MPS)

3. Fixed hardcoded device references:
   - Removed hardcoded "cuda" device references
   - Made backend selection conditional based on available hardware

4. Updated utility functions:
   - Modified `print_free_mem()` to work with MPS
   - Replaced CUDA-specific cache clearing with device-agnostic alternatives

## Usage

The code will automatically detect and use MPS if available. No changes to your workflow are needed.

## Performance Considerations

When using MPS:

1. **Memory Management**: MPS doesn't provide direct memory usage statistics, so memory management is more conservative.

2. **Performance**: While MPS provides GPU acceleration, it may not match CUDA performance for all operations. Some operations may fall back to CPU.

3. **Compatibility**: Not all PyTorch operations are optimized for MPS yet. If you encounter issues, you can force CPU usage by setting:
   ```python
   import torch
   torch.device = "cpu"  # Force CPU usage
   ```

4. **Debugging**: If you encounter MPS-specific issues, you can disable MPS by setting the environment variable:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1  # Forces CPU fallback for unsupported operations
   ```

## Requirements

- macOS 12.3 or later
- PyTorch 1.12 or later with MPS support
- Apple Silicon Mac (M1/M2/M3)

## Running the Gradio Demo

To run the video generation using Gradio on your Macbook, follow these steps:

1.  **Install dependencies**: Navigate to the root directory of the project in your terminal and install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

    This will install all the necessary libraries, including `diffusers`, `transformers`, `accelerate`, `torch`, `gradio`, and any other dependencies specified in the `requirements.txt` file.

2.  **Run the Gradio demo**: After installing the dependencies, you can run the Gradio demo script. Assuming the main demo file is `demo_gradio.py`, use the following command in your terminal from the project's root directory:

    ```bash
    python demo_gradio.py
    ```

    If `demo_gradio_f1.py` is the intended demo, replace `demo_gradio.py` with `demo_gradio_f1.py`.

This script will start a Gradio web server, and you will be provided with a local URL (usually `http://localhost:7860`) in the terminal output. Open this URL in your web browser to access the video generation interface.
