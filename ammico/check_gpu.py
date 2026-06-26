import torch
import subprocess

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total VRAM: {total_mem:.2f} GB")

    # Get current memory usage
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free",
            "--format=csv,nounits,noheader",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        used, free = map(int, result.stdout.strip().split(","))
        print(f"Used VRAM: {used / 1024:.2f} GB")
        print(f"Free VRAM: {free / 1024:.2f} GB")
else:
    print("No GPU available - will use CPU")

# we need accelerate and transformers
# do we need windows and macos compatibility? How reasonable is it that someone runs this on a workstation?

# The model supports a wide range of resolution inputs. By default, it uses the native resolution for input, but higher resolutions can enhance performance at the cost of more computation. Users can set the minimum and maximum number of pixels to achieve an optimal configuration for their needs, such as a token count range of 256-1280, to balance speed and memory usage.
