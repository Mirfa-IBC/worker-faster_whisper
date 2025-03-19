from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel

model_names = [ "medium", "large-v1"]

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

def load_model(selected_model):
    '''
    Load and cache models in parallel
    '''
    for _attempt in range(5):
        while True:
            try:
                loaded_model = WhisperModel(
                    selected_model, device="cuda", compute_type="float16")
            except (AttributeError, OSError):
                continue

            break

    return selected_model, loaded_model


models = {}

with ThreadPoolExecutor() as executor:
    for model_name, model in executor.map(load_model, model_names):
        if model_name is not None:
            models[model_name] = model
