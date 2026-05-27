import json

from .factory import create_model_from_config
from .utils import load_ckpt_state_dict

import os

from huggingface_hub import hf_hub_download

def get_pretrained_model(path_or_name: str, local: bool = False):
    """
    Get a pretrained model by name.
     - path_or_name: Local path or Hugging Face model name
     - local: Whether to load the model from a local path
    """
    
    
    if not local:
        model_config_path = hf_hub_download(path_or_name, filename="model_config.json", repo_type='model')
        # Try to download the model.safetensors file first, if it doesn't exist, download the model.ckpt file
        try:
            model_ckpt_path = hf_hub_download(path_or_name, filename="model.safetensors", repo_type='model')
        except Exception as e:
            model_ckpt_path = hf_hub_download(path_or_name, filename="model.ckpt", repo_type='model')
    else:
        model_config_path = path_or_name + "/model_config.json"
        assert os.path.exists(model_config_path), f"File not found: {model_config_path}"

        if os.path.exists(path_or_name + "/model.safetensors"):
            model_ckpt_path = path_or_name + "/model.safetensors"
        elif os.path.exists(path_or_name + "/model.ckpt"):
            model_ckpt_path = path_or_name + "/model.ckpt"
        else:
            raise FileNotFoundError(f"File not found: {path_or_name}")

    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)

    model.load_state_dict(load_ckpt_state_dict(model_ckpt_path))

    return model, model_config