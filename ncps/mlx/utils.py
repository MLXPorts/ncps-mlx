import mlx.core as mx
import json
from pathlib import Path
from typing import Union, Dict, Any

def save_model(model: Any, filepath: Union[str, Path]) -> None:
    """Save an MLX model to disk.
    
    Args:
        model: The MLX model to save
        filepath: Path to save the model to (.json extension recommended)
    """
    filepath = Path(filepath)
    state = model.state_dict()
    
    # Convert MLX arrays to lists for JSON serialization
    def convert_arrays(obj):
        if isinstance(obj, mx.array):
            return {
                "__mlx_array__": True,
                "data": obj.tolist(),
                "dtype": str(obj.dtype)
            }
        elif isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_arrays(item) for item in obj]
        return obj
    
    serializable_state = convert_arrays(state)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_state, f)

def load_model(model: Any, filepath: Union[str, Path]) -> None:
    """Load an MLX model from disk.
    
    Args:
        model: The MLX model to load into
        filepath: Path to load the model from
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        state = json.load(f)
    
    # Convert JSON lists back to MLX arrays
    def convert_to_arrays(obj):
        if isinstance(obj, dict):
            if "__mlx_array__" in obj:
                return mx.array(obj["data"], dtype=obj["dtype"])
            return {k: convert_to_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_arrays(item) for item in obj]
        return obj
    
    state = convert_to_arrays(state)
    model.load_state_dict(state)