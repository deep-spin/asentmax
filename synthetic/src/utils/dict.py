from omegaconf import DictConfig

def deep_update_dict(original: dict, update: dict) -> dict:
    """
    Recursively update keys of original dict with the values from update dict.
    """
    for key, value in update.items():
        if (
            key in original
            and isinstance(original[key], (dict, DictConfig))
            and isinstance(value, (dict, DictConfig))
        ):
            # If both values are dicts, dive deeper
            deep_update_dict(original[key], value)
        else:
            # Otherwise, overwrite original
            original[key] = value
    return original