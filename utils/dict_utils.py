"""Small dictionary helpers used across the project."""

def prefix_dict_keys(data: dict, prefix: str) -> dict:
    """Return a copy of data with keys prefixed by '<prefix>/'."""
    return {f"{prefix}/{key}" if prefix else key: value for key, value in data.items()}
