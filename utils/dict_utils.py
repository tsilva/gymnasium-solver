"""Small dictionary helpers used across the project."""

from typing import Any, Dict

def convert_dict_numeric_strings(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert scientific-notation strings back to numeric types (idempotent)."""

    # Iterate over the dictionary items (as list) to avoid 
    # modifying the dictionary size during iteration
    for key, value in list(config_dict.items()):
        # Skip non-string values
        if not isinstance(value, str): continue

        # Parse and coerce to numeric type
        try:
            parsed_value = float(value)
            if parsed_value.is_integer(): parsed_value = int(parsed_value)
            config_dict[key] = parsed_value
        except:
            # Leave value unchanged on parse failure
            pass

    # Return the modified dictionary
    return config_dict

def prefix_dict_keys(data: dict, prefix: str) -> dict:
    """Return a copy of data with keys prefixed by '<prefix>/'."""
    return {f"{prefix}/{key}" if prefix else key: value for key, value in data.items()}
