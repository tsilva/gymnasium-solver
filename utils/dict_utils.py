"""Small dictionary helpers used across the project."""

from typing import Any, Dict
from dataclasses import MISSING

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


def dataclass_defaults_dict(cls: type) -> Dict[str, Any]:
    """Collect dataclass default values without instantiation.

    Builds a dictionary of field name -> default value. For fields with a
    default_factory, the factory is invoked to obtain the default instance.
    """
    defaults: Dict[str, Any] = {}
    for f in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        if f.default is not MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
            defaults[f.name] = f.default_factory()  # type: ignore[misc]
    return defaults
