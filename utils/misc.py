
def prefix_dict_keys(data: dict, prefix: str) -> dict:
    return {f"{prefix}/{key}" if prefix else key: value for key, value in data.items()}

def print_namespaced_dict(data: dict) -> None:
    """
    Prints a dictionary with namespaced keys (e.g., 'rollout/ep_len_mean')
    in a formatted ASCII table grouped by namespaces.
    Floats are formatted to 2 decimal places.
    """
    # Group keys by their namespace prefix
    grouped = {}
    for key, value in data.items():
        if "/" in key:
            namespace, subkey = key.split("/", 1)
        else:
            namespace, subkey = key, ""
        grouped.setdefault(namespace, {})[subkey] = value

    # Format values first (floats to 2 decimals)
    formatted_grouped = {}
    for ns, subdict in grouped.items():
        formatted_grouped[ns] = {
            subkey: str(val)
            for subkey, val in subdict.items()
        }

    # Determine column widths
    max_key_len = max(len(subkey) for ns in formatted_grouped for subkey in formatted_grouped[ns]) + 4
    max_val_len = max(len(val) for ns in formatted_grouped for val in formatted_grouped[ns].values()) + 2

    # Print table
    border = "-" * (max_key_len + max_val_len + 5)
    print(border)
    for ns, subdict in formatted_grouped.items():
        print(f"| {ns + '/':<{max_key_len}} |")
        for subkey, val in subdict.items():
            print(f"|    {subkey:<{max_key_len-4}} | {val:<{max_val_len}}|")
    print(border)