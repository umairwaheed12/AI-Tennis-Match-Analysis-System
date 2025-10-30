def lower_name_map(names):
    try:
        return {int(i): str(n).lower() for i, n in names.items()}
    except Exception:
        return {}

def resolve_index_by_alias(name_map, aliases, default_idx):
    for i, n in name_map.items():
        if any(alias in n for alias in aliases):
            return i
    return default_idx
