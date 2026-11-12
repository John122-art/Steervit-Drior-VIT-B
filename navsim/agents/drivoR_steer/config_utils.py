"""Small helpers shared by the Steer-DrivoR modules."""

def cfg_value(node, key, default):
    """node[key] if present else default; safe for Hydra struct-mode DictConfig and plain dicts."""
    return node[key] if node is not None and key in node else default
