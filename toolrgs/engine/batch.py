"""Distributed batch-size helpers."""


def per_process_batch_size(global_batch_size, world_size, field_name):
    """Convert one configured global batch size to a per-process batch size."""
    global_batch_size = int(global_batch_size)
    world_size = int(world_size)
    if global_batch_size <= 0:
        raise ValueError(
            f"TRAIN.{field_name} must be positive, got {global_batch_size}."
        )
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if global_batch_size % world_size:
        raise ValueError(
            f"TRAIN.{field_name} is a global batch size and must be divisible "
            f"by world size: {global_batch_size} % {world_size} != 0."
        )
    return global_batch_size // world_size