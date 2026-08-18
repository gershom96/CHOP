"""Dataset entry points.

Keep heavyweight policy-specific imports lazy so the standalone reward-model
pipeline does not require OmniVLA/Prismatic merely to import a dataset module.
"""

__all__ = ["OmniVLAChopDataset", "VisualNavTformerCHOPDataset"]


def __getattr__(name):
    if name == "OmniVLAChopDataset":
        from .omnivla_chop_dataset import OmniVLAChopDataset
        return OmniVLAChopDataset
    if name == "VisualNavTformerCHOPDataset":
        from .visualnav_transformer_dataset import VisualNavTformerCHOPDataset
        return VisualNavTformerCHOPDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
