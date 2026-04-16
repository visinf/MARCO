"""Torch Hub entrypoints for MARCO."""

dependencies = ["torch", "torchvision", "omegaconf", "einops", "PIL", "numpy", "matplotlib"]

import torch

from models import build_marco


_CHECKPOINTS = {
    "marco": {
        "filename": "marco_release.pth",
        "url": "https://github.com/visinf/MARCO/releases/download/v1.0/marco_release.pth",
    },
}


def _load_checkpoint(model, weights, map_location=None, progress=True):
    if weights not in _CHECKPOINTS:
        raise ValueError(
            f"Unknown MARCO weights '{weights}'. "
            f"Available options: {tuple(_CHECKPOINTS)}."
        )

    checkpoint = torch.hub.load_state_dict_from_url(
        _CHECKPOINTS[weights]["url"],
        map_location=map_location,
        progress=progress,
    )
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    return model


def _preprocess_data(*args, **kwargs):
    from util.visualization import preprocess_data

    return preprocess_data(*args, **kwargs)


def _visualize_prediction(*args, **kwargs):
    from util.visualization import visualize_prediction

    return visualize_prediction(*args, **kwargs)


def _attach_hub_helpers(model, weights):
    model.preprocess_data = _preprocess_data
    model.visualize_prediction = _visualize_prediction
    model.available_weights = tuple(_CHECKPOINTS)
    model.weights_name = weights
    return model


def marco(weights="marco", pretrained=True, map_location=None, progress=True, device=None):
    """Load MARCO with optional pretrained weights from GitHub Releases.

    Args:
        weights: The released MARCO checkpoint name. Defaults to ``"marco"``.
        pretrained: If ``True``, load the selected checkpoint.
        map_location: Passed to ``torch.hub.load_state_dict_from_url``.
        progress: Whether to show download progress for the checkpoint.
        device: Device on which to place the model, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        A MARCO ``nn.Module`` with two convenience helpers attached:
        ``model.preprocess_data(...)`` and ``model.visualize_prediction(...)``.
    """
    model = build_marco()
    if pretrained:
        model = _load_checkpoint(model, weights, map_location=map_location, progress=progress)
    if device is not None:
        model = model.to(device)
    return _attach_hub_helpers(model, weights)
