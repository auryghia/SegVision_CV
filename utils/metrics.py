def binary_dice(pred, target, epsilon=1e-6, ignore_index=0):
    """
    Compute the Dice coefficient for binary segmentation, ignoring a specific label (e.g., background).

    Args:
        pred (Tensor): predicted mask, values in [0, 1] or bool, shape [H, W] or [1, H, W]
        target (Tensor): ground truth mask, same shape as pred
        epsilon (float): small value to avoid division by zero
        ignore_index (int): label to ignore in the ground truth

    Returns:
        Dice coefficient (float)
    """
    pred = pred.squeeze().bool()
    target = target.squeeze().bool()

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

    mask = target != ignore_index

    pred = pred[mask]
    target = target[mask]

    intersection = (pred & target).sum().float()
    union = pred.sum().float() + target.sum().float()

    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    return dice.item()
