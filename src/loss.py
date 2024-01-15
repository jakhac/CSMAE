import torch

def patchify(imgs: torch.Tensor, patch_size: int, num_channels: int=10) -> torch.Tensor:
    """Patchifies an image according to some patch size. 
    Adapted from https://github.com/facebookresearch/mae.
    NOTE: Modified to accept 10-channel images.

    Args:
        imgs (torch.Tensor): [N, num_channels, H, W] Tensor containing the original images.
        patch_size (int): size of each patch.

    Returns:
        torch.Tensor: [N, Tokens, pixels * pixels * num_channels] Tensor containing the patchified images.
    """

    assert imgs.size(2) == imgs.size(3) and imgs.size(2) % patch_size == 0

    h = w = imgs.size(2) // patch_size
    x = imgs.reshape(shape=(imgs.size(0), num_channels, h, patch_size, w, patch_size))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.size(0), h * w, patch_size**2 * num_channels))
    return x

def mae_loss_func(
    imgs: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    norm_pix_loss: bool = True,
    num_channels: int=10,
) -> torch.Tensor:
    """Computes MAE's loss given batch of images, the decoder predictions, the input mask and respective patch size.
    Adapted from https://github.com/facebookresearch/mae.

    Args:
        imgs (torch.Tensor): [N, num_channels, H, W] Tensor containing the original images.
        pred (torch.Tensor): [N, Tokens, pixels * pixels * num_channels] Tensor containing the predicted patches.
        mask (torch.Tensor): [N, Tokens] Tensor representing a binary mask, where value 1 means masked.
        patch_size (int): size of each patch.
        norm_pix_loss (bool): whether to normalize the pixels of each patch with their respective mean and std.

    Returns:
        torch.Tensor: MAE's loss.
    """

    target = patchify(imgs, patch_size, num_channels)

    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = torch.abs(pred - target)
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss

def mse_loss_func(
    imgs: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    norm_pix_loss: bool = True,
    num_channels: int=10,
) -> torch.Tensor:
    """Computes MAE's loss given batch of images, the decoder predictions, the input mask and respective patch size.
    Adapted from https://github.com/facebookresearch/mae.

    Args:
        imgs (torch.Tensor): [N, num_channels, H, W] Tensor containing the original images.
        pred (torch.Tensor): [N, Tokens, pixels * pixels * num_channels] Tensor containing the predicted patches.
        mask (torch.Tensor): [N, Tokens] Tensor representing a binary mask, where value 1 means masked.
        patch_size (int): size of each patch.
        norm_pix_loss (bool): whether to normalize the pixels of each patch with their respective mean and std.

    Returns:
        torch.Tensor: MAE's loss.
    """

    target = patchify(imgs, patch_size, num_channels)

    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.0e-6) ** 0.5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


def mim_loss_func(out, temp: float=0.07, key: str='feats'):
    """Compute NT-Xent loss on output of CSMAE model.

    Args:
        out (Dict): Dictionary containing keys "feats_s1" and "feats_s2" with corresponding features.
        temp (float, optional): Temperature. Defaults to 0.07.
        key (str, optional): Defaults to 'feats'.

    Returns:
        float: NT-Xent
    """
    
    assert len(out[f'{key}_s1']) == 1
    assert len(out[f'{key}_s2']) == 1

    feats_s1 = out[f'{key}_s1'][0]
    feats_s2 = out[f'{key}_s2'][0]

    feats_s1_norm = torch.norm(feats_s1, dim=1).reshape(-1, 1)
    feats_s2_norm = torch.norm(feats_s2, dim=1).reshape(-1, 1)

    feats_s1_norm = torch.div(feats_s1, feats_s1_norm)
    feats_s2_norm = torch.div(feats_s2, feats_s2_norm)
    feats_s12 = torch.cat((feats_s1_norm, feats_s2_norm))
    feats_s21 = torch.cat((feats_s2_norm, feats_s1_norm))

    similarities = torch.mm(feats_s12, feats_s12.T)
    sim_by_tau = torch.div(similarities, temp)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)

    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(feats_s12, feats_s21), temp))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators,denominators)
    neglog_num_by_den = -torch.log(num_by_den)

    return torch.mean(neglog_num_by_den)

def mde_loss_func(out):
    feats_s1 = out['feats_s1'][0]
    feats_s2 = out['feats_s2'][0]

    feats_s1_norm = torch.norm(feats_s1, dim=1).reshape(-1, 1)
    feats_s2_norm = torch.norm(feats_s2, dim=1).reshape(-1, 1)

    feats_s1_norm = torch.div(feats_s1, feats_s1_norm)
    feats_s2_norm = torch.div(feats_s2, feats_s2_norm)

    pairwise_sims = torch.diag(torch.mm(feats_s1_norm, feats_s2_norm.T))
    return torch.log(1 + torch.exp(pairwise_sims)).mean()