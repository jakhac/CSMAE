from typing import Any, Callable, Dict, List, Tuple

import omegaconf
import torch
import torch.nn as nn

from src.loss import mde_loss_func, mim_loss_func, mse_loss_func
from src.pos_encoding import generate_2d_sincos_pos_embed

from timm.models.vision_transformer import Block

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import pytorch_lightning as pl


from src.csmae_backbone import (
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)

class CSMAEDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim,
        depth,
        num_heads,
        num_patches,
        patch_size,
        mlp_ratio=4.0,
        decoder_in_chans=None,
        decoding_strategy=None,
    ) -> None:
        super().__init__()

        self.decoding_strategy = decoding_strategy
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(in_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # fixed sin-cos embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.decoder_blocks = nn.Sequential(
            *[
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(embed_dim)

        if self.decoding_strategy == "common":
            # Sensor-Common Decoder
            self.decoder_pred_s1_to_s2 = nn.Linear(embed_dim, patch_size**2 * 10, bias=True)
            self.decoder_pred_s2_to_s1 = nn.Linear(embed_dim, patch_size**2 * 2, bias=True)
        else:
            # Sensor-Specific Decoder
            self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * decoder_in_chans, bias=True)

        # init all weights according to MAE's repo
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = generate_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def flip_mask(_t):
        t = _t.clone()
        return t.int().bitwise_xor_(torch.ones_like(t.int())).float()

    def flip_restore(self, _t):
        t = _t.clone()
        L = self._vit_num_patches
        return torch.abs(t - L + 1)

    def forward(self, x, ids_restore, to_modality):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        if self.decoding_strategy == "common":
            if to_modality == "s1":
                x = self.decoder_pred_s2_to_s1(x)
            elif to_modality == "s2":
                x = self.decoder_pred_s1_to_s2(x)

        else:
            x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


class CSMAE(pl.LightningModule):
    
    _BACKBONES = {
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
    }
    
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
    ):
        """Implements CSMAE (based on MAE: https://arxiv.org/abs/2111.06377).

        Extra cfg settings:
            method_kwargs:
                mask_ratio (float): percentage of image to mask.
                decoder_embed_dim (int): number of dimensions for the embedding in the decoder
                decoder_depth (int) depth of the decoder
                decoder_num_heads (int) number of heads for the decoder
                norm_pix_loss (bool): whether to normalize the pixels of each patch with their
                    respective mean and std for the loss. Defaults to False.
        """

        super().__init__()
        self.cfg = cfg
        

        # backbone related
        self.backbone_args: Dict[str, Any] = cfg.backbone.kwargs
        assert cfg.backbone.name in CSMAE._BACKBONES
        self.base_model: Callable = self._BACKBONES[cfg.backbone.name]
        self.backbone_name: str = cfg.backbone.name

        kwargs = self.backbone_args.copy()
        self.backbone: nn.Module = self.base_model(**kwargs)
        self.features_dim: int = self.backbone.num_features
        
        # training related
        self.max_epochs: int = cfg.max_epochs

        # optimizer related
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.extra_optimizer_args: Dict[str, Any] = cfg.optimizer.kwargs

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.min_lr: float = cfg.scheduler.min_lr
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval


        # loss related
        self.apply_umr_loss = cfg.method_kwargs.apply_umr_loss
        self.apply_cmr_loss = cfg.method_kwargs.apply_cmr_loss
        self.apply_mde_loss = cfg.method_kwargs.apply_mde_loss
        self.apply_mim_loss = cfg.method_kwargs.apply_mim_loss
        self.mim_temp = cfg.method_kwargs.mim_temp

        # masking/reconstruction related
        self.mask_ratio: float = cfg.method_kwargs.mask_ratio
        self.masking_strategy = cfg.backbone.kwargs.masking_strategy
        self.norm_pix_loss: bool = cfg.method_kwargs.norm_pix_loss

        self.reconstruction_strategy = cfg.method_kwargs.reconstruction_strategy
        self.decoding_strategy = cfg.method_kwargs.decoding_strategy

        # gather backbone info from timm
        self._vit_embed_dim: int = self.backbone.pos_embed.size(-1)
        self._vit_patch_size: int = self.backbone_args.get("patch_size", 15)
        self._vit_num_patches: int = self.backbone.patch_embed.num_patches

        self.decoder, self.decoder_s1_to_s2, self.decoder_s2_to_s1 = None, None, None
        self.configure_decoders()

    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        
        backbone_params = [{"name": "backbone", "params": self.backbone.parameters()}]

        if self.decoding_strategy == "common":
            extra_learnable_params = [
                {"name": "decoder", "params": self.decoder.parameters()}
            ]
        elif self.decoding_strategy == "specific":
            extra_learnable_params = [
                {
                    "name": "decoder_s1_to_s2",
                    "params": self.decoder_s1_to_s2.parameters(),
                },
                {
                    "name": "decoder_s2_to_s1",
                    "params": self.decoder_s2_to_s1.parameters(),
                },
            ]

        return backbone_params + extra_learnable_params


    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        learnable_params = self.learnable_params()

        optimizer = torch.optim.AdamW(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        max_warmup_steps = (
            self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
            if self.scheduler_interval == "step"
            else self.warmup_epochs
        )
        max_scheduler_steps = (
            self.trainer.estimated_stepping_batches
            if self.scheduler_interval == "step"
            else self.max_epochs
        )
        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=max_warmup_steps,
                max_epochs=max_scheduler_steps,
                warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                eta_min=self.min_lr,
            ),
            "interval": self.scheduler_interval,
            "frequency": 1,
        }

        return [optimizer], [scheduler]


    def configure_decoders(self):
        """Instantiate CSMAE decoder in a Sensor-common / -specific fashion."""
        
        decoder_embed_dim: int = self.cfg.method_kwargs.decoder_embed_dim
        decoder_depth: int = self.cfg.method_kwargs.decoder_depth
        decoder_num_heads: int = self.cfg.method_kwargs.decoder_num_heads

        args = {
            "in_dim": self.features_dim,
            "embed_dim": decoder_embed_dim,
            "depth": decoder_depth,
            "num_heads": decoder_num_heads,
            "num_patches": self._vit_num_patches,
            "patch_size": self._vit_patch_size,
            "mlp_ratio": 4.0,
        }

        if self.decoding_strategy == "common":
            self.decoder = CSMAEDecoder(**args, decoding_strategy="common")

        elif self.decoding_strategy == "specific":
            self.decoder_s2_to_s1 = CSMAEDecoder(
                **args, decoding_strategy="specific", decoder_in_chans=2
            )
            self.decoder_s1_to_s2 = CSMAEDecoder(
                **args, decoding_strategy="specific", decoder_in_chans=10
            )

    def decode(self, patch_feats, ids_restore, to_modality):
        """Acts as a global decoder function call that is applies correct strategy according to confgiuration.

        Args:
            patch_feats (torch.Tensor)
            ids_restore (torch.Tensor)
            to_modality (str)

        Returns:
            torch.Tensor: original image size of to_modality
        """

        if self.decoding_strategy == "common":
            return self.decoder(patch_feats, ids_restore, to_modality)

        elif self.decoding_strategy == "specific":
            if to_modality == "s1":
                return self.decoder_s2_to_s1(patch_feats, ids_restore, to_modality)
            if to_modality == "s2":
                return self.decoder_s1_to_s2(patch_feats, ids_restore, to_modality)

        else:
            assert False, "Invalid decoding strategy."

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        X = X.to(memory_format=torch.channels_last)

        out = {}

        if self.training:
            forward_dicts = self.backbone(X, self.mask_ratio)

            # Retrieve S1/S2 passed through shared encoder
            feats_s1, patch_feats_s1, mask_s1, ids_restore_s1 = forward_dicts["s1"]
            feats_s2, patch_feats_s2, mask_s2, ids_restore_s2 = forward_dicts["s2"]

            # Cross-Modal Reconstruction
            # Depending on the masking strategy, the ids to restore the original order have to be
            # flipped. Hence, there are multiple cases for masking/reconstruction.
            if self.apply_cmr_loss:
                # Masking: disjoint | random
                if (
                    self.masking_strategy == "disjoint"
                    or self.masking_strategy == "random"
                ):
                    assert self.masking_strategy == "random" or torch.equal(
                        ids_restore_s1, self.flip_restore(ids_restore_s2)
                    )
                    assert self.masking_strategy == "random" or torch.equal(
                        ids_restore_s2, self.flip_restore(ids_restore_s1)
                    )

                    if self.reconstruction_strategy == "identical":
                        pred_s2_from_s1 = self.decode(
                            patch_feats_s1, ids_restore_s2, "s2"
                        )
                        pred_s1_from_s2 = self.decode(
                            patch_feats_s2, ids_restore_s1, "s1"
                        )
                        out.update(
                            {
                                "mask_s1_from_s2": mask_s1,
                                "mask_s2_from_s1": mask_s2,
                            }
                        )

                    elif self.reconstruction_strategy == "disjoint":
                        pred_s2_from_s1 = self.decode(
                            patch_feats_s1, ids_restore_s1, "s2"
                        )
                        pred_s1_from_s2 = self.decode(
                            patch_feats_s2, ids_restore_s2, "s1"
                        )
                        out.update(
                            {
                                "mask_s1_from_s2": mask_s2,
                                "mask_s2_from_s1": mask_s1,
                            }
                        )
                    else:
                        assert False, "Invalid reconstruction strategy."

                # Masking: identical
                elif self.masking_strategy == "identical":
                    assert torch.equal(mask_s1, mask_s2)
                    assert torch.equal(ids_restore_s1, ids_restore_s2)

                    if self.reconstruction_strategy == "identical":
                        pred_s2_from_s1 = self.decode(
                            patch_feats_s1, self.flip_restore(ids_restore_s1), "s2"
                        )
                        pred_s1_from_s2 = self.decode(
                            patch_feats_s2, self.flip_restore(ids_restore_s1), "s1"
                        )
                        out.update(
                            {
                                "mask_s1_from_s2": self.flip_mask(mask_s1),
                                "mask_s2_from_s1": self.flip_mask(mask_s1),
                            }
                        )

                    elif self.reconstruction_strategy == "disjoint":
                        pred_s2_from_s1 = self.decode(
                            patch_feats_s1, ids_restore_s1, "s2"
                        )
                        pred_s1_from_s2 = self.decode(
                            patch_feats_s2, ids_restore_s1, "s1"
                        )
                        out.update(
                            {
                                "mask_s1_from_s2": mask_s1,
                                "mask_s2_from_s1": mask_s1,
                            }
                        )
                    else:
                        assert False, "Invalid reconstruction strategy."

                out.update(
                    {
                        "pred_s1_from_s2": pred_s1_from_s2,
                        "pred_s2_from_s1": pred_s2_from_s1,
                    }
                )

            # Uni-Modal Reconstruction
            if self.apply_umr_loss:
                pred_s1_from_s1 = self.decode(patch_feats_s1, ids_restore_s1, "s1")
                pred_s2_from_s2 = self.decode(patch_feats_s2, ids_restore_s2, "s2")

                out.update(
                    {
                        "pred_s1_from_s1": pred_s1_from_s1,
                        "pred_s2_from_s2": pred_s2_from_s2,
                    }
                )

            if self.masking_strategy == "identical":
                assert torch.equal(
                    mask_s1, mask_s2
                ), "S1/S2 patches are not identically masked. Is this indented?"
                assert torch.equal(
                    ids_restore_s1, ids_restore_s2
                ), "S1/S2 patches are not identically masked. Is this indented?"

            out.update(
                {
                    "feats_s1": feats_s1,
                    "feats_s2": feats_s2,
                    "mask_s1": mask_s1,
                    "mask_s2": mask_s2,
                }
            )

        else:
            # If we are not training, we just forward features.
            forward_dicts = self.backbone(X)
            feats_s1 = forward_dicts["s1"]
            feats_s2 = forward_dicts["s2"]

        out.update(
            {
                "feats_s1": feats_s1,
                "feats_s2": feats_s2,
            }
        )

        return out

    def training_step(self, batch: List[Any], batch_idx: int) -> torch.Tensor:
        """Training step for MAE.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of CSMAE losses.
        """

        imgs = batch[1]
        patch_size = self._vit_patch_size
        
        metrics = {"mask_ratio": self.mask_ratio}

        _, X, _ = batch
        X = [X] if isinstance(X, torch.Tensor) else X
        outs = [self(x) for x in X[: 1]]
        out = {k: [out[k] for out in outs] for k in outs[0].keys()}

        reconstruction_loss = torch.zeros([1, 1], device=imgs[0].device)
        reconstruction_loss_s1_from_s1 = torch.zeros([1, 1], device=imgs[0].device)
        reconstruction_loss_s2_from_s2 = torch.zeros([1, 1], device=imgs[0].device)
        reconstruction_loss_s1_from_s2 = torch.zeros([1, 1], device=imgs[0].device)
        reconstruction_loss_s2_from_s1 = torch.zeros([1, 1], device=imgs[0].device)


        # Add Cross-Modal-Reconstruction loss if specified
        if self.apply_cmr_loss:
            reconstruction_loss_s2_from_s1 += mse_loss_func(
                imgs[0][:, :10, :, :],
                out["pred_s2_from_s1"][0],
                out["mask_s2_from_s1"][0],
                patch_size,
                norm_pix_loss=self.norm_pix_loss,
                num_channels=10,
            )

            reconstruction_loss_s1_from_s2 += mse_loss_func(
                imgs[0][:, 10:, :, :],
                out["pred_s1_from_s2"][0],
                out["mask_s1_from_s2"][0],
                patch_size,
                norm_pix_loss=self.norm_pix_loss,
                num_channels=2,
            )

            reconstruction_loss += (
                reconstruction_loss_s1_from_s2 + reconstruction_loss_s2_from_s1
            )
            metrics["train_reconstruction_loss_s1_from_s2"] = reconstruction_loss_s1_from_s2
            metrics["train_reconstruction_loss_s2_from_s1"] = reconstruction_loss_s2_from_s1

        # Add Uni-Modal-Reconstruction loss if specified
        if self.apply_umr_loss:
            reconstruction_loss_s2_from_s2 += mse_loss_func(
                imgs[0][:, :10, :, :],
                out["pred_s2_from_s2"][0],
                out["mask_s2"][0],
                patch_size,
                norm_pix_loss=self.norm_pix_loss,
                num_channels=10,
            )
            reconstruction_loss_s1_from_s1 += mse_loss_func(
                imgs[0][:, 10:, :, :],
                out["pred_s1_from_s1"][0],
                out["mask_s1"][0],
                patch_size,
                norm_pix_loss=self.norm_pix_loss,
                num_channels=2,
            )

            reconstruction_loss += (
                reconstruction_loss_s1_from_s1 + reconstruction_loss_s2_from_s2
            )
            metrics["train_reconstruction_loss_s1_from_s1"] = reconstruction_loss_s1_from_s1
            metrics["train_reconstruction_loss_s2_from_s2"] = reconstruction_loss_s2_from_s2

        metrics["train_reconstruction_loss"] = reconstruction_loss

        # Add NT-Xent loss if specified
        mim_loss = torch.zeros([1, 1], device=imgs[0].device)
        if self.apply_mim_loss:
            mim_loss = mim_loss_func(out, self.mim_temp)
            metrics["train_mim_loss"] = mim_loss

        # Add MDE if specified
        mde_loss = torch.zeros([1, 1], device=imgs[0].device)
        if self.apply_mde_loss:
            mde_loss = -mde_loss_func(out, self.mim_temp)
            metrics["train_mde_loss"] = mde_loss


        self.log_dict(metrics, on_step=True, sync_dist=True)
        total_loss = reconstruction_loss + mim_loss + mde_loss

        return total_loss
