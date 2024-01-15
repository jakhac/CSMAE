import omegaconf
from src.utils import omegaconf_select

def add_and_assert_wandb_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    """Adds specific default values/checks for wandb config.

    Args:
        cfg (omegaconf.DictConfig): DictConfig object.

    Returns:
        omegaconf.DictConfig: same as the argument, used to avoid errors.
    """

    cfg.wandb = omegaconf_select(cfg, "wandb", {})
    cfg.wandb.enabled = omegaconf_select(cfg, "wandb.enabled", False)
    cfg.wandb.entity = omegaconf_select(cfg, "wandb.entity", None)
    cfg.wandb.project = omegaconf_select(cfg, "wandb.project", "solo-learn")
    cfg.wandb.offline = omegaconf_select(cfg, "wandb.offline", False)

    return cfg


def parse_cfg(cfg: omegaconf.DictConfig):
    """Add default values to config if not provided correct.

    Args:
        cfg (omegaconf.DictConfig): config

    Returns:
        omegaconf.DictConfig: config
    """

    # default values for wandb
    cfg = add_and_assert_wandb_cfg(cfg)

    # fix number of big/small crops for CSMAEs
    cfg.data.num_large_crops = 1
    cfg.data.num_small_crops = 0

    # extra optimizer kwargs
    cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
    cfg.optimizer.kwargs.betas = omegaconf_select(cfg, "optimizer.kwargs.betas", [0.9, 0.999])

    
    cfg.name = '-'.join([
        'csmae',            
        cfg.backbone.name + '_' + str(cfg.backbone.kwargs.patch_size), 
        cfg.backbone.kwargs.global_pool,
        str(cfg.backbone.kwargs.multi_sensor_encoder_depth)+'multi_enc-' + str(cfg.backbone.kwargs.cross_sensor_encoder_depth)+'x_enc',
        cfg.method_kwargs.decoding_strategy + '_dec',
        cfg.backbone.kwargs.masking_strategy + str(int(cfg.method_kwargs.mask_ratio * 100)) + '_masking',
        cfg.method_kwargs.reconstruction_strategy + '_rec',
        'lr' + str(cfg.optimizer.lr),
        '4rec_loss' if (cfg.method_kwargs.apply_umr_loss and cfg.method_kwargs.apply_cmr_loss) else '2rec_loss' if cfg.method_kwargs.apply_cmr_loss else 'unimodalrec_loss',
        'on_masked',
        'w_MIM' + str(cfg.method_kwargs.mim_temp) + cfg.backbone.kwargs.global_pool if cfg.method_kwargs.apply_mim_loss else 'wo_MIM',
        'w_MDE' if cfg.method_kwargs.apply_mde_loss else 'wo_MDE'
        ])
    
    return cfg
