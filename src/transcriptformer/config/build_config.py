import json
import os
from omegaconf import OmegaConf


def merge_checkpoint_with_cfg(checkpoint_path: str, base_cfg):
    """Merge checkpoint config.json with a provided base cfg (YAML + overrides).

    The merge order matches the CLI behavior:
    - Start from checkpoint config (mlflow_cfg)
    - Merge base_cfg on top (YAML + any overrides)
    - Set derived paths consistently
    """
    # Load model config from checkpoint
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path) as f:
        model_config = json.load(f)
    mlflow_cfg = OmegaConf.create(model_config)

    # Merge: checkpoint first, then base cfg overrides
    cfg = OmegaConf.merge(mlflow_cfg, base_cfg)

    # Set derived paths
    cfg.model.inference_config.load_checkpoint = os.path.join(checkpoint_path, "model_weights.pt")
    cfg.model.data_config.aux_vocab_path = os.path.join(checkpoint_path, "vocabs")
    cfg.model.data_config.esm2_mappings_path = os.path.join(checkpoint_path, "vocabs")
    cfg.model.inference_config.checkpoint_path = checkpoint_path

    return cfg


