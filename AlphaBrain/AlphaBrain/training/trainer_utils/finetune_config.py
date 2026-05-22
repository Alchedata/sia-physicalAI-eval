"""
Utilities for loading finetune_config.yaml as the primary training config.

Merge order (lowest → highest priority):
  configs/models/<model>.yaml
  < configs/datasets/<dataset>.yaml
  < configs/trainer/<trainer>.yaml
  < train_recipe (if mode.config_yaml is set)
  < finetune_config global sections (environment, seed)
  < mode-derived field mappings
  < mode.framework / mode.datasets / mode.trainer direct overrides
  < mode.extra_args
  < CLI args (applied by caller)
"""
import os
import re

from omegaconf import OmegaConf

from AlphaBrain.training.trainer_utils.trainer_tools import normalize_dotlist_args


def expand_env_vars(value):
    """Expand bash-style ${VAR} / ${VAR:-default} in a string. No-op for non-strings."""
    if not isinstance(value, str):
        return value
    def _replace(m):
        var, default = m.group(1), m.group(3)
        return os.environ.get(var, default if default is not None else "")
    return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(:-(.*?))?\}', _replace, value)


def build_config_from_finetune(finetune_cfg, mode: str):
    """Build an OmegaConf training config from finetune_config.yaml + mode name."""
    all_modes = OmegaConf.to_container(finetune_cfg.modes, resolve=False)
    if mode not in all_modes:
        raise ValueError(f"Mode '{mode}' not found. Available: {list(all_modes.keys())}")

    # Work with a plain dict to avoid OmegaConf misinterpreting bash ${...} syntax
    mode_dict = OmegaConf.to_container(finetune_cfg.modes[mode], resolve=False)
    global_defaults = OmegaConf.to_container(finetune_cfg.get('defaults', {}), resolve=False)

    # ── 1. Base configs (model / dataset / trainer defaults) ──────────────────
    base_cfgs = []
    model_key    = mode_dict.get('model')    or global_defaults.get('model')
    dataset_key  = mode_dict.get('dataset')  or global_defaults.get('dataset')
    trainer_key  = mode_dict.get('trainer_defaults') or global_defaults.get('trainer')
    if model_key:   base_cfgs.append(OmegaConf.load(f"configs/models/{model_key}.yaml"))
    if dataset_key: base_cfgs.append(OmegaConf.load(f"configs/datasets/{dataset_key}.yaml"))
    if trainer_key: base_cfgs.append(OmegaConf.load(f"configs/trainer/{trainer_key}.yaml"))

    # Optional train recipe (backward compat; mode.config_yaml)
    recipe_path = mode_dict.get('config_yaml', '')
    if recipe_path and os.path.exists(recipe_path):
        recipe = OmegaConf.load(recipe_path)
        if '_model_config_' in recipe:
            recipe = OmegaConf.merge(OmegaConf.load(recipe.pop('_model_config_')), recipe)
        if 'defaults' in recipe:
            rd = recipe.pop('defaults')
            if 'model' in rd:
                recipe = OmegaConf.merge(OmegaConf.load(f"configs/models/{rd.model}.yaml"), recipe)
        base_cfgs.append(recipe)

    base = OmegaConf.merge(*base_cfgs) if base_cfgs else OmegaConf.create({})

    # ── 2. Global overrides from finetune_config (environment, seed) ──────────
    # NOTE: 'paths' is intentionally excluded — it's only for path resolution,
    #       not part of the training config, and its bash ${...} values would
    #       break OmegaConf interpolation resolution later.
    global_ov = {}
    for key in ('environment', 'seed'):
        if key in finetune_cfg:
            val = finetune_cfg[key]
            global_ov[key] = OmegaConf.to_container(val, resolve=False) if OmegaConf.is_config(val) else val

    # ── 3. Mode field mappings ─────────────────────────────────────────────────
    mode_ov = {}

    if 'run_id' in mode_dict:
        mode_ov['run_id'] = mode_dict['run_id']

    if 'output_root_dir' in mode_dict:
        mode_ov['output_root_dir'] = mode_dict['output_root_dir']
    elif 'common' in finetune_cfg and 'output_root_dir' in finetune_cfg.common:
        mode_ov['output_root_dir'] = finetune_cfg.common.output_root_dir

    if 'framework_name' in mode_dict:
        mode_ov.setdefault('framework', {})['name'] = mode_dict['framework_name']

    if 'base_vlm' in mode_dict:
        base_vlm = expand_env_vars(mode_dict['base_vlm'])
        # 预训练模型目录统一从环境变量 PRETRAINED_MODELS_DIR 读取
        pretrained_dir = os.environ.get('PRETRAINED_MODELS_DIR', 'data/pretrained_models')
        if not os.path.isabs(base_vlm) and not base_vlm.startswith('./') and not base_vlm.startswith('data/'):
            base_vlm = os.path.join(pretrained_dir, base_vlm)
        mode_ov.setdefault('framework', {}).setdefault('qwenvl', {})['base_vlm'] = base_vlm

    if 'data_root' in mode_dict:
        mode_ov.setdefault('datasets', {}).setdefault('vla_data', {})['data_root_dir'] = expand_env_vars(mode_dict['data_root'])
    if 'dataset_mix' in mode_dict:
        mode_ov.setdefault('datasets', {}).setdefault('vla_data', {})['dataset_mix'] = mode_dict['dataset_mix']

    training = mode_dict.get('training', {})
    for field in ('gradient_accumulation_steps', 'max_train_steps', 'save_interval', 'eval_interval', 'freeze_modules', 'pretrained_checkpoint'):
        if field in training:
            mode_ov.setdefault('trainer', {})[field] = training[field]
    if 'per_device_batch_size' in training:
        mode_ov.setdefault('datasets', {}).setdefault('vla_data', {})['per_device_batch_size'] = training['per_device_batch_size']

    # ── 4. Direct nested overrides (framework / datasets / trainer in mode) ───
    direct_ov = {k: mode_dict[k] for k in ('framework', 'datasets', 'trainer', 'trackers', 'wandb_project', 'wandb_entity', 'is_debug', 'stdp', 'lora') if k in mode_dict}

    def _recursive_expand_env(obj):
        """Recursively expand ${VAR} / ${VAR:-default} in all string values."""
        if isinstance(obj, str):
            return expand_env_vars(obj)
        elif isinstance(obj, dict):
            return {k: _recursive_expand_env(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_recursive_expand_env(v) for v in obj]
        return obj

    direct_ov = _recursive_expand_env(direct_ov)

    # ── 5. Merge everything ───────────────────────────────────────────────────
    cfg = OmegaConf.merge(base, OmegaConf.create(global_ov), OmegaConf.create(mode_ov), OmegaConf.create(direct_ov))

    # ── 6. extra_args ─────────────────────────────────────────────────────────
    extra_args = mode_dict.get('extra_args', [])
    if extra_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(extra_args)))

    return cfg
