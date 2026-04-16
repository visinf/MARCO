from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf


_CONFIGS_DIR = Path(__file__).parent / "configs"
_COMMON_CFG_DEFAULT = _CONFIGS_DIR / "common.yaml"
_TRAIN_CFG_DEFAULT = _CONFIGS_DIR / "train" / "train.yaml"
_MODEL_CFG_DEFAULT = _CONFIGS_DIR / "model" / "marco.yaml"
_DATASET_CFG_DIR = _CONFIGS_DIR / "dataset"


def _load_optional_config(path: Path):
    return OmegaConf.load(path) if path.is_file() else OmegaConf.create()


def _merge_cli_subconfig(base_cfg, cli_cfg, key: str):
    """Merge CLI overrides into a sub-config.
    """
    base_keys = set(OmegaConf.to_container(base_cfg, resolve=True).keys())

    # Flat overrides: any CLI key whose name matches a key in this sub-config
    flat = {k: v for k, v in OmegaConf.to_container(cli_cfg, resolve=True).items()
            if k in base_keys}
    if flat:
        base_cfg = OmegaConf.merge(base_cfg, OmegaConf.create(flat))

    # Namespaced overrides: e.g. model_cfg.n_upscale=3 (applied last, takes priority)
    if key in cli_cfg:
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg[key])

    return base_cfg


def load_model_config(args, cli_cfg=None, config_path=None):
    cli_cfg = cli_cfg or OmegaConf.from_cli()
    model_cfg = _load_optional_config(Path(config_path) if config_path else _MODEL_CFG_DEFAULT)
    args.model_cfg = _merge_cli_subconfig(model_cfg, cli_cfg, "model_cfg")
    return args


def load_dataset_config(args, cli_cfg=None):
    cli_cfg = cli_cfg or OmegaConf.from_cli()
    cfg_path = _DATASET_CFG_DIR / f"{args.dataset}.yaml"
    dataset_cfg = _load_optional_config(cfg_path)
    args.dataset_cfg = _merge_cli_subconfig(dataset_cfg, cli_cfg, "dataset_cfg")
    return args


def load_eval_config(args):
    cli_cfg = OmegaConf.from_cli()
    common_cfg = _load_optional_config(_COMMON_CFG_DEFAULT)
    base = OmegaConf.merge(common_cfg, OmegaConf.create(vars(args)), cli_cfg)
    # Write back merged values to args
    for k, v in OmegaConf.to_container(base, resolve=True).items():
        if not isinstance(v, dict):
            setattr(args, k, v)
    load_model_config(args, cli_cfg)
    load_dataset_config(args, cli_cfg)
    return args


def load_train_config(args, config_path=None):
    cli_cfg = OmegaConf.from_cli()

    common_cfg = _load_optional_config(_COMMON_CFG_DEFAULT)
    base_cfg = OmegaConf.create(vars(args))
    train_cfg = _load_optional_config(Path(config_path) if config_path else _TRAIN_CFG_DEFAULT)

    cfg = OmegaConf.merge(common_cfg, base_cfg, train_cfg, cli_cfg)

    load_model_config(cfg, cli_cfg)
    load_dataset_config(cfg, cli_cfg)

    return cfg


def get_args_parser():
    parser = ArgumentParser("MARCO training and evaluation.", add_help=False)


    parser.add_argument(
        "--checkpoint", 
        default="", 
        help="load checkpoint for inference"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="spair",
        type=str,
        help="Dataset name",
        choices=["spair", "spair-u", "pf-pascal", "ap-10k", "mp-100"],
    )
    return parser
