import configparser
from pathlib import Path
from typing import Any, Optional


class CfgError(ValueError):
    """Raised when configuration data is missing or invalid."""


def cfg_bool(cfg_value: Any) -> bool:
    cfg_text = str(cfg_value).strip().lower()
    if cfg_text in {"1", "true", "yes", "y", "on"}:
        return True
    if cfg_text in {"0", "false", "no", "n", "off"}:
        return False
    raise CfgError(f"Invalid boolean value: {cfg_value}")


def cfg_load(cfg_path: str) -> configparser.ConfigParser:
    cfg_file = Path(cfg_path)
    if not cfg_file.is_file():
        raise CfgError(
            f"Config file not found: {cfg_file}. "
            "Create it from config/config-example.ini."
        )

    cfg_data = configparser.ConfigParser()
    cfg_data.read(cfg_file)
    return cfg_data


def cfg_get(
    cfg_data: configparser.ConfigParser,
    cfg_section: str,
    cfg_key: str,
    cfg_default: Optional[str] = None,
) -> str:
    if cfg_data.has_option(cfg_section, cfg_key):
        return cfg_data.get(cfg_section, cfg_key)
    if cfg_default is not None:
        return cfg_default
    raise CfgError(f"Missing config value: [{cfg_section}] {cfg_key}")


def cfg_get_int(
    cfg_data: configparser.ConfigParser,
    cfg_section: str,
    cfg_key: str,
    cfg_default: Optional[int] = None,
) -> int:
    cfg_fallback = None if cfg_default is None else str(cfg_default)
    cfg_value = cfg_get(cfg_data, cfg_section, cfg_key, cfg_fallback)
    try:
        return int(cfg_value)
    except ValueError as cfg_err:
        raise CfgError(
            f"Config value [{cfg_section}] {cfg_key} must be an integer"
        ) from cfg_err


def cfg_get_float(
    cfg_data: configparser.ConfigParser,
    cfg_section: str,
    cfg_key: str,
    cfg_default: Optional[float] = None,
) -> float:
    cfg_fallback = None if cfg_default is None else str(cfg_default)
    cfg_value = cfg_get(cfg_data, cfg_section, cfg_key, cfg_fallback)
    try:
        return float(cfg_value)
    except ValueError as cfg_err:
        raise CfgError(
            f"Config value [{cfg_section}] {cfg_key} must be a number"
        ) from cfg_err


def cfg_get_bool(
    cfg_data: configparser.ConfigParser,
    cfg_section: str,
    cfg_key: str,
    cfg_default: Optional[bool] = None,
) -> bool:
    cfg_fallback = None
    if cfg_default is not None:
        cfg_fallback = "true" if cfg_default else "false"
    return cfg_bool(cfg_get(cfg_data, cfg_section, cfg_key, cfg_fallback))


def cfg_pick(cli_value: Any, cfg_value: Any) -> Any:
    return cfg_value if cli_value is None else cli_value


def cfg_list(cli_values: Optional[list[str]], cfg_value: str) -> list[str]:
    if cli_values:
        return cli_values
    return [cfg_item.strip() for cfg_item in cfg_value.split(",") if cfg_item.strip()]
