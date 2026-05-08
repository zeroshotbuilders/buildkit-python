from __future__ import annotations

import copy
import json
import logging
import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar, cast

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def _maybe_load_yaml(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Loading YAML config requires PyYAML to be installed.") from exc

    with config_path.open("r", encoding="utf-8") as file_handle:
        loaded = yaml.safe_load(file_handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected mapping config in {config_path}, got {type(loaded).__name__}")
    return cast(dict[str, Any], loaded)


def _load_raw_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _maybe_load_yaml(config_path)
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as file_handle:
            loaded = json.load(file_handle)
        if not isinstance(loaded, dict):
            raise TypeError(
                f"Expected mapping config in {config_path}, got {type(loaded).__name__}"
            )
        return cast(dict[str, Any], loaded)
    raise ValueError(f"Unsupported config file format: {config_path}")


def _parse_scalar_env_value(key: str, value: str) -> Any:
    if key.endswith("_numeric"):
        return float(value) if "." in value else int(value)
    if key.endswith("_boolean"):
        return value == "true"
    return value


def parse_env_variables(prefix: str = "app_") -> dict[str, Any]:
    parsed_variables: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        LOGGER.info("Environment variable override of %s detected.", key)
        nested_keys = key.removeprefix(prefix).split("___")
        current = parsed_variables

        for index, nested_key in enumerate(nested_keys):
            is_last = index == len(nested_keys) - 1
            normalized_key = nested_key.removesuffix("_numeric").removesuffix("_boolean")
            if is_last:
                current[normalized_key] = _parse_scalar_env_value(nested_key, value)
            else:
                current = cast(dict[str, Any], current.setdefault(normalized_key, {}))
    return parsed_variables


def deep_merge(
    obj1: Mapping[str, Any],
    obj2: Mapping[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(obj1))
    for key, value in obj2.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(cast(Mapping[str, Any], merged[key]), value)
        elif isinstance(value, Mapping):
            merged[key] = deep_merge({}, value)
        else:
            merged[key] = value
    return merged


def load_config(
    main_dir: str,
    config_key: str = "",
    config_file_path: str = "assets/config.yaml",
) -> Any:
    package_root = Path(main_dir).resolve().parent
    config_path = package_root / config_file_path
    config = _load_raw_config(config_path)
    merged_config = deep_merge(config, parse_env_variables())

    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        from .postgres_connection import PostgresConnectionConfig

        pg = PostgresConnectionConfig.from_url(database_url)
        merged_config = deep_merge(
            merged_config,
            {
                "postgres": {
                    "host": pg.host,
                    "port": pg.port,
                    "username": pg.username,
                    "password": pg.password,
                    "database": pg.database,
                }
            },
        )

    if not config_key:
        return merged_config

    sub_config: Any = merged_config
    for key in config_key.split("."):
        if isinstance(sub_config, Mapping) and key in sub_config:
            sub_config = sub_config[key]
        else:
            raise KeyError(f"Config key '{config_key}' does not exist.")

    LOGGER.debug(
        "CONFIG { main_dir: %s, config_key: %s, config_file_path: %s }: %s",
        main_dir,
        config_key,
        config_file_path,
        json.dumps(sub_config),
    )
    return sub_config


@contextmanager
def _temp_env(env_vars: list[tuple[str, str]]) -> Any:
    old_env = os.environ.copy()
    os.environ.update(dict(env_vars))
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def run_with_env[T](runnable: Callable[[], T], env_vars: list[tuple[str, str]]) -> T:
    with _temp_env(env_vars):
        return runnable()
