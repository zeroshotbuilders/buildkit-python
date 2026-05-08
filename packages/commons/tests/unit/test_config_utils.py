from __future__ import annotations

import json
from pathlib import Path

from zeroshot_commons import (
    ApplicationConfig,
    deep_merge,
    load_config,
    parse_env_variables,
    run_with_env,
)


def test_deep_merge_recursively_overrides_values() -> None:
    assert deep_merge(
        {"a": {"b": 1, "c": 2}, "x": "base"},
        {"a": {"c": 3, "d": 4}, "y": "override"},
    ) == {
        "a": {"b": 1, "c": 3, "d": 4},
        "x": "base",
        "y": "override",
    }


def test_parse_env_variables_supports_nested_numeric_and_boolean_values() -> None:
    parsed = run_with_env(
        lambda: parse_env_variables(),
        [
            ("app_service___port_numeric", "8080"),
            ("app_service___enabled_boolean", "true"),
            ("app_service___name", "demo"),
        ],
    )

    assert parsed == {
        "service": {
            "port": 8080,
            "enabled": True,
            "name": "demo",
        }
    }


def test_load_config_reads_json_and_applies_env_overrides(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    main_dir = package_root / "src"
    assets_dir = package_root / "assets"
    main_dir.mkdir(parents=True)
    assets_dir.mkdir()
    (assets_dir / "config.json").write_text(
        json.dumps(
            {
                "local": False,
                "port": 3000,
                "service": {"host": "base", "port": 1234},
            }
        ),
        encoding="utf-8",
    )

    config = run_with_env(
        lambda: load_config(
            str(main_dir),
            config_file_path="assets/config.json",
        ),
        [
            ("app_service___port_numeric", "8080"),
            ("app_local_boolean", "true"),
        ],
    )

    assert config["local"] is True
    assert config["service"] == {"host": "base", "port": 8080}


def test_load_config_applies_database_url_override(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    main_dir = package_root / "src"
    assets_dir = package_root / "assets"
    main_dir.mkdir(parents=True)
    assets_dir.mkdir()
    (assets_dir / "config.json").write_text(
        json.dumps(
            {
                "postgres": {
                    "host": "old-host",
                    "port": 5432,
                    "username": "old",
                    "password": "old",
                    "database": "old_db",
                }
            }
        ),
        encoding="utf-8",
    )

    config = run_with_env(
        lambda: load_config(
            str(main_dir),
            config_file_path="assets/config.json",
        ),
        [("DATABASE_URL", "postgresql://newuser:newpass@newhost:6543/new_db")],
    )

    assert config["postgres"]["host"] == "newhost"
    assert config["postgres"]["port"] == 6543
    assert config["postgres"]["username"] == "newuser"
    assert config["postgres"]["password"] == "newpass"
    assert config["postgres"]["database"] == "new_db"


def test_load_config_reads_sub_config_and_application_config(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    main_dir = package_root / "src"
    assets_dir = package_root / "assets"
    main_dir.mkdir(parents=True)
    assets_dir.mkdir()
    (assets_dir / "config.json").write_text(
        json.dumps(
            {
                "local": True,
                "port": 9000,
                "useRemoteSecrets": True,
                "service": {"name": "demo"},
            }
        ),
        encoding="utf-8",
    )

    service_config: dict[str, object] = load_config(
        str(main_dir),
        config_key="service",
        config_file_path="assets/config.json",
    )
    assert service_config == {"name": "demo"}

    application_config = ApplicationConfig.from_root(str(main_dir))
    assert application_config.local is False
    assert application_config.port == 0

    created = ApplicationConfig.create(local=True, port=8080, application_root=str(main_dir))
    assert created.local is True
    assert created.port == 8080
