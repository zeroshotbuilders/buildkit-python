from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus, unquote, urlparse

from .application_config import ApplicationConfig
from .config_utils import load_config


@dataclass(frozen=True, slots=True)
class PostgresConnectionConfig:
    host: str
    port: int
    username: str
    password: str
    database: str
    logging: bool = False
    standard_conforming_strings: bool = True
    ignore_client_min_messages: bool = False
    pool_max: int | None = None
    pool_acquire: int | None = None
    pool_idle: int | None = None

    POSTGRES_CONFIG_KEY = "postgres"

    @classmethod
    def from_url(cls, url: str) -> PostgresConnectionConfig:
        parsed = urlparse(url)
        if parsed.scheme not in ("postgresql", "postgres"):
            raise ValueError(
                f"Unsupported scheme '{parsed.scheme}', expected 'postgresql' or 'postgres'"
            )
        return cls(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            username=unquote(parsed.username or ""),
            password=unquote(parsed.password or ""),
            database=(parsed.path or "/").lstrip("/"),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PostgresConnectionConfig:
        return cls(
            host=str(data["host"]),
            port=int(data["port"]),
            username=str(data["username"]),
            password=str(data["password"]),
            database=str(data["database"]),
            logging=bool(data.get("logging", False)),
            standard_conforming_strings=bool(data.get("standardConformingStrings", True)),
            ignore_client_min_messages=bool(data.get("ignoreClientMinMessages", False)),
            pool_max=int(data["poolMax"]) if data.get("poolMax") is not None else None,
            pool_acquire=(
                int(data["poolAcquire"]) if data.get("poolAcquire") is not None else None
            ),
            pool_idle=int(data["poolIdle"]) if data.get("poolIdle") is not None else None,
        )

    @classmethod
    def from_application_config(
        cls,
        application_config: ApplicationConfig,
    ) -> PostgresConnectionConfig:
        if not application_config.application_root:
            raise ValueError("application_root is required to load postgres config")
        config: dict[str, Any] = load_config(
            application_config.application_root,
            cls.POSTGRES_CONFIG_KEY,
        )
        return cls.from_mapping(config)

    @property
    def url(self) -> str:
        return (
            "postgresql://"
            f"{quote_plus(self.username)}:{quote_plus(self.password)}@"
            f"{self.host}:{self.port}/{self.database}"
        )

    def sqlalchemy_url(self, driver: str = "asyncpg") -> str:
        return (
            "postgresql+"
            f"{driver}://{quote_plus(self.username)}:{quote_plus(self.password)}@"
            f"{self.host}:{self.port}/{self.database}"
        )
