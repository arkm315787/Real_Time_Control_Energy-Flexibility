"""Runtime settings for external market services."""

from __future__ import annotations

import os
from dataclasses import dataclass

from flexihome.core.market_data import ENTSOE_API_URL, FINGRID_CURRENT_API_URL, load_local_env


@dataclass(frozen=True)
class AppSettings:
    """Configuration shared by service-layer adapters."""

    entsoe_api_key: str | None = None
    fingrid_api_key: str | None = None
    entsoe_api_url: str = ENTSOE_API_URL
    fingrid_api_url: str = FINGRID_CURRENT_API_URL
    market_data_cache_ttl_seconds: int = 900
    market_data_cache_dir: str = ".cache/market_data"
    default_market_lookback_days: int = 90

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "AppSettings":
        load_local_env(env_path)
        return cls(
            entsoe_api_key=os.getenv("ENTSOE_API_KEY") or os.getenv("ENTSOE_SECURITY_TOKEN"),
            fingrid_api_key=os.getenv("FINGRID_API_KEY") or os.getenv("FINGRID_OPENDATA_API_KEY"),
            market_data_cache_ttl_seconds=int(os.getenv("COVERLY_MARKET_CACHE_TTL_SECONDS", "900")),
            market_data_cache_dir=os.getenv("COVERLY_MARKET_CACHE_DIR", ".cache/market_data"),
            default_market_lookback_days=int(os.getenv("COVERLY_MARKET_LOOKBACK_DAYS", "90")),
        )
