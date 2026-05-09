"""Runtime configuration loaded from TELETOP_* environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TELETOP_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: Path = Path.home() / "teletop"
    web_dist_dir: Path = Path("/home/talat/teletop/web/dist")
    # Becomes mandatory in Task 12 (auth layer).
    auth_token: str | None = None


def get_settings() -> Settings:
    return Settings()
