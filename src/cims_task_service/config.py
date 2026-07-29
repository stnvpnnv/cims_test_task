"""Runtime configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings with a project-specific environment prefix."""

    model_config = SettingsConfigDict(
        env_prefix="CIMS_",
        frozen=True,
    )

    debug: bool = False
