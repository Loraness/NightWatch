from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "NightWatch"
    DEBUG: bool = False


    POSTGRES_USER: str = 'postgres'
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str = 'localhost'
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "nightwatch_db"

    REDIS_URL: str = "redis://localhost:6379/0"

    @computed_field
    @property
    def async_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
settings = Settings()

