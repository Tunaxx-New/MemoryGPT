from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from database import DatabaseSettings
from memory.settings import GoogleSettings
from memory.settings import MemorySettings
from memory.settings import YandexSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter='__')

    database: DatabaseSettings
    memory: MemorySettings
    google: GoogleSettings
    yandex: YandexSettings
