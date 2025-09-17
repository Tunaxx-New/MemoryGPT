from pydantic import confloat
from pydantic_settings import BaseSettings

from common import Language


class GoogleSettings(BaseSettings):
    api_key: str
    max_output_tokens: int


class YandexSettings(BaseSettings):
    api_key: str


class MemorySettings(BaseSettings):
    language: Language
    attention_threshold: confloat(ge=0.0, le=1.0)
    truncation_percentage: confloat(ge=0.0, le=1.0)
    similarity_percentage: confloat(ge=0.0, le=1.0)

    embedding_similarity_percentage: confloat(ge=0.0, le=1.0)
    embedding_top_n: int
