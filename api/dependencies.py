from fastapi import Depends

from database import Database
from memory.clients import AttentionClientInterface
from memory.clients import AttentionClientV1
from memory.clients import DictionaryClientInterface
from memory.clients import GeminiClient
from memory.clients import GptClientInterface
from memory.clients import SentenceEmbeddingInterface
from memory.clients import SentenceEmbeddingV1
from memory.clients import YandexDictionaryClient
from memory.repositories import AssociationRepositoryInterface
from memory.repositories import AssociationRepositoryV1
from memory.services import MemoryServiceInterface
from memory.services import MemoryServiceV1
from memory.services import MemoryServiceV2
from settings import Settings


def get_settings() -> Settings:
    return Settings()  # noqa


def get_database(settings: Settings = Depends(get_settings)) -> Database:
    return Database(settings.database)


def get_association_repository_v1(database: Database = Depends(get_database)) -> AssociationRepositoryInterface:
    return AssociationRepositoryV1(database.session)


def get_gpt_client(settings: Settings = Depends(get_settings)) -> GptClientInterface:
    return GeminiClient(settings.google, "default", 0.5)


def get_attention_client_v1() -> AttentionClientInterface:
    return AttentionClientV1()


def get_dictionary_client(settings: Settings = Depends(get_settings)) -> DictionaryClientInterface:
    return YandexDictionaryClient(settings.yandex)


def get_sentence_embedding_client() -> SentenceEmbeddingInterface:
    return SentenceEmbeddingV1()


def get_association_service_v1(
        settings: Settings = Depends(get_settings),
        client: GptClientInterface = Depends(get_gpt_client),
        repository: AssociationRepositoryInterface = Depends(get_association_repository_v1),
        attention_client: AttentionClientInterface = Depends(get_attention_client_v1),
        dictionary_client: DictionaryClientInterface = Depends(get_dictionary_client),
        sentence_embedding_client: SentenceEmbeddingInterface = Depends(get_sentence_embedding_client)) -> MemoryServiceInterface:
    return MemoryServiceV1(settings.memory, client, repository, attention_client, dictionary_client, sentence_embedding_client)


def get_association_service_v2(
        settings: Settings = Depends(get_settings),
        client: GptClientInterface = Depends(get_gpt_client),
        repository: AssociationRepositoryInterface = Depends(get_association_repository_v1),
        sentence_embedding_client: SentenceEmbeddingInterface = Depends(get_sentence_embedding_client)) -> MemoryServiceInterface:
    return MemoryServiceV2(settings.memory, client, repository, sentence_embedding_client)
