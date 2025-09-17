from pydantic import BaseModel
from datetime import datetime

from pydantic import ConfigDict

from common import Language


class AssociationCreateDTO(BaseModel):
    key: str
    conversation_id: int
    embedding_id: int


class AssociationDTO(AssociationCreateDTO):
    model_config = ConfigDict(from_attributes=True)

    id: int


class ConversationCreateDTO(BaseModel):
    user_name: str
    user_message: str
    my_message: str
    my_name: str
    emotion: str
    language: Language | None


class ConversationDTO(ConversationCreateDTO):
    model_config = ConfigDict(from_attributes=True)

    id: int
    date: datetime


class AttentionWord(BaseModel):
    word: str
    value: float


class EmbeddingCreateDTO(BaseModel):
    embedding: list[float]


class EmbeddingDTO(EmbeddingCreateDTO):
    model_config = ConfigDict(from_attributes=True)

    id: int
