from typing import Any

import numpy as np

from common import HumanResponse
from common import Language
from memory.dtos import AssociationCreateDTO
from memory.dtos import ConversationCreateDTO
from memory.dtos import EmbeddingCreateDTO


def to_association_create_dto(association: str,
                              conversation_id: int,
                              embedding_id: int) -> AssociationCreateDTO:
    return AssociationCreateDTO(
        key = association,
        conversation_id = conversation_id,
        embedding_id = embedding_id
    )


def to_conversation_create_dto(human_response: HumanResponse,
                              user_name: str,
                              user_message: str) -> ConversationCreateDTO:
    return ConversationCreateDTO(
        user_name=user_name,
        user_message=user_message,
        my_message=human_response.answer,
        my_name=human_response.my_name_is,
        emotion=human_response.emotion,
        language=human_response.language
    )


def to_human_response(raw: dict[str, Any]) -> HumanResponse:
    return HumanResponse(
        my_name_is=raw.get("my_name_is", ""),
        language=try_enum(Language, raw.get("language")),
        emotion=raw["emotion"],
        motion=raw['motion'],
        thought=raw["thought"],
        answer=raw["answer"],
        association_words=raw.get("association_words", []),
    )


def try_enum(enum_class, value):
    if value is None:
        return None
    try:
        return enum_class(value)
    except ValueError:
        return None


def to_embedding_create_dto(embedding: np.ndarray) -> EmbeddingCreateDTO:
    return EmbeddingCreateDTO(embedding=embedding.tolist())
