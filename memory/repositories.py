import random
from datetime import date
from typing import Protocol

from sqlalchemy import func
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload

from memory.dtos import AssociationCreateDTO
from memory.dtos import AssociationDTO
from memory.dtos import ConversationCreateDTO
from memory.dtos import ConversationDTO
from memory.dtos import EmbeddingCreateDTO
from memory.dtos import EmbeddingDTO
from memory.models import Association
from memory.models import Conversation
from memory.models import Embedding


class AssociationRepositoryInterface(Protocol):
    def create_association(self, create_dto: AssociationCreateDTO) -> AssociationDTO:
        raise NotImplementedError

    def create_conversation(self, create_dto: ConversationCreateDTO) -> ConversationDTO:
        raise NotImplementedError

    def create_embedding(self, create_dto: EmbeddingCreateDTO) -> EmbeddingDTO:
        raise NotImplementedError

    def get_conversation_by_id(self, conversation_id: int) -> ConversationDTO:
        raise NotImplementedError

    def get_by_key(self, key: str, similarity_threshold: float) -> list[AssociationDTO]:
        raise NotImplementedError

    def get_similar_embedding(self, embedding: str, top_n: int, similarity_threshold: float) -> list[ConversationDTO]:
        raise NotImplementedError

    def get_random_by_date(self, conversation_date: date, limit: int) -> list[ConversationDTO]:
        raise NotImplementedError

class AssociationRepositoryV1(AssociationRepositoryInterface):
    def __init__(self, session: Session):
        self._session = session

    def create_association(self, create_dto: AssociationCreateDTO) -> AssociationDTO:
        association = Association(**create_dto.model_dump())
        self._session.add(association)
        self._session.commit()
        self._session.refresh(association)
        return AssociationDTO.model_validate(association)

    def create_embedding(self, create_dto: EmbeddingCreateDTO) -> EmbeddingDTO:
        embedding = Embedding(vector=create_dto.embedding)
        self._session.add(embedding)
        self._session.commit()
        self._session.refresh(embedding)
        return EmbeddingDTO(id=embedding.id, embedding=embedding.vector)

    def create_conversation(self, create_dto: ConversationCreateDTO) -> ConversationDTO:
        conversation = Conversation(**create_dto.model_dump())
        self._session.add(conversation)
        self._session.commit()
        self._session.refresh(conversation)
        return ConversationDTO.model_validate(conversation)

    def get_conversation_by_id(self, conversation_id: int) -> ConversationDTO:
        return ConversationDTO.model_validate(self._session.get(Conversation, conversation_id))

    def get_by_key(self, key: str, similarity_threshold: float) -> list[AssociationDTO]:
        q = self._session.query(Association, func.similarity(Association.key, key).label("sim"))
        q = q.filter(func.similarity(Association.key, key) >= similarity_threshold)
        q = q.order_by(func.similarity(Association.key, key).desc())
        results = []
        for association, sim in q.all():  # unpack tuple
            results.append(AssociationDTO.model_validate(association))
        return results

    def get_similar_embedding(self, embedding: str, top_n: int, similarity_threshold: float) -> list[ConversationDTO]:
        # Prepare vector string for pgvector input
        embedding_str = f"[{', '.join(map(str, embedding))}]"

        # Raw SQL with cosine distance <#>
        sql = text("""
                   SELECT id, vector, vector <#> :embedding AS distance
                   FROM embeddings
                   ORDER BY distance ASC LIMIT :top_n
                   """)

        # Execute and fetch embeddings with similarity scores
        results = self._session.execute(
            sql, {"embedding": embedding_str, "top_n": top_n}
        ).fetchall()

        # Filter by similarity threshold (cosine similarity = 1 - distance)
        valid_ids = [row.id for row in results if (1.0 - row.distance) >= similarity_threshold]

        if not valid_ids:
            return []

        # Fetch Associations + Conversations via ORM
        q = self._session.query(Association)
        q = q.options(joinedload(Association.conversation))
        q = q.filter(Association.embedding_id.in_(valid_ids))
        associations = q.all()

        # Map to DTOs
        return [ConversationDTO.model_validate(association.conversation) for association in associations]

    def get_random_by_date(self, conversation_date: date, limit: int) -> list[ConversationDTO]:
        query = self._session.query(Conversation)
        if conversation_date:
            query = query.filter(func.date(Conversation.date) == conversation_date)

        total = query.count()
        if total == 0:
            return []

        random_indices = random.sample(range(total), min(limit, total))
        conversations = [query.offset(idx).first() for idx in random_indices]
        return [ConversationDTO.model_validate(c) for c in conversations]