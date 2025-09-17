from datetime import datetime

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.orm import relationship

from common import Language
from database import Base


class Conversation(Base):
    __tablename__ = "conversations"

    language = Column(Enum(Language), nullable=True)
    id = Column(Integer, primary_key=True, index=True)
    emotion = Column(String, nullable=False)
    user_name = Column(String, nullable=False)
    user_message = Column(String, nullable=False)
    my_name = Column(String, nullable=False)
    my_message = Column(String, nullable=False)
    date = Column(DateTime, default=datetime.now())

    associations = relationship("Association", back_populates="conversation")


class Association(Base):
    __tablename__ = "associations"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    embedding_id = Column(Integer, ForeignKey("embeddings.id"))

    conversation = relationship("Conversation", back_populates="associations")
    embedding = relationship("Embedding", back_populates="associations")


class Embedding(Base):
    __tablename__ = 'embeddings'

    id = Column(Integer, primary_key=True)
    vector = Column(VECTOR(768))

    associations = relationship("Association", back_populates="embedding")
