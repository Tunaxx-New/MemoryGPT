from enum import Enum

from pydantic import BaseModel


class Language(Enum):
    RU = 'ru'
    EN = 'en'


class HumanResponse(BaseModel):
    my_name_is: str
    language: Language | None
    emotion: str
    thought: str
    answer: str
    motion: str
    association_words: list[str]


class SingleWord(BaseModel):
    word: str
