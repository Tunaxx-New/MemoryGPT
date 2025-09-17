import re
from datetime import datetime
from datetime import timedelta
from difflib import SequenceMatcher
from typing import Protocol

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from common import HumanResponse
from memory.clients import AttentionClientInterface
from memory.clients import DictionaryClientInterface
from memory.clients import GptClientInterface
from memory.clients import SentenceEmbeddingInterface
from memory.converters import to_association_create_dto
from memory.converters import to_conversation_create_dto
from memory.converters import to_embedding_create_dto
from memory.dtos import ConversationDTO
from memory.repositories import AssociationRepositoryInterface
from memory.settings import MemorySettings


class MemoryServiceInterface(Protocol):
    def chat(self, user_response: HumanResponse) -> HumanResponse:
        raise NotImplementedError

    def get_context(self, embedding_string: str, top_k: int) -> str:
        """Returns context in str, from where info can be got"""
        raise NotImplementedError


class MemoryServiceV1(MemoryServiceInterface):
    def __init__(self, settings: MemorySettings,
                 client: GptClientInterface,
                 association_repository: AssociationRepositoryInterface,
                 attention_client: AttentionClientInterface,
                 dictionary_client: DictionaryClientInterface):
        """
        That memory service works with words association
        """
        self._settings = settings
        self._client = client
        self._repository = association_repository
        self._attention_client = attention_client
        self._dictionary_client = dictionary_client

    def chat(self, user_response: HumanResponse) -> HumanResponse:
        user_name = user_response.my_name_is
        message = user_response.answer
        word_attentions = self._get_word_attentions(message)

        # Find for similarity words associations
        context = ""
        seen_conversation_ids = set()
        for word_attention in word_attentions:
            triggers = [word_attention]
            triggers += self._dictionary_client.synonyms(word_attention, self._settings.language)

            for trigger in triggers:
                associations = self._repository.get_by_key(trigger, self._settings.similarity_percentage)
                for association in associations:
                    if association.conversation_id in seen_conversation_ids:
                        continue

                    seen_conversation_ids.add(association.conversation_id)

        for conversation_id in seen_conversation_ids:
            conversation = self._repository.get_conversation_by_id(conversation_id)
            context += f"[{conversation.date}]({conversation.emotion})"
            context += f"Negotiator({conversation.user_name}): {conversation.user_message}\n"
            context += f"Your name({conversation.my_name}): {conversation.my_message}\n\n"

        human_response = self._client.chat_prompt(context, message)

        # Create conversation and emotion associations
        conversation = self._repository.create_conversation(
            to_conversation_create_dto(human_response, user_name, message))
        self._repository.create_association(
            to_association_create_dto(human_response.emotion, conversation.id, -1))

        # Create associations in answer
        word_attentions += self._get_word_attentions(human_response.answer)
        for word_attention in word_attentions:
            self._repository.create_association(
                to_association_create_dto(word_attention, conversation.id, -1))

        # Create associations in thought
        word_attentions += self._get_word_attentions(human_response.thought)
        for word_attention in word_attentions:
            self._repository.create_association(
                to_association_create_dto(word_attention, conversation.id, -1))

        # Create associations in thought
        word_attentions += self._get_word_attentions(' '.join(human_response.association_words))
        for word_attention in word_attentions:
            self._repository.create_association(
                to_association_create_dto(word_attention, conversation.id, -1))

        return human_response

    def _get_word_attentions(self, message: str) -> list[str]:
        word_attentions = self._attention_client.attention_scores(message)

        # Threshold
        word_attentions = [w for w in word_attentions if w.value >= self._settings.attention_threshold]

        # Sort
        word_attentions.sort(key=lambda w: w.value, reverse=True)

        # Truncate to top percentage
        top_count = max(1, int(len(word_attentions) * self._settings.truncation_percentage))
        return [wa.word for wa in word_attentions[:top_count]]


class MemoryServiceV2(MemoryServiceInterface):
    def __init__(self, settings: MemorySettings,
                 client: GptClientInterface,
                 association_repository: AssociationRepositoryInterface,
                 sentence_embedding: SentenceEmbeddingInterface):
        """
        That memory service associate with embeddings vectors
        """
        self._settings = settings
        self._client = client
        self._repository = association_repository
        self._sentence_embedding_client = sentence_embedding

    def chat(self, user_response: HumanResponse) -> HumanResponse:
        user_name = user_response.my_name_is
        message = user_response.answer
        emotion = user_response.emotion

        context = ""
        seen_conversations = {}
        embeddings = self._get_sentences_embeddings(message, user_name)
        for embedding in embeddings:
            conversations = self._repository.get_similar_embedding(embedding, self._settings.embedding_top_n,
                                                                   self._settings.embedding_similarity_percentage)
            for conversation in conversations:
                if conversation.id in seen_conversations:
                    continue

                # If both messages are too similar
                is_skip = False
                for sc in seen_conversations.values():
                    if SequenceMatcher(None, sc.user_message, conversation.user_message).ratio() >= 0.9 and \
                        SequenceMatcher(None, sc.my_message, conversation.my_message).ratio() >= 0.9:
                        is_skip = True
                        break
                if is_skip:
                    continue

                seen_conversations[conversation.id] = conversation

        for conversation in seen_conversations.values():
            context = self._append_context(context, conversation)

        if self._is_about('вчера', embeddings, 0.9):
            yesterday = datetime.now().date() - timedelta(days=1)
            conversations = self._repository.get_random_by_date(yesterday, self._settings.embedding_top_n)
            for conversation in conversations:
                context += self._append_context(context, conversation)
        if self._is_about('сегодня', embeddings, 0.9):
            today = datetime.now().date()
            conversations = self._repository.get_random_by_date(today, self._settings.embedding_top_n)
            for conversation in conversations:
                context += self._append_context(context, conversation)

        human_response = self._client.chat_prompt(context, f"{user_name}: {message}\n Emotion: {emotion}")

        # My name is
        my_name_is = f'{human_response.my_name_is}: ' if human_response.my_name_is else ''

        # Create conversation and emotion associations with embeddings
        conversation = self._repository.create_conversation(
            to_conversation_create_dto(human_response, user_name, message))
        embedding = self._repository.create_embedding(
            to_embedding_create_dto(self._sentence_embedding_client.get_sentences_embeddings(
                [human_response.emotion])[0]))
        self._repository.create_association(
            to_association_create_dto(human_response.emotion, conversation.id, embedding.id))

        # Save associations with user message and emotion
        self._create_associations(message, conversation.id, user_name)
        self._create_associations(f"{user_name} {emotion}", conversation.id, user_name)

        # Create associations in answer
        self._create_associations(human_response.answer, conversation.id, human_response.my_name_is)

        # Create associations in thought
        self._create_associations(human_response.thought, conversation.id, human_response.my_name_is)

        # Create associations with gpt subjective associations
        self._create_associations(
            '. '.join(f"{s.strip()}" for s in
                      human_response.association_words) + '.', conversation.id, "")

        return human_response

    def _append_context(self, context: str, conversation: ConversationDTO) -> str:
        context += f"[{conversation.date}]({conversation.emotion})"
        context += f"Negotiator({conversation.user_name}): {conversation.user_message}\n"
        context += f"Your name({conversation.my_name}): {conversation.my_message}\n\n"
        return context

    def _create_associations(self, text: str, conversation_id: int, user_name: str):
        embeddings = self._get_sentences_embeddings(text, user_name)
        sentences = self._get_sentences(text)
        for sentence, embedding in zip(sentences, embeddings):
            embedding = self._repository.create_embedding(to_embedding_create_dto(embedding))
            self._repository.create_association(to_association_create_dto(sentence, conversation_id, embedding.id))

    def _get_sentences(self, text: str, user_name: str = "") -> list[str]:
        return [f"{f'{user_name}: ' if user_name else ''}{s.strip()}"
                for s in re.split(r'\.\s*', text) if s.strip()]

    def _get_sentences_embeddings(self, text: str, user_name: str = "") -> np.ndarray:
        return self._sentence_embedding_client.get_sentences_embeddings(self._get_sentences(text, user_name))

    def _is_about(self, about_str: str, embedding: np.ndarray, threshold: float) -> bool:
        sentence_embedding = self._sentence_embedding_client.get_sentences_embeddings([about_str])
        sim = cosine_similarity(embedding, sentence_embedding)[0][0]
        return sim >= threshold
