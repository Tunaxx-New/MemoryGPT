import json
from typing import Protocol

import google.generativeai as genai
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from transformers import AutoTokenizer

from common import HumanResponse
from common import Language
from common import SingleWord
from memory.converters import to_human_response
from memory.dtos import AttentionWord
from memory.settings import GoogleSettings
from memory.settings import YandexSettings


class GptClientInterface(Protocol):
    def chat_prompt(self, context: str, message: str) -> HumanResponse:
        raise NotImplementedError

    def single_word(self, context: str, message: str) -> SingleWord:
        raise NotImplementedError


class GeminiClient(GptClientInterface):
    def __init__(self, settings: GoogleSettings, system_instruction: str, temperature: float):
        genai.configure(api_key=settings.api_key)
        self._temperature = temperature
        self._system_instruction = system_instruction
        self._max_output_tokens = settings.max_output_tokens

    def chat_prompt(self, context: str, message: str) -> HumanResponse:
        model = genai.GenerativeModel(model_name="gemini-2.5-flash",
                                      generation_config={
                                          "temperature": self._temperature,
                                          "response_mime_type": "application/json",
                                          "response_schema": HumanResponse
                                      },
                                      system_instruction="Answer emotion field with emojies. Be sure to mention a littlge about each message of the background context that is provided in the message. Use emojies that differs from user.")
        chat = model.start_chat()
        response = chat.send_message(f"Background context: {context}\nMessage: {message}")
        print(f"Background context: {context}\nMessage: {message}")
        try:
            json_text = response.candidates[0].content.parts[0].text
            data = json.loads(json_text)
            human_response = to_human_response(data)
            return human_response
        except Exception as e:
            print(e, response)
            return HumanResponse(
                my_name_is="",
                motion="",
                language=None,
                emotion="",
                thought="",
                answer="",
                association_words=[]
            )

    def single_word(self, context: str, message: str) -> SingleWord:
        model = genai.GenerativeModel(model_name="gemini-2.5-flash",
                                      generation_config={
                                          "temperature": self._temperature,
                                          "response_mime_type": "application/json",
                                          "response_schema": SingleWord
                                      },
                                      system_instruction="Answer emotion fields with emojies. Be sure to mention a littlge about each message of the background context that is provided in the message. Use emojies that differs from user.")
        chat = model.start_chat()
        response = chat.send_message(f"Background context: {context}\nMessage: {message}")
        try:
            json_text = response.candidates[0].content.parts[0].text
            data = json.loads(json_text)
            word_instance = SingleWord(**data)
            return word_instance
        except:
            return SingleWord(word="")


class AttentionClientInterface(Protocol):
    def attention_scores(self, sentence: str) -> list[AttentionWord]:
        raise NotImplementedError


class AttentionClientV1(AttentionClientInterface):
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
        self._model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased", output_attentions=True)

    def attention_scores(self, sentence: str) -> list[AttentionWord]:
        inputs = self._tokenizer(sentence, return_tensors="pt", return_attention_mask=True)
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Last 4 layers, average over heads
        att = torch.stack(outputs.attentions[-4:])  # [4, batch, heads, tokens, tokens]
        att = att.mean(dim=0).mean(dim=1)  # [batch, tokens, tokens]
        att = att[0]

        # Token importance
        importance = att.sum(dim=0)  # [tokens]

        # Convert token ids to words
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        words_with_scores = []
        current_word = ""
        current_score = 0.0
        count = 0

        for token, score in zip(tokens, importance):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if token.startswith("##"):
                # Subword continuation
                current_word += token[2:]
            else:
                # Save previous word
                if current_word:
                    words_with_scores.append(
                        AttentionWord(word=current_word, value=current_score / count)
                    )
                current_word = token
                current_score = 0.0
                count = 0

            current_score += score.item()
            count += 1

        # Add last word
        if current_word:
            words_with_scores.append(
                AttentionWord(word=current_word, value=current_score / count)
            )

        # Min-max normalize
        if words_with_scores:
            values = [w.value for w in words_with_scores]
            min_val, max_val = min(values), max(values)
            range_val = max_val - min_val if max_val != min_val else 1.0
            for w in words_with_scores:
                w.value = (w.value - min_val) / range_val

        return words_with_scores

    def get_embedding(self, text: str) -> np.ndarray:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self._model(**inputs)
        # Mean pooling over token embeddings
        emb = outputs.last_hidden_state.mean(dim=1)[0]
        return emb.numpy()


class DictionaryClientInterface(Protocol):
    def synonyms(self, word: str, language: Language) -> list[str]:
        raise NotImplementedError


class YandexDictionaryClient(DictionaryClientInterface):
    def __init__(self, settings: YandexSettings):
        self._url = f'https://dictionary.yandex.net/api/v1/dicservice.json/lookup?key={settings.api_key}'

    def synonyms(self, word: str, language: Language) -> list[str]:
        response = requests.get(f"{self._url}&lang={self._get_language_prefix(language)}&text={word}")
        if response.status_code == 200:
            json = response.json()
            if json['code'] == 200 and json['def'] and len(json['def']) > 0:
                return [synonym['text'] for synonym in json['def'][0]['tr']]
        return []

    def _get_language_prefix(self, language: Language) -> str:
        if language == Language.RU:
            return 'ru-ru'
        return ''


class SentenceEmbeddingInterface(Protocol):
    def get_sentences_embeddings(self, sentences: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceEmbeddingV1(SentenceEmbeddingInterface):
    def __init__(self):
        self._model = SentenceTransformer("sentence-transformers/LaBSE")

    def get_sentences_embeddings(self, sentences: list[str]) -> np.ndarray:
        return self._model.encode(sentences)
