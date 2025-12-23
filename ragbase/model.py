from langchain_community.chat_models import ChatOllama
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

import os
from typing import Optional, List, Any

from types import SimpleNamespace
from huggingface_hub import InferenceClient
from ragbase.config import Config


class NoOpLLM(BaseLanguageModel):
    """Fallback LLM that returns a helpful message instead of contacting an API."""

    def invoke(self, input, **kwargs):
        message = (
            "No LLM configured. Set HUGGINGFACEHUB_API_TOKEN or GROQ_API_KEY in your environment"
        )
        return SimpleNamespace(content=message)


class HuggingFaceSimpleLLM(Runnable):
    """Minimal Hugging Face Inference API LLM wrapper.

    Inherits from Runnable to work with LangChain chains.
    """

    def __init__(self, model_id: str, api_token: str, temperature: float = 0.0, max_tokens: int = 512):
        self.model_id = model_id
        self.api_token = api_token
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, input, **kwargs):
        if isinstance(input, list):
            # Assume it's a list of messages
            prompt_text = "\n".join(str(m.content) if hasattr(m, 'content') else str(m) for m in input)
        elif isinstance(input, dict):
            prompt_text = input.get("question") or input.get("input") or str(input)
        else:
            prompt_text = str(input)
        
        client = InferenceClient(model=self.model_id, token=self.api_token)
        try:
            resp = client.text_generation(prompt_text, temperature=self.temperature, max_new_tokens=self.max_tokens)
            if isinstance(resp, str):
                content = resp
            else:
                content = resp.get("generated_text") if isinstance(resp, dict) else str(resp)
                if content is None:
                    content = str(resp)
        except Exception as e:
            content = f"HuggingFace API error: {e}"
        return SimpleNamespace(content=content)

    def predict(self, prompt: str, **kwargs) -> str:
        return self.invoke(prompt, **kwargs).content

    async def apredict(self, prompt: str, **kwargs) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.predict(prompt, **kwargs))

    def generate_prompt(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    async def agenerate_prompt(self, *args, **kwargs):
        return await self.apredict(*args, **kwargs)

    def predict_messages(self, messages, **kwargs):
        if isinstance(messages, (list, tuple)):
            prompt = "\n".join(str(m) for m in messages)
        else:
            prompt = str(messages)
        return SimpleNamespace(content=self.predict(prompt, **kwargs))

    async def apredict_messages(self, messages, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.predict_messages(messages, **kwargs))


def create_hf_llm() -> BaseLanguageModel:
    """Attempt to create a Hugging Face-backed LLM.

    The full LangChain-compatible HF wrapper isn't implemented here to avoid
    tight coupling with `langchain-huggingface`. As a pragmatic fallback,
    return the configured remote or local model so the app can run.
    """
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        return HuggingFaceSimpleLLM(
            model_id=Config.Model.HUGGING_FACE_MODEL,
            api_token=token,
            temperature=Config.Model.TEMPERATURE,
            max_tokens=Config.Model.MAX_TOKENS,
        )

    # If no HF token, avoid instantiating remote providers that need API keys.
    # Prefer local LLM if configured, otherwise return a NoOp fallback.
    if Config.Model.USE_LOCAL:
        return ChatOllama(
            model=Config.Model.LOCAL_LLM,
            temperature=Config.Model.TEMPERATURE,
            keep_alive="1h",
            max_tokens=Config.Model.MAX_TOKENS,
        )

    return NoOpLLM()


def create_llm() -> BaseLanguageModel:
    # Prefer Hugging Face if enabled
    if getattr(Config.Model, "USE_HUGGINGFACE", False):
        return create_hf_llm()

    if Config.Model.USE_LOCAL:
        return ChatOllama(
            model=Config.Model.LOCAL_LLM,
            temperature=Config.Model.TEMPERATURE,
            keep_alive="1h",
            max_tokens=Config.Model.MAX_TOKENS,
        )
    else:
        # Only instantiate ChatGroq if the GROQ_API_KEY is present; otherwise
        # return a NoOp fallback to avoid pydantic validation errors at init.
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            return ChatGroq(
                temperature=Config.Model.TEMPERATURE,
                model_name=Config.Model.REMOTE_LLM,
                max_tokens=Config.Model.MAX_TOKENS,
            )
        return NoOpLLM()


def create_multi_llms() -> list[BaseLanguageModel]:
    """Create multiple LLMs for ensemble-based QA.

    If Hugging Face is enabled, create a single HF LLM (ensemble with other remotes
    could be added if desired). Otherwise, default to remote/local models.
    """
    llms = []

    if getattr(Config.Model, "USE_HUGGINGFACE", False):
        llms.append(create_hf_llm())
        # Optionally also add a second HF/remote model if desired
        if hasattr(Config.Model, "REMOTE_LLM") and Config.Model.REMOTE_LLM:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                llms.append(
                    ChatGroq(
                        temperature=Config.Model.TEMPERATURE,
                        model_name=Config.Model.REMOTE_LLM,
                        max_tokens=Config.Model.MAX_TOKENS,
                    )
                )
            else:
                llms.append(NoOpLLM())
        return llms

    if Config.Model.USE_LOCAL:
        llms.append(
            ChatOllama(
                model=Config.Model.LOCAL_LLM,
                temperature=Config.Model.TEMPERATURE,
                keep_alive="1h",
                max_tokens=Config.Model.MAX_TOKENS,
            )
        )
    else:
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            # Add primary remote LLM
            llms.append(
                ChatGroq(
                    temperature=Config.Model.TEMPERATURE,
                    model_name=Config.Model.REMOTE_LLM,
                    max_tokens=Config.Model.MAX_TOKENS,
                )
            )
            # Add secondary remote LLM if configured
            if hasattr(Config.Model, "REMOTE_LLM_2") and Config.Model.REMOTE_LLM_2:
                llms.append(
                    ChatGroq(
                        temperature=Config.Model.TEMPERATURE,
                        model_name=Config.Model.REMOTE_LLM_2,
                        max_tokens=Config.Model.MAX_TOKENS,
                    )
                )
        else:
            # No GROQ key: fall back to NoOp placeholders so the app can run.
            llms.append(NoOpLLM())
            if hasattr(Config.Model, "REMOTE_LLM_2") and Config.Model.REMOTE_LLM_2:
                llms.append(NoOpLLM())

    return llms


def create_embeddings():
    # Prefer the local SentenceTransformers-based embeddings wrapper if available
    try:
        from ragbase.ingestor import SentenceTransformersEmbeddings

        return SentenceTransformersEmbeddings()
    except Exception:
        return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)


def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model=Config.Model.RERANKER)
