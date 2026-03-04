import os
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

from rgai.util.utils import load_yaml
from rgai.auth.auth import get_api_key

load_dotenv()

cfg = load_yaml("config/models_config.yaml")

def getChatOpenAIModel() -> ChatOpenAI:
    
    username = os.getenv("OPEN_AI_PROXY_USERNAME")
    password = os.getenv("OPEN_AI_PROXY_PASSWORD")
    client_id = os.getenv("OAUTH_CLIENT_ID")
    client_secret = os.getenv("OAUTH_CLIENT_SECRET")

    model_cfg = cfg["openai"]
    auth_cfg = model_cfg["oauth"]

    token = get_api_key(auth_cfg, client_id=client_id, client_secret=client_secret, username=username, password=password)

    base_url = model_cfg["base_url"]
    api_version = model_cfg["api_version"]
    default_query = {"api-version": api_version}
    return ChatOpenAI(base_url=base_url, api_key=token, default_query=default_query)

def getChatOCIGenAIModel() -> ChatOCIGenAI:
    
    model_cfg = cfg["oci"]

    model_id = model_cfg["model_id"]
    service_endpoint = model_cfg["service_endpoint"]
    compartment_id = model_cfg["compartment_id"]
    auth_type = model_cfg["auth_type"]
    auth_profile = model_cfg["auth_profile"]
    kw_args = model_cfg["model_kwargs"]
    
    model_kwargs = {"temperature": kw_args["temperature"], "top_p": kw_args["top_p"], "max_tokens": kw_args["max_tokens"]}

    return ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=service_endpoint,
        compartment_id=compartment_id,
        auth_type=auth_type,
        auth_profile=auth_profile,
        model_kwargs=model_kwargs
    )

def getOllamaModel() -> ChatOllama:

    model_cfg = cfg["local"]
    
    model_id = model_cfg["model_id"]
    return ChatOllama(model=model_id, temperature=0)

class LLMFactory:
    """Factory class to create different LLM instances based on configuration."""
    
    @staticmethod
    def create_llm(provider: str) -> BaseChatModel:
        """Get an LLM by provider.

        Args:
            provider: Name of the provider's model to retrieve

        Returns:
            BaseChatModel instance

        Raises:
            ValueError: If provider is not supported
        """
        if (provider.lower() == "local"):
            return getOllamaModel()
        elif (provider.lower() == "openai"):
            return getChatOpenAIModel()
        elif (provider.lower() == "oci"):
            return getChatOCIGenAIModel()
        else:
            raise ValueError(f"Unsupported provider '{provider}'. Supported providers: ollama, openai, oci")
    
