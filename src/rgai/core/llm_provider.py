import os
from dotenv import load_dotenv

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

from rgai.util.utils import load_yaml
from rgai.auth.auth import get_bearer_token

load_dotenv()

def getOpenAIModel41() -> ChatOpenAI:
    
    username = os.getenv("OPEN_AI_PROXY_USERNAME")
    password = os.getenv("OPEN_AI_PROXY_PASSWORD")

    cfg = load_yaml("config/model_config.yaml")
    auth_cfg = cfg["auth"]

    token = get_bearer_token(auth_cfg, username=username, password=password)

    model_cfg = cfg["model_gateway"]
    root_url = model_cfg["root_url"]
    deployment_name = model_cfg["deployment_name"]
    api_version = model_cfg["api_version"]
    base_url = f"{root_url}/openai/deployments/{deployment_name}"
    default_query = {"api-version": api_version}
    return ChatOpenAI(base_url=base_url, api_key=token, default_query=default_query)
  
class LLMRegistry:
    """Registry of available LLM models with pre-initialized instances.

    This class maintains a list of LLM configurations and provides
    methods to retrieve them by name with optional argument overrides.
    """

    # Class-level variable containing all available LLM models
    LLMS: List[Dict[str, Any]] = [
        {
            "name": "qwen2.5-14b",
            "llm": ChatOllama(model="qwen2.5:14b",temperature=0),
        },
        {
            "name": "gpt-41",
            "llm": getOpenAIModel41(),
        },
        # {
        #     "name": "cohere.command-r-08-2024",
        #     "llm": ChatOCIGenAI(
        #         model_id="cohere.command-r-08-2024",
        #         service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        #         compartment_id="ocid1.compartment.oc1..xxxxx",
        #         auth_type="API_KEY",
        #         auth_profile="xxxxx",
        #         model_kwargs={"temperature": 0, "top_p": 0.75, "max_tokens": 512}
        #     ),
        # },
    ]

    @classmethod
    def get(cls, model_name: str) -> BaseChatModel:
        """Get an LLM by name.

        Args:
            model_name: Name of the model to retrieve

        Returns:
            BaseChatModel instance

        Raises:
            ValueError: If model_name is not found in LLMS
        """
        # Find the model in the registry
        model_entry = None
        for entry in cls.LLMS:
            if entry["name"] == model_name:
                model_entry = entry
                break

        if not model_entry:
            available_models = [entry["name"] for entry in cls.LLMS]
            raise ValueError(
                f"model '{model_name}' not found in registry. available models: {', '.join(available_models)}"
            )
        return model_entry["llm"]