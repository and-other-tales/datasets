"""
LLM Provider Utilities for Dataset Creator Agent
"""

import os
from typing import List, Dict, Any, Optional, Union

# Base LLM imports
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.base import BaseCallbackHandler

# Multiple LLM providers
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from langchain_groq import ChatGroq
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    from langchain_azure import AzureChatOpenAI
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from langchain_huggingface import ChatHuggingFace
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

def get_llm(callbacks: Optional[List[BaseCallbackHandler]] = None) -> BaseChatModel:
    """
    Get an LLM instance based on environment configuration.
    
    Supported providers:
    - bedrock (default): Amazon Bedrock
    - openai: OpenAI API
    - anthropic: Anthropic API (direct)
    - azure: Azure OpenAI
    - google: Google Gemini
    - groq: Groq API
    - huggingface: Hugging Face Inference API
    
    Returns:
        A configured LLM instance
    """
    provider = os.environ.get("LLM_PROVIDER", "bedrock").lower()
    
    # Common parameters
    model_kwargs = {}
    
    # Get the provider-specific LLM
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        temperature = float(os.environ.get("OPENAI_TEMPERATURE", 0.7))
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            model_kwargs=model_kwargs,
            callbacks=callbacks
        )
    
    elif provider == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
        temperature = float(os.environ.get("ANTHROPIC_TEMPERATURE", 0.7))
        
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            model_kwargs=model_kwargs,
            callbacks=callbacks
        )
    
    elif provider == "bedrock":
        model_id = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-7-sonnet-20250219-v1:0")
        temperature = float(os.environ.get("BEDROCK_TEMPERATURE", 0.7))
        
        # Region-specific settings
        region = os.environ.get("AWS_REGION", "us-west-2")
        
        # For Anthropic models
        if "anthropic" in model_id.lower():
            model_kwargs.update({
                "temperature": temperature,
                "max_tokens": 4096,
            })
        
        return ChatBedrock(
            model_id=model_id, 
            region_name=region,
            model_kwargs=model_kwargs,
            callbacks=callbacks
        )
    
    elif provider == "azure" and AZURE_AVAILABLE:
        deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        temperature = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", 0.7))
        
        return AzureChatOpenAI(
            deployment_name=deployment_name,
            temperature=temperature,
            model_kwargs=model_kwargs,
            callbacks=callbacks
        )
    
    elif provider == "google" and GOOGLE_AVAILABLE:
        model = os.environ.get("GOOGLE_MODEL", "gemini-1.5-pro")
        temperature = float(os.environ.get("GOOGLE_TEMPERATURE", 0.7))
        
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            callbacks=callbacks
        )
    
    elif provider == "groq":
        model = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
        temperature = float(os.environ.get("GROQ_TEMPERATURE", 0.7))
        
        return ChatGroq(
            model=model,
            temperature=temperature,
            callbacks=callbacks
        )
    
    elif provider == "huggingface" and HUGGINGFACE_AVAILABLE:
        model_id = os.environ.get("HUGGINGFACE_MODEL_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")
        temperature = float(os.environ.get("HUGGINGFACE_TEMPERATURE", 0.7))
        
        return ChatHuggingFace(
            model=model_id,
            temperature=temperature,
            callbacks=callbacks
        )
    
    # Default to Bedrock if the provider isn't supported
    print(f"Provider '{provider}' not supported or missing dependencies. Using Bedrock as default.")
    return ChatBedrock(
        model_id="anthropic.claude-3-7-sonnet-20250219-v1:0",
        region_name=os.environ.get("AWS_REGION", "us-west-2"),
        callbacks=callbacks
    )