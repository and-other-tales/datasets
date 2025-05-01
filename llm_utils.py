"""
LLM Provider Utilities for Dataset Creator Agent

This module provides utility functions to initialize various LLM providers
based on environment variables. It supports multiple LLM providers including:
- OpenAI (ChatGPT)
- Anthropic (Claude)
- AWS Bedrock
- Azure OpenAI
- Google Generative AI (Gemini)
- Groq
- HuggingFace

Environment variables:
- LLM_PROVIDER: The provider to use (openai, anthropic, bedrock, azure, google, groq, huggingface)
"""

import os
import logging
from typing import Dict, Any, Optional, Union, Callable

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS Bedrock
try:
    from langchain_aws import ChatBedrockConverse
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    logger.warning("AWS Bedrock not available. Install with `pip install langchain-aws`")

# OpenAI
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install with `pip install langchain-openai`")

# Anthropic
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not available. Install with `pip install langchain-anthropic`")

# Azure
try:
    from langchain_openai import AzureChatOpenAI
    AZURE_AVAILABLE = OPENAI_AVAILABLE
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure OpenAI not available. Install with `pip install langchain-openai`")

# Google
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.warning("Google Generative AI not available. Install with `pip install langchain-google-genai`")

# Groq
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq not available. Install with `pip install langchain-groq`")

# HuggingFace
try:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEndpoint
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("HuggingFace not available. Install with `pip install langchain-huggingface`")

class MockLLM:
    """Mock LLM for testing without API keys."""
    
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        
    def invoke(self, input, config=None, **kwargs):
        """Mock invoke method."""
        return {"content": "This is a mock response for testing"}
    
    async def ainvoke(self, input, config=None, **kwargs):
        """Mock async invoke method."""
        return {"content": "This is a mock async response for testing"}

def get_llm(callbacks=None) -> Any:
    """
    Get LLM based on environment variables.
    
    Supports multiple LLM providers:
    - openai: OpenAI models (ChatGPT)
    - anthropic: Anthropic Claude models
    - bedrock: AWS Bedrock models
    - azure: Azure OpenAI models
    - google: Google Gemini models
    - groq: Groq models
    - huggingface: HuggingFace models
    
    Returns:
        An instance of the specified LLM provider
    """
    # Get provider from environment variable
    provider = os.environ.get("LLM_PROVIDER", "").lower()
    
    # Validate provider if specified
    valid_providers = [
        "openai", "anthropic", "bedrock", "azure", 
        "google", "groq", "huggingface"
    ]
    
    if provider and provider not in valid_providers:
        logger.warning(f"Invalid LLM_PROVIDER value: '{provider}'. Using auto-detection.")
        provider = ""
    
    # If no provider specified, auto-detect based on available API keys
    if not provider:
        if os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("AZURE_OPENAI_API_KEY"):
            provider = "azure"
        elif os.environ.get("GOOGLE_API_KEY"):
            provider = "google"
        elif os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("HUGGINGFACE_API_KEY"):
            provider = "huggingface"
        elif BEDROCK_AVAILABLE and (os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE")):
            provider = "bedrock"
        else:
            # Default to bedrock if no other provider found
            provider = "bedrock"
    
    # 1. OpenAI models
    if provider == "openai":
        if not OPENAI_AVAILABLE:
            logger.warning("OpenAI support not available. Install with `pip install langchain-openai`")
            return MockLLM()
            
        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o")
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        org_id = os.environ.get("OPENAI_ORG_ID")
        temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))
        
        if not api_key:
            logger.warning("OpenAI API key not found, using mock LLM")
            return MockLLM()
            
        logger.info(f"Using OpenAI with model {model_name}")
        
        # Initialize with optional parameters if provided
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "api_key": api_key,
        }
        
        if base_url:
            kwargs["base_url"] = base_url
        if org_id:
            kwargs["organization"] = org_id
        if callbacks:
            kwargs["callbacks"] = callbacks
            
        return ChatOpenAI(**kwargs)
    
    # 2. Anthropic Claude models
    elif provider == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic support not available. Install with `pip install langchain-anthropic`")
            return MockLLM()
            
        model_name = os.environ.get("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        temperature = float(os.environ.get("ANTHROPIC_TEMPERATURE", "0.2"))
        
        if not api_key:
            logger.warning("Anthropic API key not found, using mock LLM")
            return MockLLM()
            
        logger.info(f"Using Anthropic Claude with model {model_name}")
        
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "anthropic_api_key": api_key,
        }
        
        if callbacks:
            kwargs["callbacks"] = callbacks
            
        return ChatAnthropic(**kwargs)
    
    # 3. AWS Bedrock models
    elif provider == "bedrock":
        if not BEDROCK_AVAILABLE:
            logger.warning("AWS Bedrock support not available. Install with `pip install langchain-aws`")
            return MockLLM()
            
        model_id = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-7-sonnet-20250219-v1:0")
        region = os.environ.get("AWS_REGION", "us-east-1")
        temperature = float(os.environ.get("BEDROCK_TEMPERATURE", "0.2"))
        max_tokens = int(os.environ.get("BEDROCK_MAX_TOKENS", "2000"))
        
        logger.info(f"Using AWS Bedrock with model {model_id}")
        
        kwargs = {
            "model_id": model_id,
            "region_name": region,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if callbacks:
            kwargs["callbacks"] = callbacks
            
        return ChatBedrockConverse(**kwargs)
    
    # 4. Azure OpenAI models
    elif provider == "azure":
        if not AZURE_AVAILABLE:
            logger.warning("Azure OpenAI support not available. Install with `pip install langchain-openai`")
            return MockLLM()
            
        deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        temperature = float(os.environ.get("AZURE_OPENAI_TEMPERATURE", "0.2"))
        
        if not all([deployment_name, endpoint, api_key]):
            logger.warning("Azure OpenAI configuration incomplete, using mock LLM")
            return MockLLM()
            
        logger.info(f"Using Azure OpenAI with deployment {deployment_name}")
        
        kwargs = {
            "deployment_name": deployment_name,
            "azure_endpoint": endpoint,
            "api_key": api_key,
            "api_version": api_version,
            "temperature": temperature,
        }
        
        if callbacks:
            kwargs["callbacks"] = callbacks
            
        return AzureChatOpenAI(**kwargs)
    
    # 5. Google Gemini models
    elif provider == "google":
        if not GOOGLE_AVAILABLE:
            logger.warning("Google Generative AI support not available. Install with `pip install langchain-google-genai`")
            return MockLLM()
            
        model_name = os.environ.get("GOOGLE_MODEL", "gemini-1.5-pro")
        api_key = os.environ.get("GOOGLE_API_KEY")
        temperature = float(os.environ.get("GOOGLE_TEMPERATURE", "0.2"))
        
        if not api_key:
            logger.warning("Google API key not found, using mock LLM")
            return MockLLM()
            
        logger.info(f"Using Google Gemini with model {model_name}")
        
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "google_api_key": api_key,
        }
        
        if callbacks:
            kwargs["callbacks"] = callbacks
            
        return ChatGoogleGenerativeAI(**kwargs)
    
    # 6. Groq models
    elif provider == "groq":
        if not GROQ_AVAILABLE:
            logger.warning("Groq support not available. Install with `pip install langchain-groq`")
            return MockLLM()
            
        model_name = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
        api_key = os.environ.get("GROQ_API_KEY")
        temperature = float(os.environ.get("GROQ_TEMPERATURE", "0.2"))
        
        if not api_key:
            logger.warning("Groq API key not found, using mock LLM")
            return MockLLM()
            
        logger.info(f"Using Groq with model {model_name}")
        
        kwargs = {
            "model_name": model_name,
            "temperature": temperature,
            "api_key": api_key,
        }
        
        if callbacks:
            kwargs["callbacks"] = callbacks
            
        return ChatGroq(**kwargs)
    
    # 7. HuggingFace models
    elif provider == "huggingface":
        if not HUGGINGFACE_AVAILABLE:
            logger.warning("HuggingFace support not available. Install with `pip install langchain-huggingface`")
            return MockLLM()
            
        # Check if using endpoint or local
        if os.environ.get("HUGGINGFACE_ENDPOINT_URL"):
            # Using HuggingFace Endpoint
            endpoint_url = os.environ.get("HUGGINGFACE_ENDPOINT_URL")
            api_key = os.environ.get("HUGGINGFACE_API_KEY")
            
            if not all([endpoint_url, api_key]):
                logger.warning("HuggingFace Endpoint configuration incomplete, using mock LLM")
                return MockLLM()
                
            logger.info(f"Using HuggingFace Endpoint {endpoint_url}")
            
            # Create HuggingFace Endpoint
            hf_endpoint = HuggingFaceEndpoint(
                endpoint_url=endpoint_url,
                huggingfacehub_api_token=api_key,
                task="text-generation",
                max_new_tokens=int(os.environ.get("HUGGINGFACE_MAX_TOKENS", "512")),
                temperature=float(os.environ.get("HUGGINGFACE_TEMPERATURE", "0.2"))
            )
            
            # Wrap with ChatHuggingFace
            kwargs = {"llm": hf_endpoint}
            if callbacks:
                kwargs["callbacks"] = callbacks
                
            return ChatHuggingFace(**kwargs)
        else:
            # Using local HuggingFace model
            model_id = os.environ.get("HUGGINGFACE_MODEL_ID", "mistralai/Mixtral-8x7B-Instruct-v0.1")
            
            logger.info(f"Using local HuggingFace model {model_id}")
            
            try:
                # Create HuggingFace Pipeline LLM
                hf_pipeline = HuggingFacePipeline.from_model_id(
                    model_id=model_id,
                    task="text-generation",
                    device=int(os.environ.get("HUGGINGFACE_DEVICE", "-1")),  # Default to CPU with -1
                    pipeline_kwargs={
                        "max_new_tokens": int(os.environ.get("HUGGINGFACE_MAX_TOKENS", "512")),
                        "do_sample": True,
                        "temperature": float(os.environ.get("HUGGINGFACE_TEMPERATURE", "0.2")),
                        "top_k": int(os.environ.get("HUGGINGFACE_TOP_K", "50")),
                        "top_p": float(os.environ.get("HUGGINGFACE_TOP_P", "0.95")),
                    },
                )
                
                # Create ChatHuggingFace using the pipeline
                kwargs = {"llm": hf_pipeline}
                if callbacks:
                    kwargs["callbacks"] = callbacks
                    
                return ChatHuggingFace(**kwargs)
            except Exception as e:
                logger.error(f"Error initializing HuggingFace model: {e}")
                return MockLLM()
    
    # Default: fall back to AWS Bedrock
    else:
        if BEDROCK_AVAILABLE:
            logger.info("No specific provider configured, using AWS Bedrock as default")
            model_id = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-7-sonnet-20250219-v1:0")
            region = os.environ.get("AWS_REGION", "us-east-1")
            
            kwargs = {
                "model_id": model_id,
                "region_name": region,
                "temperature": 0.2,
                "max_tokens": 2000,
            }
            
            if callbacks:
                kwargs["callbacks"] = callbacks
                
            return ChatBedrockConverse(**kwargs)
        else:
            logger.warning("AWS Bedrock not available, using mock LLM")
            return MockLLM()