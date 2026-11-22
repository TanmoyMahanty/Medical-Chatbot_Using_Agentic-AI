# streamlining project configuration by creating a centralized config.py file
# to securely load and validate environment variables for consistent access across the application.

import os
from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    # """
    # Settings class for managing application configuration using environment variables.
    # """
    
    #LLM Provider
    GROQ_API_KEY: SecretStr 
    
    #Pinecone Configuration
    PINECONE_API_KEY: SecretStr |None = None
    PINECONE_INDEX_NAME: str | None = None
    
    #Hugging Face Configuration
    HF_TOKEN: SecretStr 
    
    #LangSmith Observability Configuration
    LANGSMITH_API_KEY: SecretStr| None = None
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_PROJECT: str ='Medical-Chatbot'
    LANGSMITH_TRACING: bool = False
    
    #Model Configuration
    CHAT_MODEL: str = "groq:openai/gpt-oss-120b"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    model_config = SettingsConfigDict(env_file=".env",env_file_encoding="utf-8", extra = 'ignore')
    
settings = Settings()
