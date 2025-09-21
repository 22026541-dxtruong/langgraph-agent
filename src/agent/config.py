"""Configuration settings for RAG system."""
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()


class Configuration(BaseModel):
    """Configurable parameters for RAG system."""
    
    # API Keys
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Model settings
    llm_model: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    
    # Database settings
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    collection_name: str = "rag_documents"
    
    # Retrieval settings
    top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.3
    
    # Quality thresholds
    min_quality_score: float = 0.6
    max_retries: int = 2
    
    # Query transformation
    max_transformed_queries: int = 3
    
    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return bool(self.google_api_key)