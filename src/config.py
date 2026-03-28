from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    ollama_model: str = Field(default='llama3.2')
    ollama_base_url: str = Field(default='http://localhost:11434')
    chroma_collection: str = Field(default='documents')
    chroma_persist_dir: str = Field(default='./chroma_db')
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    top_k: int = Field(default=5)
    api_host: str = Field(default='0.0.0.0')
    api_port: int = Field(default=8000)

    class Config:
        env_file = '.env'
        env_prefix = 'RAG_'

settings = Settings()
