from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """API configuration settings."""
    enable_user_models: bool = True
    model_path: str = "models/"
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_prefix = "LOYALTY_"