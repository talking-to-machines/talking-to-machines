"""Application configuration for the Talking to Machines platform.

This module defines environment-aware configuration classes that load
API keys and runtime settings from environment variables (via a ``.env``
file).  Three profiles are provided: development, testing, and production.
"""

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=False)


class Config:
    """Base configuration loaded from environment variables.

    Attributes:
        DEBUG (bool): Enable debug mode. Defaults to ``False``.
        TESTING (bool): Enable testing mode. Defaults to ``False``.
        DATABASE_URI (str): Database connection URI.
        OPENAI_API_KEY (str): API key for OpenAI services.
        QUALTRICS_API_KEY (str): API key for Qualtrics integration.
        OTREE_API_KEY (str): API key for oTree integration.
        HF_API_KEY (str): API key for Hugging Face inference.
        OPENROUTER_API_KEY (str): API key for OpenRouter.ai.
        ANTHROPIC_API_KEY (str): API key for Anthropic models.
        GOOGLE_API_KEY (str): API key for Google AI models.
        MISTRAL_API_KEY (str): API key for Mistral AI models.
        XAI_API_KEY (str): API key for xAI models.
        DEEPSEEK_API_KEY (str): API key for DeepSeek models.
        BUDGET_CAP_USD (float): Maximum spend in USD (0 means no cap).
    """

    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv("DATABASE_URI", "")
    # Existing providers
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    QUALTRICS_API_KEY = os.getenv("QUALTRICS_API_KEY", "")
    OTREE_API_KEY = os.getenv("OTREE_API_KEY", "")
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    # New providers (Phase 4)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    # Budget cap in USD (0 = no cap). Defaults to 0.0 if env var is non-numeric.
    try:
        BUDGET_CAP_USD: float = float(os.getenv("BUDGET_CAP_USD", "0") or "0")
    except ValueError:
        BUDGET_CAP_USD = 0.0


class DevelopmentConfig(Config):
    """Development configuration with debug mode enabled.

    Attributes:
        DEBUG (bool): Always ``True`` in development.
        DATABASE_URI (str): Defaults to a local SQLite database (``dev.db``).
    """

    DEBUG = True
    DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///dev.db")


class TestingConfig(Config):
    """Testing configuration with the testing flag enabled.

    Attributes:
        TESTING (bool): Always ``True`` in testing.
        DATABASE_URI (str): Defaults to a local SQLite database (``test.db``).
    """

    TESTING = True
    DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///test.db")


class ProductionConfig(Config):
    """Production configuration with conservative defaults.

    Attributes:
        TESTING (bool): Always ``False`` in production.
        DATABASE_URI (str): Defaults to a PostgreSQL connection string.
    """

    TESTING = False
    DATABASE_URI = os.getenv("DATABASE_URI", "postgresql://user@localhost/prod")
