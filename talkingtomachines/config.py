import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=False)


class Config:
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv("DATABASE_URI", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    QUALTRICS_API_KEY = os.getenv("QUALTRICS_API_KEY", "")
    OTREE_API_KEY = os.getenv("OTREE_API_KEY", "")
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


class DevelopmentConfig(Config):
    DEBUG = True
    DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///dev.db")


class TestingConfig(Config):
    TESTING = True
    DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///test.db")


class ProductionConfig(Config):
    TESTING = False
    DATABASE_URI = os.getenv("DATABASE_URI", "postgresql://user@localhost/prod")
