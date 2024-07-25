import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///:memory:")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
    QUALTRICS_API_KEY = os.getenv("QUALTRICS_API_KEY", "your_qualtrics_api_key")
    OTREE_API_KEY = os.getenv("OTREE_API_KEY", "your_otree_api_key")


class DevelopmentConfig(Config):
    DEBUG = True
    DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///dev.db")


class TestingConfig(Config):
    TESTING = True
    DATABASE_URI = os.getenv("DATABASE_URI", "sqlite:///test.db")


class ProductionConfig(Config):
    DATABASE_URI = os.getenv("DATABASE_URI", "postgresql://user@localhost/prod")
