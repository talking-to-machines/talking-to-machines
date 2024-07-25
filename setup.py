from setuptools import setup, find_packages

setup(
    name="talkingtomachines",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "requests",
        "pandas",
        "numpy",
        "scikit-learn",
        "nltk",
        "sqlalchemy",
        "psycopg2-binary",
        "openai",
        "otree",
    ],
    entry_points={
        "console_scripts": ["talkingtomachines = talkingtomachines.main:app.run"]
    },
)
