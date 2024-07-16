# Entry point of the application
from flask import Flask
from talkingtomachines.config import DevelopmentConfig


def create_app(config_class=DevelopmentConfig):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config_class)

    @app.route("/")
    def home():
        return "Welcome to the Talking To Machines Platform"

    return app


if __name__ == "__main__":
    app = create_app()
    app.run()
