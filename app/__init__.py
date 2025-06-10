from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Імпортуємо маршрути ПІСЛЯ створення app, щоб уникнути циклів
from .routes import register_routes
register_routes(app)