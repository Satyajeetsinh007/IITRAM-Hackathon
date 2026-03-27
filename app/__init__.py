from urllib.parse import urlparse

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient

from app.admin.routes import admin_bp
from app.cli import register_commands
from app.config import CONFIG_BY_NAME
from app.extensions import csrf
from app.farmer.routes import farmer_bp
from app.offers.routes import offers_bp
from app.repositories.schema import ensure_database_structure

load_dotenv()


def _resolve_db_name(app):
    configured = app.config.get("DATABASE_NAME")
    if configured:
        return configured

    parsed = urlparse(app.config["MONGO_URI"])
    name_from_uri = (parsed.path or "").strip("/")
    return name_from_uri or "farm_to_market"


def create_app(config_name=None, test_config=None):
    app = Flask(__name__)

    if test_config:
        app.config.from_mapping(test_config)
    else:
        env = config_name or "development"
        app.config.from_object(CONFIG_BY_NAME.get(env, CONFIG_BY_NAME["development"]))

    if app.config.get("DEBUG"):
        app.config["SESSION_COOKIE_SECURE"] = False

    csrf.init_app(app)

    mongo_client = app.config.get("MONGO_CLIENT") or MongoClient(app.config["MONGO_URI"])
    app.mongo_client = mongo_client
    app.db = mongo_client[_resolve_db_name(app)]

    ensure_database_structure(app.db, apply_validators=True)

    register_commands(app)
    app.register_blueprint(admin_bp)
    app.register_blueprint(offers_bp)
    app.register_blueprint(farmer_bp)

    @app.get("/")
    def healthcheck():
        return jsonify({"status": "ok", "service": "farm-to-market-admin"})

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(500)
    @app.errorhandler(Exception)
    def handle_http_errors(error):
        wants_json = (
            request.path.startswith("/admin/api")
            or request.path.startswith("/offers")
            or request.path.startswith("/farmer/api")
            or request.accept_mimetypes.best == "application/json"
        )
        code = getattr(error, "code", 500)
        description = getattr(error, "description", "Unexpected error")
        if wants_json:
            return jsonify({"error": description, "status": code}), code
        return render_template("admin/error.html", code=code, message=description), code

    return app
