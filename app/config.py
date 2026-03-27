import os
from datetime import timedelta


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    MONGO_URI = os.getenv(
        "MONGODB_URI",
        os.getenv("MONGO_URI", "mongodb://localhost:27017/farm_to_market"),
    )
    DATABASE_NAME = os.getenv("DATABASE_NAME", "farm_to_market")
    DEBUG = _as_bool(os.getenv("FLASK_DEBUG"), False)
    TESTING = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _as_bool(os.getenv("SESSION_COOKIE_SECURE"), False)
    PERMANENT_SESSION_LIFETIME = timedelta(hours=8)
    WTF_CSRF_TIME_LIMIT = None
    ADMIN_PAGE_SIZE = 20
    OFFER_EXPIRY_HOURS = int(os.getenv("OFFER_EXPIRY_HOURS", "24"))
    AI_REFRESH_HOURS = int(os.getenv("AI_REFRESH_HOURS", "6"))
    AI_MIN_CONFIDENCE = int(os.getenv("AI_MIN_CONFIDENCE", "70"))
    AI_MIN_TREND_SCORE = int(os.getenv("AI_MIN_TREND_SCORE", "65"))
    AI_MODEL_VERSION = os.getenv("AI_MODEL_VERSION", "ml-lite-v1")
    AI_MODEL_STORE_PATH = os.getenv(
        "AI_MODEL_STORE_PATH",
        "artifacts/order_forecast_model.pkl",
    )
    AI_MIN_TRAIN_SAMPLES = int(os.getenv("AI_MIN_TRAIN_SAMPLES", "45"))
    AI_RETRAIN_HOURS = int(os.getenv("AI_RETRAIN_HOURS", "24"))
    WEATHER_PROVIDER = os.getenv("WEATHER_PROVIDER", "open_meteo")
    WEATHER_LAT = float(os.getenv("WEATHER_LAT", "23.0225"))
    WEATHER_LON = float(os.getenv("WEATHER_LON", "72.5714"))
    WEATHER_TIMEOUT_SEC = int(os.getenv("WEATHER_TIMEOUT_SEC", "4"))


class DevelopmentConfig(Config):
    DEBUG = True
    SESSION_COOKIE_SECURE = False


class TestingConfig(Config):
    TESTING = True
    SESSION_COOKIE_SECURE = False
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True


CONFIG_BY_NAME = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
