from datetime import timedelta
import uuid

from app.repositories.market_intelligence_repository import MarketIntelligenceRepository
from app.services.market_intelligence_service import MarketIntelligenceService
from app.utils.time_utils import utcnow


class StaticWeatherClient:
    def __init__(self, payload=None):
        self.payload = payload or {"provider": "test", "available": False, "days": []}

    def fetch_daily_forecast(self, days=7):
        return self.payload


class FailingWeatherClient:
    def fetch_daily_forecast(self, days=7):
        raise RuntimeError("weather unavailable")


def build_service(db, weather_client=None, min_confidence=70, min_trend_score=65):
    model_path = f"/tmp/test_market_model_{uuid.uuid4().hex}.pkl"
    return MarketIntelligenceService(
        MarketIntelligenceRepository(db),
        weather_client=weather_client or StaticWeatherClient(),
        refresh_hours=6,
        min_confidence=min_confidence,
        min_trend_score=min_trend_score,
        model_version="ml-lite-v1-test",
        model_store_path=model_path,
        min_train_samples=20,
        retrain_hours=24,
    )


def _seed_orders(db, days=30, base=8):
    now = utcnow()
    rows = []
    for idx in range(days):
        dt = now - timedelta(days=(days - idx))
        count = base + (idx % 5)
        for order_idx in range(count):
            rows.append(
                {
                    "order_number": f"ORD-MI-{idx}-{order_idx}",
                    "status": "confirmed",
                    "total_amount": 1000 + idx,
                    "quantity": 5,
                    "created_at": dt,
                    "updated_at": dt,
                }
            )
    if rows:
        db.orders.insert_many(rows)


def _seed_tomato_trend(db, farmer_id):
    now = utcnow()
    product_id = db.products.insert_one(
        {
            "name": "Premium Tomato",
            "category": "vegetable",
            "price": 1500,
            "status": "approved",
            "farmer_id": str(farmer_id),
            "created_at": now - timedelta(days=5),
            "updated_at": now - timedelta(days=5),
        }
    ).inserted_id

    offers = []
    for idx in range(2):
        dt = now - timedelta(days=13 - idx)
        offers.append(
            {
                "product_id": str(product_id),
                "buyer_id": f"buyer-prev-{idx}",
                "farmer_id": str(farmer_id),
                "price": 900,
                "quantity": 4,
                "status": "pending",
                "history": [],
                "created_at": dt,
                "updated_at": dt,
                "expires_at": dt + timedelta(days=1),
            }
        )

    for idx in range(6):
        dt = now - timedelta(days=2, hours=idx)
        offers.append(
            {
                "product_id": str(product_id),
                "buyer_id": f"buyer-recent-{idx}",
                "farmer_id": str(farmer_id),
                "price": 1700,
                "quantity": 14,
                "status": "countered",
                "history": [],
                "created_at": dt,
                "updated_at": dt,
                "expires_at": dt + timedelta(days=1),
            }
        )

    db.offers.insert_many(offers)


def test_forecast_non_negative_and_confidence_range(db):
    _seed_orders(db, days=35, base=6)
    service = build_service(db)

    service.refresh_insights(notify=False)
    forecast = service.get_forecast("1d")

    assert forecast["predicted_orders"] >= 0
    assert 0 <= forecast["confidence"] <= 100
    assert forecast["horizon"] == "1d"


def test_forecast_empty_dataset_safe_fallback(db):
    db.orders.delete_many({})
    service = build_service(db)

    service.refresh_insights(notify=False)
    forecast = service.get_forecast("3d")

    assert forecast["predicted_orders"] >= 0
    assert forecast["forecast_for_date"] is not None


def test_weather_failure_falls_back_without_crash(db):
    _seed_orders(db, days=20, base=5)
    service = build_service(db, weather_client=FailingWeatherClient())

    result = service.refresh_insights(notify=False)
    assert result["forecasts"]


def test_crop_trends_filter_invalid_values(db):
    farmer = db.users.find_one({"role": "farmer"})
    now = utcnow()
    product_id = db.products.insert_one(
        {
            "name": "Organic Tomato",
            "category": "vegetable",
            "price": 1000,
            "status": "approved",
            "farmer_id": str(farmer["_id"]),
            "created_at": now,
            "updated_at": now,
        }
    ).inserted_id

    db.offers.insert_many(
        [
            {
                "product_id": str(product_id),
                "buyer_id": "b1",
                "farmer_id": str(farmer["_id"]),
                "price": None,
                "quantity": 0,
                "status": "pending",
                "history": [],
                "created_at": now - timedelta(days=10),
                "updated_at": now - timedelta(days=10),
                "expires_at": now,
            },
            {
                "product_id": str(product_id),
                "buyer_id": "b2",
                "farmer_id": str(farmer["_id"]),
                "price": 1600,
                "quantity": 8,
                "status": "countered",
                "history": [],
                "created_at": now - timedelta(days=1),
                "updated_at": now - timedelta(days=1),
                "expires_at": now,
            },
        ]
    )

    service = build_service(db, min_confidence=0, min_trend_score=0)
    service.refresh_insights(notify=False)
    trends = service.get_crop_trends("1d")

    assert isinstance(trends["items"], list)
    if trends["items"]:
        assert all(item["crop"] for item in trends["items"])


def test_confidence_gating_can_suppress_weak_trends(db):
    farmer = db.users.find_one({"role": "farmer"})
    _seed_tomato_trend(db, farmer["_id"])
    service = build_service(db, min_confidence=99, min_trend_score=99)

    service.refresh_insights(notify=False)
    trends = service.get_crop_trends("7d")

    assert trends["items"] == []


def test_notifications_sent_only_to_relevant_farmers_and_deduped(db):
    now = utcnow()
    farmer_1 = db.users.find_one({"email": "farmer1@example.com"})
    farmer_2_id = db.users.insert_one(
        {
            "name": "Farmer Two",
            "email": "farmer2@example.com",
            "role": "farmer",
            "status": "active",
            "created_at": now - timedelta(days=10),
            "updated_at": now - timedelta(days=10),
        }
    ).inserted_id

    _seed_tomato_trend(db, farmer_1["_id"])

    db.products.insert_one(
        {
            "name": "Fresh Potato",
            "category": "vegetable",
            "price": 1300,
            "status": "approved",
            "farmer_id": str(farmer_2_id),
            "created_at": now - timedelta(days=3),
            "updated_at": now - timedelta(days=3),
        }
    )

    service = build_service(db, min_confidence=0, min_trend_score=0)
    first = service.refresh_insights(notify=True)
    second = service.refresh_insights(notify=True)

    notifications = list(db.farmer_notifications.find({"payload.crop": "tomato"}))
    notified_farmer_ids = {row.get("farmer_id") for row in notifications}

    assert str(farmer_1["_id"]) in notified_farmer_ids
    assert str(farmer_2_id) not in notified_farmer_ids
    assert len(notifications) >= 1
    assert second["notifications_sent"] <= first["notifications_sent"]


def test_training_persists_model_and_marks_forecast_model_meta(db):
    _seed_orders(db, days=80, base=6)
    service = build_service(db, min_confidence=0, min_trend_score=0)

    result = service.refresh_insights(notify=False)
    assert result["forecasts"]

    model_path = service.model_store_path
    with open(model_path, "rb") as fp:
        payload = fp.read()
    assert payload

    forecast = service.get_forecast("1d")
    model_meta = result["forecasts"][0].get("model") or {}
    assert model_meta.get("type") in {"sklearn_random_forest", "weekday_baseline"}
    assert forecast["predicted_orders"] >= 0
