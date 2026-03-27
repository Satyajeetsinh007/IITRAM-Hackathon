from datetime import timedelta

import mongomock
import pytest
import uuid
from werkzeug.security import generate_password_hash

from app import create_app
from app.utils.time_utils import utcnow


@pytest.fixture
def mongo_client():
    return mongomock.MongoClient()


@pytest.fixture
def app(mongo_client):
    model_path = f"/tmp/farm_to_market_test_model_{uuid.uuid4().hex}.pkl"
    app = create_app(
        test_config={
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "WTF_CSRF_ENABLED": False,
            "MONGO_CLIENT": mongo_client,
            "DATABASE_NAME": "farm_to_market_test",
            "ADMIN_PAGE_SIZE": 20,
            "AI_MODEL_STORE_PATH": model_path,
            "AI_MODEL_VERSION": "ml-lite-v1-test",
            "AI_MIN_TRAIN_SAMPLES": 20,
            "AI_RETRAIN_HOURS": 1,
        }
    )

    with app.app_context():
        db = app.db
        now = utcnow()

        db.admin_users.insert_one(
            {
                "email": "admin@example.com",
                "password_hash": generate_password_hash("adminpass123"),
                "status": "active",
                "created_at": now,
                "updated_at": now,
                "last_login_at": None,
            }
        )

        db.users.insert_many(
            [
                {
                    "name": "Farmer One",
                    "email": "farmer1@example.com",
                    "role": "farmer",
                    "status": "active",
                    "created_at": now - timedelta(days=5),
                    "updated_at": now - timedelta(days=5),
                },
                {
                    "name": "Buyer One",
                    "email": "buyer1@example.com",
                    "role": "buyer",
                    "status": "active",
                    "created_at": now - timedelta(days=3),
                    "updated_at": now - timedelta(days=3),
                },
            ]
        )

        db.products.insert_many(
            [
                {
                    "name": "Wheat",
                    "description": "Premium wheat",
                    "category": "grain",
                    "price": 2000,
                    "status": "pending",
                    "created_at": now - timedelta(days=4),
                    "updated_at": now - timedelta(days=4),
                },
                {
                    "name": "Tomato",
                    "description": "Organic tomato",
                    "category": "vegetable",
                    "price": 1200,
                    "status": "approved",
                    "created_at": now - timedelta(days=2),
                    "updated_at": now - timedelta(days=2),
                },
            ]
        )

        db.orders.insert_many(
            [
                {
                    "order_number": "ORD-1001",
                    "buyer_name": "Buyer One",
                    "status": "pending",
                    "total_amount": 5000,
                    "created_at": now - timedelta(days=2),
                    "updated_at": now - timedelta(days=2),
                },
                {
                    "order_number": "ORD-1002",
                    "buyer_name": "Store B",
                    "status": "confirmed",
                    "total_amount": 7000,
                    "created_at": now - timedelta(days=1),
                    "updated_at": now - timedelta(days=1),
                },
            ]
        )

        db.transactions.insert_many(
            [
                {
                    "transaction_ref": "TXN-1001",
                    "order_number": "ORD-1001",
                    "amount": 5000,
                    "payment_status": "pending",
                    "review_state": "pending_review",
                    "created_at": now - timedelta(days=2),
                    "updated_at": now - timedelta(days=2),
                },
                {
                    "transaction_ref": "TXN-1002",
                    "order_number": "ORD-1002",
                    "amount": 7000,
                    "payment_status": "paid",
                    "review_state": "reviewed",
                    "created_at": now - timedelta(days=1),
                    "updated_at": now - timedelta(days=1),
                },
            ]
        )

    return app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def db(app):
    return app.db


@pytest.fixture
def admin_credentials():
    return {"email": "admin@example.com", "password": "adminpass123"}


@pytest.fixture
def logged_in_client(client, admin_credentials):
    response = client.post(
        "/admin/login",
        data={
            "email": admin_credentials["email"],
            "password": admin_credentials["password"],
        },
        follow_redirects=False,
    )
    assert response.status_code in {302, 303}
    return client
