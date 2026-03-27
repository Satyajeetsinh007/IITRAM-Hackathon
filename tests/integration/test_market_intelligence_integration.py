from datetime import timedelta

from app.utils.time_utils import utcnow


def _headers(user_id, role):
    return {
        "X-User-Id": str(user_id),
        "X-User-Role": role,
        "X-CSRFToken": "test-token",
    }


def _seed_admin_trend_data(db):
    now = utcnow()
    farmer = db.users.find_one({"role": "farmer"})
    buyer = db.users.find_one({"role": "buyer"})

    product_id = db.products.insert_one(
        {
            "name": "Fresh Tomato",
            "category": "vegetable",
            "price": 1500,
            "status": "approved",
            "farmer_id": str(farmer["_id"]),
            "seller_email": farmer["email"],
            "created_at": now - timedelta(days=6),
            "updated_at": now - timedelta(days=6),
        }
    ).inserted_id

    offer_docs = []
    for idx in range(3):
        dt = now - timedelta(days=12 - idx)
        offer_docs.append(
            {
                "product_id": str(product_id),
                "buyer_id": f"{buyer['_id']}-prev-{idx}",
                "farmer_id": str(farmer["_id"]),
                "price": 850,
                "quantity": 4,
                "status": "pending",
                "history": [],
                "created_at": dt,
                "updated_at": dt,
                "expires_at": dt + timedelta(days=1),
            }
        )

    for idx in range(8):
        dt = now - timedelta(days=2, hours=idx)
        offer_docs.append(
            {
                "product_id": str(product_id),
                "buyer_id": f"{buyer['_id']}-recent-{idx}",
                "farmer_id": str(farmer["_id"]),
                "price": 1700,
                "quantity": 16,
                "status": "countered",
                "history": [],
                "created_at": dt,
                "updated_at": dt,
                "expires_at": dt + timedelta(days=1),
            }
        )

    db.offers.insert_many(offer_docs)

    for idx in range(40):
        dt = now - timedelta(days=39 - idx)
        db.orders.insert_one(
            {
                "order_number": f"ORD-MI-INT-{idx}",
                "status": "confirmed",
                "total_amount": 1000 + idx,
                "quantity": 5,
                "created_at": dt,
                "updated_at": dt,
            }
        )


def test_admin_forecast_api_requires_auth(client):
    response = client.get("/admin/api/analytics/forecast?horizon=1d")
    assert response.status_code == 401


def test_admin_forecast_api_returns_valid_shape(logged_in_client, db):
    _seed_admin_trend_data(db)

    response = logged_in_client.get("/admin/api/analytics/forecast?horizon=1d")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["horizon"] == "1d"
    assert "predicted_orders" in payload
    assert "confidence" in payload
    assert "forecast_for_date" in payload


def test_admin_forecast_api_invalid_horizon(logged_in_client):
    response = logged_in_client.get("/admin/api/analytics/forecast?horizon=14d")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_admin_refresh_api_runs_and_returns_payload(logged_in_client, db):
    _seed_admin_trend_data(db)

    response = logged_in_client.post(
        "/admin/api/analytics/refresh",
        json={"notify": True},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert "generated_at" in payload
    assert "forecasts" in payload
    assert "notifications_sent" in payload


def test_admin_crop_trends_api_returns_structure(logged_in_client, db):
    _seed_admin_trend_data(db)
    logged_in_client.post("/admin/api/analytics/refresh", json={"notify": False})

    response = logged_in_client.get("/admin/api/analytics/crop-trends?horizon=3d")
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["horizon"] == "3d"
    assert "items" in payload
    assert isinstance(payload["items"], list)


def test_farmer_notifications_api_rejects_non_farmer(client, db):
    buyer = db.users.find_one({"role": "buyer"})
    response = client.get(
        "/farmer/api/notifications",
        headers=_headers(buyer["_id"], "buyer"),
    )
    assert response.status_code == 403


def test_farmer_notifications_list_and_mark_read(client, db):
    now = utcnow()
    farmer = db.users.find_one({"role": "farmer"})
    other_farmer_id = db.users.insert_one(
        {
            "name": "Other Farmer",
            "email": "other_farmer@example.com",
            "role": "farmer",
            "status": "active",
            "created_at": now - timedelta(days=4),
            "updated_at": now - timedelta(days=4),
        }
    ).inserted_id

    mine_id = db.farmer_notifications.insert_one(
        {
            "farmer_id": str(farmer["_id"]),
            "type": "crop_trend_alert",
            "title": "Demand alert: Tomato",
            "message": "Tomato trend is positive",
            "payload": {
                "crop": "tomato",
                "horizon_days": 3,
                "forecast_for_date": (now.date() + timedelta(days=3)).isoformat(),
            },
            "priority": "high",
            "is_read": False,
            "created_at": now,
            "expires_at": now + timedelta(days=7),
            "read_at": None,
        }
    ).inserted_id

    db.farmer_notifications.insert_one(
        {
            "farmer_id": str(other_farmer_id),
            "type": "crop_trend_alert",
            "title": "Demand alert: Onion",
            "message": "Onion trend is positive",
            "payload": {
                "crop": "onion",
                "horizon_days": 3,
                "forecast_for_date": (now.date() + timedelta(days=3)).isoformat(),
            },
            "priority": "medium",
            "is_read": False,
            "created_at": now,
            "expires_at": now + timedelta(days=7),
            "read_at": None,
        }
    )

    list_response = client.get(
        "/farmer/api/notifications",
        headers=_headers(farmer["_id"], "farmer"),
    )
    assert list_response.status_code == 200

    payload = list_response.get_json()
    assert len(payload["items"]) == 1
    assert payload["items"][0]["title"] == "Demand alert: Tomato"

    mark_response = client.patch(
        f"/farmer/api/notifications/{mine_id}/read",
        headers=_headers(farmer["_id"], "farmer"),
    )
    assert mark_response.status_code == 200

    mark_payload = mark_response.get_json()
    assert mark_payload["item"]["is_read"] is True
    assert mark_payload["item"].get("read_at") is not None
