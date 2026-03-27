from pymongo import ASCENDING, DESCENDING


def ensure_indexes(db):
    db.admin_users.create_index([("email", ASCENDING)], unique=True)
    db.users.create_index([("role", ASCENDING), ("status", ASCENDING)])
    db.products.create_index(
        [("status", ASCENDING), ("category", ASCENDING), ("created_at", ASCENDING)]
    )
    db.orders.create_index([("status", ASCENDING), ("created_at", ASCENDING)])
    db.orders.create_index([("created_at", ASCENDING)])
    db.transactions.create_index(
        [("payment_status", ASCENDING), ("created_at", ASCENDING)]
    )
    db.offers.create_index([("product_id", ASCENDING), ("created_at", ASCENDING)])
    db.offers.create_index([("farmer_id", ASCENDING), ("status", ASCENDING)])
    db.offers.create_index([("buyer_id", ASCENDING), ("status", ASCENDING)])
    db.offers.create_index([("status", ASCENDING), ("expires_at", ASCENDING)])
    db.offers.create_index(
        [("product_id", ASCENDING), ("buyer_id", ASCENDING)],
        unique=True,
        partialFilterExpression={"status": {"$in": ["pending", "countered"]}},
        name="uniq_active_offer_per_buyer_product",
    )
    db.admin_activity_logs.create_index([("created_at", ASCENDING)])
    db.admin_alerts.create_index([("is_resolved", ASCENDING), ("triggered_at", ASCENDING)])
    db.admin_alerts.create_index([("type", ASCENDING), ("is_resolved", ASCENDING)])
    db.ml_order_forecasts.create_index(
        [("forecast_for_date", ASCENDING), ("horizon_days", ASCENDING)],
        unique=True,
    )
    db.ml_crop_trends.create_index(
        [("created_at", ASCENDING), ("crop", ASCENDING), ("horizon_days", ASCENDING)]
    )
    db.farmer_notifications.create_index(
        [("farmer_id", ASCENDING), ("is_read", ASCENDING), ("created_at", DESCENDING)]
    )
    db.farmer_notifications.create_index(
        [("farmer_id", ASCENDING), ("type", ASCENDING), ("created_at", ASCENDING)]
    )
