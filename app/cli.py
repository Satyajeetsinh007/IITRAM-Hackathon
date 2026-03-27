import random
from datetime import timedelta

import click
from flask import current_app
from werkzeug.security import generate_password_hash

from app.repositories.market_intelligence_repository import (
    MarketIntelligenceRepository,
)
from app.repositories.schema import ensure_database_structure
from app.services.analytics_service import compute_quality_score
from app.services.market_intelligence_service import MarketIntelligenceService
from app.services.weather_client import WeatherClient
from app.utils.time_utils import utcnow

FARM_PRODUCTS = {
    "grain": ["Wheat", "Rice", "Maize", "Bajra", "Barley"],
    "vegetable": ["Tomato", "Potato", "Onion", "Brinjal", "Cauliflower", "Peas"],
    "fruit": ["Mango", "Banana", "Guava", "Papaya", "Pomegranate"],
    "pulse": ["Chickpea", "Lentil", "Green Gram", "Black Gram", "Toor Dal"],
    "spice": ["Cumin", "Turmeric", "Coriander", "Chili", "Fenugreek"],
}

ORDER_STATUSES = ["pending", "confirmed", "packed", "shipped", "delivered", "cancelled"]
PAYMENT_STATUSES = ["pending", "paid", "failed", "refunded"]
REVIEW_STATES = ["pending_review", "reviewed", "flagged"]
USER_STATUSES = ["active", "active", "active", "inactive", "suspended"]


def _random_ts(days_back=120):
    now = utcnow()
    back_days = random.randint(0, max(days_back - 1, 0))
    back_minutes = random.randint(0, 24 * 60 - 1)
    return now - timedelta(days=back_days, minutes=back_minutes)


def _weighted_order_status():
    roll = random.random()
    if roll < 0.1:
        return "pending"
    if roll < 0.2:
        return "confirmed"
    if roll < 0.3:
        return "packed"
    if roll < 0.45:
        return "shipped"
    if roll < 0.9:
        return "delivered"
    return "cancelled"


def _payment_for_order_status(order_status):
    if order_status == "cancelled":
        return random.choice(["failed", "refunded", "pending"])
    if order_status in {"shipped", "delivered"}:
        return random.choice(["paid", "paid", "paid", "pending"])
    return random.choice(["pending", "paid", "failed"])


def _review_state_for_payment(payment_status):
    if payment_status in {"failed", "refunded"}:
        return random.choice(["flagged", "pending_review", "reviewed"])
    return random.choice(["reviewed", "pending_review"])


def _build_demo_users(count):
    first_names = [
        "Aarav",
        "Vihaan",
        "Saanvi",
        "Ananya",
        "Riya",
        "Arjun",
        "Karan",
        "Meera",
        "Ishaan",
        "Neha",
    ]
    last_names = [
        "Patel",
        "Sharma",
        "Singh",
        "Yadav",
        "Prajapati",
        "Desai",
        "Kumar",
        "Rathod",
        "Chauhan",
        "Joshi",
    ]

    users = []
    for i in range(count):
        created_at = _random_ts(days_back=120)
        role = "farmer" if i < int(count * 0.55) else "buyer"
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        email = f"{name.lower().replace(' ', '.')}.{i}@demo.local"
        users.append(
            {
                "name": name,
                "email": email,
                "role": role,
                "status": random.choice(USER_STATUSES),
                "created_at": created_at,
                "updated_at": created_at,
            }
        )
    return users


def _build_demo_products(count, farmers):
    quality_tags = ["Premium", "Organic", "Fresh", "Export Grade", "Sorted"]
    products = []
    catalog = [(category, item) for category, items in FARM_PRODUCTS.items() for item in items]

    for i in range(count):
        created_at = _random_ts(days_back=120)
        category, item_name = random.choice(catalog)
        farmer = random.choice(farmers) if farmers else None
        status = random.choices(
            ["approved", "pending", "hidden", "rejected"],
            weights=[70, 15, 10, 5],
            k=1,
        )[0]
        review_note = ""
        if status == "approved":
            review_note = "Verified quality and quantity."
        elif status == "rejected":
            review_note = "Image and quantity mismatch."
        elif status == "hidden":
            review_note = "Temporarily hidden due to stock issue."

        products.append(
            {
                "name": f"{random.choice(quality_tags)} {item_name}",
                "description": f"{item_name} sourced from local farms.",
                "category": category,
                "price": random.randint(500, 7000),
                "stock_qty": random.randint(20, 600),
                "unit": random.choice(["kg", "quintal", "crate"]),
                "image_url": f"https://example.com/images/{item_name.lower().replace(' ', '-')}-{i}.jpg",
                "seller_verified": random.choice([True, True, False]),
                "status": status,
                "seller_name": farmer.get("name") if farmer else None,
                "seller_email": farmer.get("email") if farmer else None,
                "review_note": review_note,
                "reviewed_by": None,
                "reviewed_at": created_at if status in {"approved", "rejected", "hidden"} else None,
                "created_at": created_at,
                "updated_at": created_at,
            }
        )

        products[-1]["quality_score"] = compute_quality_score(products[-1])

    return products


def _build_demo_orders_and_transactions(count, buyers):
    orders = []
    transactions = []
    now = utcnow()

    for i in range(count):
        created_at = _random_ts(days_back=90)
        buyer = random.choice(buyers) if buyers else {"name": f"Buyer {i+1}", "email": f"buyer{i+1}@demo.local"}
        status = _weighted_order_status()
        total_amount = random.randint(1200, 25000)
        quantity = random.randint(5, 120)
        order_number = f"ORD-{created_at.strftime('%Y%m%d')}-{1000 + i}"

        orders.append(
            {
                "order_number": order_number,
                "buyer_name": buyer.get("name"),
                "buyer_email": buyer.get("email"),
                "status": status,
                "quantity": quantity,
                "total_price": total_amount,
                "total_amount": total_amount,
                "delivery_address": random.choice(
                    [
                        "Ahmedabad Central Market",
                        "Surat Fresh Hub",
                        "Vadodara Retail Zone",
                        "Rajkot Supply Point",
                    ]
                ),
                "created_at": created_at,
                "updated_at": min(now, created_at + timedelta(days=random.randint(0, 5))),
            }
        )

        payment_status = _payment_for_order_status(status)
        transactions.append(
            {
                "transaction_ref": f"TXN-{created_at.strftime('%y%m%d')}-{20000 + i}",
                "order_number": order_number,
                "amount": total_amount,
                "payment_status": payment_status,
                "review_state": _review_state_for_payment(payment_status),
                "gateway": random.choice(["razorpay", "stripe", "upi"]),
                "created_at": created_at + timedelta(minutes=random.randint(5, 180)),
                "updated_at": created_at + timedelta(days=random.randint(0, 3)),
            }
        )

    return orders, transactions


def _build_demo_activity_logs(admin_id, users, products, orders, transactions, count=120):
    actions = [
        ("user_status_updated", "user"),
        ("product_reviewed", "product"),
        ("order_status_updated", "order"),
        ("transaction_state_updated", "transaction"),
        ("admin_login", "admin_user"),
    ]

    logs = []
    entity_lookup = {
        "user": users,
        "product": products,
        "order": orders,
        "transaction": transactions,
    }

    for _ in range(count):
        action, entity_type = random.choice(actions)
        created_at = _random_ts(days_back=40)
        entity_id = None
        details = {}

        if entity_type in entity_lookup and entity_lookup[entity_type]:
            entity = random.choice(entity_lookup[entity_type])
            entity_id = str(entity.get("_id")) if entity.get("_id") else None
            if entity_type == "order":
                details = {"new_status": entity.get("status")}
            elif entity_type == "transaction":
                details = {"payment_status": entity.get("payment_status")}
            elif entity_type == "product":
                details = {"status": entity.get("status")}
            elif entity_type == "user":
                details = {"new_status": entity.get("status")}

        logs.append(
            {
                "admin_id": str(admin_id) if admin_id else None,
                "action": action,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "details": details,
                "created_at": created_at,
            }
        )

    logs.sort(key=lambda item: item["created_at"])
    return logs


def _extract_crop_key(name):
    tokens = str(name or "").strip().lower().split()
    return tokens[-1] if tokens else ""


def _build_trend_seed_offers(products, farmers, buyers, offers_per_crop=18):
    if not products or not farmers or not buyers:
        return []

    now = utcnow()
    offers_per_crop = max(int(offers_per_crop), 8)
    farmer_by_email = {
        str(farmer.get("email") or "").strip().lower(): str(farmer["_id"])
        for farmer in farmers
        if farmer.get("_id")
    }

    crop_products = {}
    for product in products:
        crop = _extract_crop_key(product.get("name"))
        if not crop:
            continue
        crop_products.setdefault(crop, []).append(product)

    if not crop_products:
        return []

    top_crops = sorted(
        crop_products.keys(),
        key=lambda key: len(crop_products[key]),
        reverse=True,
    )[:3]

    rows = []

    def resolve_farmer_id(product_doc):
        seller_email = str(product_doc.get("seller_email") or "").strip().lower()
        if seller_email and seller_email in farmer_by_email:
            return farmer_by_email[seller_email]
        owner_id = product_doc.get("farmer_id") or product_doc.get("seller_id")
        if owner_id:
            return str(owner_id)
        fallback = random.choice(farmers)
        return str(fallback["_id"])

    def make_offer(product_doc, buyer_doc, farmer_id, price, quantity, updated_at):
        created_at = updated_at - timedelta(hours=random.randint(2, 36))
        history = [
            {
                "event": "created",
                "actor_id": str(buyer_doc["_id"]),
                "actor_role": "buyer",
                "price": float(price),
                "quantity": int(quantity),
                "note": "Demo trend seed offer",
                "at": created_at,
            },
            {
                "event": "accepted",
                "actor_id": str(farmer_id),
                "actor_role": "farmer",
                "price": float(price),
                "quantity": int(quantity),
                "note": "Accepted for trend simulation",
                "at": updated_at,
            },
        ]
        return {
            "product_id": str(product_doc["_id"]),
            "buyer_id": str(buyer_doc["_id"]),
            "farmer_id": str(farmer_id),
            "price": float(price),
            "quantity": int(max(quantity, 1)),
            "status": "accepted",
            "history": history,
            "created_at": created_at,
            "updated_at": updated_at,
            "expires_at": created_at + timedelta(days=5),
            "seed_tag": "trend_demo_v1",
        }

    for crop in top_crops:
        products_for_crop = crop_products.get(crop) or []
        if not products_for_crop:
            continue

        previous_count = offers_per_crop
        recent_count = offers_per_crop

        base_qty = random.randint(10, 18)
        base_price = random.randint(900, 1800)
        qty_growth = random.uniform(1.35, 1.68)
        price_growth = random.uniform(1.30, 1.64)

        for _ in range(previous_count):
            product = random.choice(products_for_crop)
            buyer = random.choice(buyers)
            farmer_id = resolve_farmer_id(product)
            updated_at = now - timedelta(
                days=random.randint(15, 27),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            qty = max(int(round(base_qty + random.uniform(-2, 3))), 1)
            price = max(int(round(base_price + random.uniform(-120, 180))), 1)
            rows.append(make_offer(product, buyer, farmer_id, price, qty, updated_at))

        for _ in range(recent_count):
            product = random.choice(products_for_crop)
            buyer = random.choice(buyers)
            farmer_id = resolve_farmer_id(product)
            updated_at = now - timedelta(
                days=random.randint(0, 13),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            qty = max(
                int(round((base_qty * qty_growth) + random.uniform(-2, 4))),
                1,
            )
            price = max(
                int(round((base_price * price_growth) + random.uniform(-150, 260))),
                1,
            )
            rows.append(make_offer(product, buyer, farmer_id, price, qty, updated_at))

    return rows


def _build_market_intelligence_service(config, db):
    return MarketIntelligenceService(
        MarketIntelligenceRepository(db),
        weather_client=WeatherClient(
            provider=config.get("WEATHER_PROVIDER", "open_meteo"),
            latitude=config.get("WEATHER_LAT", 23.0225),
            longitude=config.get("WEATHER_LON", 72.5714),
            timeout_sec=config.get("WEATHER_TIMEOUT_SEC", 4),
        ),
        refresh_hours=config.get("AI_REFRESH_HOURS", 6),
        min_confidence=config.get("AI_MIN_CONFIDENCE", 70),
        min_trend_score=config.get("AI_MIN_TREND_SCORE", 65),
        model_version=config.get("AI_MODEL_VERSION", "ml-lite-v1"),
        model_store_path=config.get(
            "AI_MODEL_STORE_PATH",
            "artifacts/order_forecast_model.pkl",
        ),
        min_train_samples=config.get("AI_MIN_TRAIN_SAMPLES", 45),
        retrain_hours=config.get("AI_RETRAIN_HOURS", 24),
    )


def register_commands(app):
    @app.cli.command("init-db")
    @click.option(
        "--skip-validators",
        is_flag=True,
        default=False,
        help="Create collections and indexes without applying Mongo validators.",
    )
    def init_db(skip_validators):
        summary = ensure_database_structure(
            current_app.db,
            apply_validators=not skip_validators,
        )

        click.echo("Database initialization complete:")
        for collection_name, status in summary.items():
            click.echo(f"- {collection_name}: {status}")

    @app.cli.command("seed-admin")
    @click.option("--email", required=True, help="Admin email")
    @click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
    def seed_admin(email, password):
        admin_users = current_app.db.admin_users
        now = utcnow()
        existing = admin_users.find_one({"email": email.lower()})

        payload = {
            "email": email.lower(),
            "password_hash": generate_password_hash(password),
            "status": "active",
            "updated_at": now,
        }

        if existing:
            admin_users.update_one({"_id": existing["_id"]}, {"$set": payload})
            click.echo("Updated existing admin user")
            return

        payload["created_at"] = now
        payload["last_login_at"] = None
        admin_users.insert_one(payload)
        click.echo("Created admin user")

    @app.cli.command("seed-demo-data")
    @click.option("--users", "users_count", default=80, show_default=True, type=int)
    @click.option("--products", "products_count", default=120, show_default=True, type=int)
    @click.option("--orders", "orders_count", default=220, show_default=True, type=int)
    @click.option(
        "--activity-logs",
        "activity_count",
        default=120,
        show_default=True,
        type=int,
    )
    @click.option("--offers", "offers_count", default=160, show_default=True, type=int)
    @click.option(
        "--reset",
        is_flag=True,
        default=False,
        help=(
            "Delete existing users/products/orders/transactions/offers/activity logs "
            "before seeding."
        ),
    )
    @click.option(
        "--refresh-ai/--no-refresh-ai",
        default=True,
        show_default=True,
        help="Run AI refresh immediately so forecast/trending panels use fresh seeded data.",
    )
    def seed_demo_data(
        users_count,
        products_count,
        orders_count,
        activity_count,
        offers_count,
        reset,
        refresh_ai,
    ):
        db = current_app.db
        ensure_database_structure(db, apply_validators=True)

        if reset:
            db.users.delete_many({})
            db.products.delete_many({})
            db.orders.delete_many({})
            db.transactions.delete_many({})
            db.offers.delete_many({})
            db.admin_activity_logs.delete_many({})
            db.ml_order_forecasts.delete_many({})
            db.ml_crop_trends.delete_many({})
            db.farmer_notifications.delete_many({})
            click.echo("Cleared existing marketplace demo collections.")

        users = _build_demo_users(users_count)
        if users:
            user_insert = db.users.insert_many(users)
            for idx, oid in enumerate(user_insert.inserted_ids):
                users[idx]["_id"] = oid

        farmers = [user for user in users if user.get("role") == "farmer"]
        buyers = [user for user in users if user.get("role") == "buyer"]

        products = _build_demo_products(products_count, farmers)
        if products:
            product_insert = db.products.insert_many(products)
            for idx, oid in enumerate(product_insert.inserted_ids):
                products[idx]["_id"] = oid

        orders, transactions = _build_demo_orders_and_transactions(orders_count, buyers)
        if orders:
            db.orders.insert_many(orders)
        if transactions:
            db.transactions.insert_many(transactions)

        offers = _build_trend_seed_offers(
            products=[product for product in products if product.get("category") == "vegetable"],
            farmers=farmers,
            buyers=buyers,
            offers_per_crop=max(offers_count // 3, 8),
        )
        if offers:
            db.offers.insert_many(offers)

        admin = db.admin_users.find_one({}, sort=[("created_at", 1)])
        logs = _build_demo_activity_logs(
            admin_id=admin.get("_id") if admin else None,
            users=users,
            products=products,
            orders=orders,
            transactions=transactions,
            count=activity_count,
        )
        if logs:
            db.admin_activity_logs.insert_many(logs)

        click.echo("Demo data seeded successfully:")
        click.echo(f"- users: {len(users)}")
        click.echo(f"- products: {len(products)}")
        click.echo(f"- orders: {len(orders)}")
        click.echo(f"- transactions: {len(transactions)}")
        click.echo(f"- offers: {len(offers)}")
        click.echo(f"- activity logs: {len(logs)}")

        if refresh_ai:
            config = current_app.config
            service = _build_market_intelligence_service(config, current_app.db)
            result = service.refresh_insights(notify=False)
            click.echo("AI insights refreshed after seeding:")
            click.echo(f"- generated_at: {result.get('generated_at')}")
            click.echo(
                f"- trend_items_1d: {len((result.get('trends') or {}).get('1d') or [])}"
            )

    @app.cli.command("seed-trend-offers")
    @click.option(
        "--offers-per-crop",
        default=20,
        show_default=True,
        type=int,
        help="How many recent offers to create per top vegetable crop.",
    )
    @click.option(
        "--reset",
        is_flag=True,
        default=False,
        help="Delete only previously seeded trend offers (seed_tag=trend_demo_v1).",
    )
    @click.option(
        "--refresh-ai/--no-refresh-ai",
        default=True,
        show_default=True,
        help="Refresh AI insights after offer seeding.",
    )
    def seed_trend_offers(offers_per_crop, reset, refresh_ai):
        """Seed offer signals so Trending Vegetables panel has meaningful demo data."""
        db = current_app.db
        ensure_database_structure(db, apply_validators=True)

        if reset:
            removed = db.offers.delete_many({"seed_tag": "trend_demo_v1"})
            click.echo(f"Removed {removed.deleted_count} previously seeded trend offers.")

        farmers = list(
            db.users.find(
                {"role": "farmer"},
                {"_id": 1, "email": 1, "role": 1, "status": 1},
            )
        )
        buyers = list(
            db.users.find(
                {"role": "buyer"},
                {"_id": 1, "email": 1, "role": 1, "status": 1},
            )
        )
        products = list(
            db.products.find(
                {"category": "vegetable"},
                {"_id": 1, "name": 1, "category": 1, "seller_email": 1},
            )
        )

        if not farmers or not buyers or not products:
            click.echo(
                "Seed users/products first. Required: farmers, buyers, and vegetable products."
            )
            return

        rows = _build_trend_seed_offers(
            products=products,
            farmers=farmers,
            buyers=buyers,
            offers_per_crop=offers_per_crop,
        )
        if not rows:
            click.echo("No trend offers generated from current dataset.")
            return

        db.offers.insert_many(rows)
        click.echo(f"Inserted {len(rows)} trend-seed offers.")

        if refresh_ai:
            service = _build_market_intelligence_service(current_app.config, db)
            result = service.refresh_insights(notify=False)
            click.echo("AI insights refreshed:")
            click.echo(f"- generated_at: {result.get('generated_at')}")
            click.echo(
                f"- trend_items_1d: {len((result.get('trends') or {}).get('1d') or [])}"
            )

    @app.cli.command("seed-3d-data")
    @click.option("--count", default=140, show_default=True, type=int)
    @click.option("--days", default=30, show_default=True, type=int)
    @click.option(
        "--reset-only-3d",
        is_flag=True,
        default=False,
        help="Remove only synthetic 3D demo orders before insert.",
    )
    def seed_3d_data(count, days, reset_only_3d):
        """Seed synthetic orders with quantity/total_price for 3D analytics chart."""
        db = current_app.db
        now = utcnow()
        buyers = list(db.users.find({"role": "buyer"}))
        if not buyers:
            click.echo("No buyers found. Seed users first (seed-demo-data).")
            return

        if reset_only_3d:
            db.orders.delete_many({"order_number": {"$regex": r"^ORD-3D-"}})

        rows = []
        for idx in range(max(count, 1)):
            buyer = random.choice(buyers)
            created_at = now - timedelta(
                days=random.randint(0, max(days - 1, 0)),
                minutes=random.randint(0, 23 * 60 + 59),
            )
            quantity = random.randint(1, 160)
            unit_price = random.randint(120, 900)
            total_price = quantity * unit_price

            rows.append(
                {
                    "order_number": f"ORD-3D-{created_at.strftime('%Y%m%d')}-{10000 + idx}",
                    "buyer_name": buyer.get("name"),
                    "buyer_email": buyer.get("email"),
                    "status": random.choice(["confirmed", "packed", "shipped", "delivered"]),
                    "quantity": quantity,
                    "total_price": total_price,
                    "total_amount": total_price,
                    "created_at": created_at,
                    "updated_at": created_at + timedelta(hours=random.randint(1, 48)),
                }
            )

        if rows:
            db.orders.insert_many(rows)
        click.echo(f"Seeded {len(rows)} 3D analytics order rows.")

    @app.cli.command("refresh-ai-insights")
    @click.option(
        "--notify/--no-notify",
        default=False,
        show_default=True,
        help="Create farmer notifications from high-confidence crop trends.",
    )
    def refresh_ai_insights(notify):
        """Refresh AI forecasts and crop trends, then optionally notify farmers."""
        service = _build_market_intelligence_service(current_app.config, current_app.db)

        result = service.refresh_insights(notify=notify)
        click.echo("AI insights refreshed successfully:")
        click.echo(f"- generated_at: {result.get('generated_at')}")
        click.echo(f"- notifications_sent: {result.get('notifications_sent', 0)}")
