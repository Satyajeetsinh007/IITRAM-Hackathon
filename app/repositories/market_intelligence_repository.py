import re
from datetime import timedelta, timezone

from pymongo import ReturnDocument

from app.utils.pagination import build_pagination_meta
from app.utils.serialization import serialize_document, to_object_id
from app.utils.time_utils import utcnow


class MarketIntelligenceRepository:
    def __init__(self, db):
        self.db = db
        self.orders = db.orders
        self.offers = db.offers
        self.products = db.products
        self.users = db.users
        self.forecasts = db.ml_order_forecasts
        self.crop_trends = db.ml_crop_trends
        self.notifications = db.farmer_notifications

    def fetch_order_daily_counts(self, days=90):
        now = self._as_naive_utc(utcnow())
        start = now - timedelta(days=max(days, 1) - 1)
        docs = self.orders.find(
            {"created_at": {"$gte": start, "$lte": now}},
            {"_id": 0, "created_at": 1},
        )

        buckets = {}
        for doc in docs:
            created_at = doc.get("created_at")
            if not created_at:
                continue
            key = created_at.date().isoformat()
            buckets[key] = buckets.get(key, 0) + 1

        return buckets

    def fetch_offer_crop_signals(self, days=14):
        now = self._as_naive_utc(utcnow())
        window_days = max(days, 1)
        recent_start = now - timedelta(days=window_days)
        previous_start = recent_start - timedelta(days=window_days)

        projections = {
            "_id": 0,
            "product_id": 1,
            "price": 1,
            "quantity": 1,
            "created_at": 1,
            "updated_at": 1,
            "status": 1,
        }
        offers = list(
            self.offers.find(
                {
                    "$or": [
                        {"updated_at": {"$gte": previous_start}},
                        {"created_at": {"$gte": previous_start}},
                    ]
                },
                projections,
            )
        )

        product_object_ids = []
        for offer in offers:
            oid = to_object_id(offer.get("product_id"))
            if oid:
                product_object_ids.append(oid)

        products_map = {}
        if product_object_ids:
            product_docs = self.products.find(
                {"_id": {"$in": product_object_ids}, "category": "vegetable"},
                {"_id": 1, "name": 1, "category": 1, "created_at": 1},
            )
            for doc in product_docs:
                products_map[str(doc["_id"])] = doc

        metrics = {}

        def get_bucket(crop):
            if crop not in metrics:
                metrics[crop] = {
                    "crop": crop,
                    "recent_qty": 0.0,
                    "previous_qty": 0.0,
                    "recent_price_sum": 0.0,
                    "recent_price_count": 0,
                    "previous_price_sum": 0.0,
                    "previous_price_count": 0,
                    "recent_supply": 0,
                    "previous_supply": 0,
                    "recent_offer_count": 0,
                    "previous_offer_count": 0,
                }
            return metrics[crop]

        for offer in offers:
            product = products_map.get(str(offer.get("product_id")))
            if not product:
                continue

            crop = self._extract_crop_name(product.get("name") or "")
            if not crop:
                continue

            event_time = self._as_naive_utc(offer.get("updated_at") or offer.get("created_at"))
            if not event_time or event_time < previous_start:
                continue

            bucket = get_bucket(crop)
            period = "recent" if event_time >= recent_start else "previous"

            quantity = offer.get("quantity")
            price = offer.get("price")
            try:
                quantity = float(quantity)
            except (TypeError, ValueError):
                quantity = 0.0
            try:
                price = float(price)
            except (TypeError, ValueError):
                price = 0.0

            if quantity > 0:
                if period == "recent":
                    bucket["recent_qty"] += quantity
                else:
                    bucket["previous_qty"] += quantity

            if price > 0:
                if period == "recent":
                    bucket["recent_price_sum"] += price
                    bucket["recent_price_count"] += 1
                else:
                    bucket["previous_price_sum"] += price
                    bucket["previous_price_count"] += 1

            if period == "recent":
                bucket["recent_offer_count"] += 1
            else:
                bucket["previous_offer_count"] += 1

        supply_products = self.products.find(
            {
                "category": "vegetable",
                "created_at": {"$gte": previous_start, "$lte": now},
            },
            {"_id": 0, "name": 1, "created_at": 1},
        )

        for product in supply_products:
            created_at = self._as_naive_utc(product.get("created_at"))
            if not created_at:
                continue
            crop = self._extract_crop_name(product.get("name") or "")
            if not crop:
                continue

            bucket = get_bucket(crop)
            if created_at >= recent_start:
                bucket["recent_supply"] += 1
            else:
                bucket["previous_supply"] += 1

        return {
            "generated_at": now,
            "recent_start": recent_start,
            "previous_start": previous_start,
            "items": list(metrics.values()),
        }

    @staticmethod
    def _as_naive_utc(value):
        if not value:
            return None
        if getattr(value, "tzinfo", None) is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def _extract_crop_name(self, value):
        text = str(value or "").strip().lower()
        if not text:
            return ""
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = [token for token in text.split() if token]
        if not tokens:
            return ""
        return tokens[-1]

    def find_active_farmer_by_id(self, farmer_id):
        oid = to_object_id(farmer_id)
        if not oid:
            return None
        farmer = self.users.find_one({"_id": oid, "role": "farmer", "status": "active"})
        return serialize_document(farmer)

    def find_relevant_farmer_ids_for_crop(self, crop):
        crop = str(crop or "").strip()
        if not crop:
            return []

        pattern = re.escape(crop)
        product_docs = list(
            self.products.find(
                {
                    "category": "vegetable",
                    "name": {"$regex": pattern, "$options": "i"},
                },
                {
                    "_id": 0,
                    "farmer_id": 1,
                    "seller_id": 1,
                    "owner_id": 1,
                    "user_id": 1,
                    "seller_email": 1,
                    "owner_email": 1,
                    "farmer_email": 1,
                },
            )
        )

        farmer_ids = set()
        user_id_cache = {}
        email_cache = {}

        id_fields = ("farmer_id", "seller_id", "owner_id", "user_id")
        email_fields = ("seller_email", "owner_email", "farmer_email")

        for product in product_docs:
            for field in id_fields:
                raw_id = product.get(field)
                if not raw_id:
                    continue
                if raw_id in user_id_cache:
                    maybe_id = user_id_cache[raw_id]
                else:
                    farmer = self.find_active_farmer_by_id(raw_id)
                    maybe_id = farmer.get("id") if farmer else None
                    user_id_cache[raw_id] = maybe_id
                if maybe_id:
                    farmer_ids.add(maybe_id)

            for field in email_fields:
                email = str(product.get(field) or "").strip().lower()
                if not email:
                    continue
                if email in email_cache:
                    maybe_id = email_cache[email]
                else:
                    doc = self.users.find_one(
                        {"email": email, "role": "farmer", "status": "active"},
                        {"_id": 1},
                    )
                    maybe_id = str(doc["_id"]) if doc else None
                    email_cache[email] = maybe_id
                if maybe_id:
                    farmer_ids.add(maybe_id)

        return sorted(farmer_ids)

    def upsert_order_forecast(
        self,
        horizon_days,
        forecast_for_date,
        predicted_orders,
        confidence,
        model_version,
        weather,
        created_at,
    ):
        self.forecasts.update_one(
            {
                "horizon_days": int(horizon_days),
                "forecast_for_date": forecast_for_date,
            },
            {
                "$set": {
                    "predicted_orders": float(predicted_orders),
                    "confidence": float(confidence),
                    "model_version": model_version,
                    "weather": weather,
                    "created_at": created_at,
                }
            },
            upsert=True,
        )

    def latest_order_forecast(self, horizon_days):
        doc = self.forecasts.find_one(
            {"horizon_days": int(horizon_days)},
            sort=[("created_at", -1)],
        )
        return serialize_document(doc)

    def latest_insight_timestamp(self):
        latest_forecast = self.forecasts.find_one({}, sort=[("created_at", -1)])
        latest_trend = self.crop_trends.find_one({}, sort=[("created_at", -1)])

        timestamps = []
        if latest_forecast and latest_forecast.get("created_at"):
            timestamps.append(latest_forecast["created_at"])
        if latest_trend and latest_trend.get("created_at"):
            timestamps.append(latest_trend["created_at"])

        return max(timestamps) if timestamps else None

    def store_crop_trends(self, horizon_days, trends, created_at):
        self.crop_trends.delete_many({"horizon_days": int(horizon_days)})
        if not trends:
            return
        rows = []
        for trend in trends:
            rows.append(
                {
                    "horizon_days": int(horizon_days),
                    "crop": trend["crop"],
                    "trend_score": float(trend["trend_score"]),
                    "confidence": float(trend["confidence"]),
                    "demand_growth_pct": float(trend["demand_growth_pct"]),
                    "price_momentum_pct": float(trend["price_momentum_pct"]),
                    "signal": trend["signal"],
                    "created_at": created_at,
                }
            )
        if rows:
            self.crop_trends.insert_many(rows)

    def latest_crop_trends(self, horizon_days):
        latest = self.crop_trends.find_one(
            {"horizon_days": int(horizon_days)},
            sort=[("created_at", -1)],
            projection={"created_at": 1},
        )
        if not latest:
            return {"items": [], "created_at": None}

        created_at = latest.get("created_at")
        items = list(
            self.crop_trends.find(
                {
                    "horizon_days": int(horizon_days),
                    "created_at": created_at,
                },
                {"_id": 0},
            ).sort("trend_score", -1)
        )
        return {"items": items, "created_at": created_at}

    def create_notification_if_absent(
        self,
        farmer_id,
        notification_type,
        title,
        message,
        payload,
        priority,
        created_at,
        expires_at,
    ):
        day_start = created_at.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        query = {
            "farmer_id": str(farmer_id),
            "type": str(notification_type),
            "created_at": {"$gte": day_start, "$lt": day_end},
            "payload.crop": payload.get("crop"),
            "payload.horizon_days": payload.get("horizon_days"),
            "payload.forecast_for_date": payload.get("forecast_for_date"),
        }

        existing = self.notifications.find_one(query)
        if existing:
            return serialize_document(existing), False

        result = self.notifications.insert_one(
            {
                "farmer_id": str(farmer_id),
                "type": str(notification_type),
                "title": str(title),
                "message": str(message),
                "payload": payload,
                "priority": str(priority),
                "is_read": False,
                "created_at": created_at,
                "expires_at": expires_at,
                "read_at": None,
            }
        )
        inserted = self.notifications.find_one({"_id": result.inserted_id})
        return serialize_document(inserted), True

    def list_farmer_notifications(self, farmer_id, page, per_page, unread_only=False):
        query = {"farmer_id": str(farmer_id)}
        if unread_only:
            query["is_read"] = False

        skip = (page - 1) * per_page
        cursor = (
            self.notifications.find(query)
            .sort("created_at", -1)
            .skip(skip)
            .limit(per_page)
        )
        items = [serialize_document(doc) for doc in cursor]
        total = self.notifications.count_documents(query)
        return items, build_pagination_meta(page, per_page, total)

    def mark_notification_read(self, notification_id, farmer_id):
        oid = to_object_id(notification_id)
        if not oid:
            return None

        updated = self.notifications.find_one_and_update(
            {
                "_id": oid,
                "farmer_id": str(farmer_id),
            },
            {
                "$set": {
                    "is_read": True,
                    "read_at": utcnow(),
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        return serialize_document(updated)
