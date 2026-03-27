import logging
import math
import os
import pickle
from datetime import datetime, time, timedelta, timezone

from app.services.exceptions import ServiceError
from app.utils.time_utils import utcnow


logger = logging.getLogger(__name__)

ALLOWED_FORECAST_HORIZONS = {"1d": 1, "3d": 3, "7d": 7}
FORECAST_FEATURES = (
    "day_of_week",
    "sin_dow",
    "cos_dow",
    "lag_1",
    "lag_7",
    "rolling_3",
    "rolling_7",
    "rolling_14",
    "trend_7",
)

try:  # pragma: no cover - optional dependency in test/dev environments
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # pragma: no cover - if sklearn is unavailable
    RandomForestRegressor = None


class MarketIntelligenceService:
    def __init__(
        self,
        repository,
        weather_client,
        refresh_hours=6,
        min_confidence=70,
        min_trend_score=65,
        model_version="ml-lite-v1",
        model_store_path="artifacts/order_forecast_model.pkl",
        min_train_samples=45,
        retrain_hours=24,
    ):
        self.repository = repository
        self.weather_client = weather_client
        self.refresh_hours = max(int(refresh_hours), 1)
        self.min_confidence = max(min(int(min_confidence), 100), 0)
        self.min_trend_score = max(min(int(min_trend_score), 100), 0)
        self.model_version = str(model_version or "ml-lite-v1")
        self.model_store_path = self._resolve_model_store_path(model_store_path)
        self.min_train_samples = max(int(min_train_samples), 20)
        self.retrain_hours = max(int(retrain_hours), 1)
        self._cached_model_bundle = None

    def parse_horizon(self, value):
        key = str(value or "1d").strip().lower()
        if key not in ALLOWED_FORECAST_HORIZONS:
            raise ServiceError("horizon must be one of 1d, 3d, 7d", status_code=400)
        return key, ALLOWED_FORECAST_HORIZONS[key]

    def refresh_insights(self, notify=False):
        now = utcnow()
        try:
            weather = self.weather_client.fetch_daily_forecast(days=7)
        except Exception as exc:  # pragma: no cover - defensive integration guard
            logger.warning("Weather client failed; using fallback", extra={"error": str(exc)})
            weather = {"provider": "fallback", "available": False, "days": []}

        weather_map = {item.get("date"): item for item in weather.get("days", [])}
        weather_available = bool(weather.get("available"))

        order_buckets = self.repository.fetch_order_daily_counts(days=90)
        order_series = self._build_dense_series(order_buckets, days=90)
        model_bundle = self._ensure_trained_order_model(order_series)

        forecasts = []
        for _, horizon_days in ALLOWED_FORECAST_HORIZONS.items():
            forecast = self._forecast_for_horizon(
                order_series=order_series,
                horizon_days=horizon_days,
                weather_map=weather_map,
                model_bundle=model_bundle,
            )
            forecast_for_dt = datetime.combine(
                datetime.fromisoformat(forecast["forecast_for_date"]).date(),
                time.min,
            )
            self.repository.upsert_order_forecast(
                horizon_days=horizon_days,
                forecast_for_date=forecast_for_dt,
                predicted_orders=forecast["predicted_orders"],
                confidence=forecast["confidence"],
                model_version=self.model_version,
                weather=forecast["weather"],
                created_at=now,
            )
            forecasts.append(forecast)

        crop_raw = self.repository.fetch_offer_crop_signals(days=14)
        crop_items = crop_raw.get("items") or []

        trends_by_horizon = {}
        notifications_sent = 0
        for key, horizon_days in ALLOWED_FORECAST_HORIZONS.items():
            trends = self._compute_crop_trends(
                crop_items=crop_items,
                horizon_days=horizon_days,
                weather_available=weather_available,
            )
            self.repository.store_crop_trends(
                horizon_days=horizon_days,
                trends=trends,
                created_at=now,
            )
            trends_by_horizon[key] = trends

            if notify:
                notifications_sent += self._send_trend_notifications(
                    trends=trends,
                    horizon_days=horizon_days,
                    generated_at=now,
                )

        return {
            "generated_at": now.isoformat() + "Z",
            "forecasts": forecasts,
            "trends": trends_by_horizon,
            "notifications_sent": notifications_sent,
        }

    def get_forecast(self, horizon_key):
        key, horizon_days = self.parse_horizon(horizon_key)
        latest = self.repository.latest_order_forecast(horizon_days)
        warning = None

        if self._is_stale():
            try:
                self.refresh_insights(notify=False)
                latest = self.repository.latest_order_forecast(horizon_days)
            except Exception as exc:
                logger.warning("Auto refresh failed", extra={"error": str(exc)})
                if latest:
                    warning = "Using last successful forecast. Auto-refresh failed."
                else:
                    raise ServiceError("No forecast data available", status_code=503) from exc

        if not latest:
            self.refresh_insights(notify=False)
            latest = self.repository.latest_order_forecast(horizon_days)
            if not latest:
                raise ServiceError("No forecast data available", status_code=503)

        forecast_date = latest.get("forecast_for_date")
        generated_at = latest.get("created_at")

        payload = {
            "horizon": key,
            "forecast_for_date": (
                forecast_date.date().isoformat()
                if isinstance(forecast_date, datetime)
                else None
            ),
            "predicted_orders": max(
                int(round(float(latest.get("predicted_orders") or 0))),
                0,
            ),
            "confidence": int(round(float(latest.get("confidence") or 0))),
            "weather": latest.get("weather") or {},
            "generated_at": (
                generated_at.isoformat() + "Z"
                if isinstance(generated_at, datetime)
                else None
            ),
        }
        if warning:
            payload["warning"] = warning
        return payload

    def get_crop_trends(self, horizon_key):
        key, horizon_days = self.parse_horizon(horizon_key)
        warning = None

        if self._is_stale():
            try:
                self.refresh_insights(notify=False)
            except Exception as exc:
                logger.warning("Trend auto refresh failed", extra={"error": str(exc)})
                warning = "Using last successful crop trends. Auto-refresh failed."

        bundle = self.repository.latest_crop_trends(horizon_days)
        created_at = bundle.get("created_at")
        items = bundle.get("items") or []

        response = {
            "horizon": key,
            "items": [
                {
                    "crop": item.get("crop"),
                    "trend_score": round(float(item.get("trend_score") or 0), 2),
                    "confidence": round(float(item.get("confidence") or 0), 2),
                    "signal": item.get("signal"),
                    "demand_growth_pct": round(
                        float(item.get("demand_growth_pct") or 0),
                        2,
                    ),
                    "price_momentum_pct": round(
                        float(item.get("price_momentum_pct") or 0),
                        2,
                    ),
                }
                for item in items
            ],
            "generated_at": (
                created_at.isoformat() + "Z"
                if isinstance(created_at, datetime)
                else None
            ),
        }
        if warning:
            response["warning"] = warning
        return response

    def list_farmer_notifications(self, query_args, actor):
        farmer = self._authorize_farmer(actor)

        try:
            page = max(int(query_args.get("page", 1) or 1), 1)
        except (TypeError, ValueError):
            page = 1
        try:
            per_page = max(int(query_args.get("per_page", 20) or 20), 1)
        except (TypeError, ValueError):
            per_page = 20
        per_page = min(per_page, 100)

        unread_raw = str(query_args.get("unread_only", "")).strip().lower()
        unread_only = unread_raw in {"1", "true", "yes", "on"}

        items, pagination = self.repository.list_farmer_notifications(
            farmer_id=farmer["id"],
            page=page,
            per_page=per_page,
            unread_only=unread_only,
        )
        return {"items": items, "pagination": pagination}

    def mark_notification_read(self, notification_id, actor):
        farmer = self._authorize_farmer(actor)
        updated = self.repository.mark_notification_read(
            notification_id,
            farmer_id=farmer["id"],
        )
        if not updated:
            raise ServiceError("notification not found", status_code=404)
        return updated

    def _authorize_farmer(self, actor):
        role = str(actor.get("role") or "").strip().lower()
        if role != "farmer":
            raise ServiceError("Only farmers can access this endpoint", status_code=403)

        user_id = str(actor.get("user_id") or "").strip()
        farmer = self.repository.find_active_farmer_by_id(user_id)
        if not farmer:
            raise ServiceError("Unauthorized farmer", status_code=401)
        return farmer

    def _is_stale(self):
        latest = self.repository.latest_insight_timestamp()
        if not latest:
            return True
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        return (utcnow() - latest) >= timedelta(hours=self.refresh_hours)

    def _build_dense_series(self, buckets, days=90):
        today = utcnow().date()
        start = today - timedelta(days=max(days, 1) - 1)
        output = []
        for idx in range(days):
            current = start + timedelta(days=idx)
            output.append((current, int(buckets.get(current.isoformat(), 0))))
        return output

    def _forecast_for_horizon(self, order_series, horizon_days, weather_map, model_bundle=None):
        if not order_series:
            target_date = utcnow().date() + timedelta(days=horizon_days)
            return {
                "horizon": f"{horizon_days}d",
                "forecast_for_date": target_date.isoformat(),
                "predicted_orders": 0,
                "confidence": 0,
                "weather": {"available": False},
                "model": self._model_meta(model_bundle),
            }

        target_date = order_series[-1][0] + timedelta(days=horizon_days)
        model_prediction = self._predict_with_model(
            model_bundle,
            order_series,
            horizon_days,
        )

        counts = [count for _, count in order_series]
        used_model = model_prediction is not None
        if model_prediction is None:
            model_prediction = self._heuristic_forecast(order_series, horizon_days)

        weather_day = weather_map.get(target_date.isoformat())
        weather_multiplier = self._weather_multiplier(weather_day)
        predicted_orders = max(int(round(float(model_prediction) * weather_multiplier)), 0)

        slope = self._simple_slope(counts[-14:])
        confidence = self._confidence_score(
            counts=counts,
            slope=slope,
            weather_available=bool(weather_day),
            model_bundle=model_bundle,
            used_model=used_model,
        )

        return {
            "horizon": f"{horizon_days}d",
            "forecast_for_date": target_date.isoformat(),
            "predicted_orders": predicted_orders,
            "confidence": confidence,
            "weather": weather_day or {"available": False},
            "model": self._model_meta(model_bundle),
        }

    def _heuristic_forecast(self, order_series, horizon_days):
        counts = [count for _, count in order_series]
        recent_window = counts[-7:] if len(counts) >= 7 else counts
        recent_avg = (sum(recent_window) / len(recent_window)) if recent_window else 0.0

        target_date = order_series[-1][0] + timedelta(days=horizon_days)
        same_weekday_counts = [
            count for day, count in order_series if day.weekday() == target_date.weekday()
        ]
        weekday_avg = (
            sum(same_weekday_counts) / len(same_weekday_counts)
            if same_weekday_counts
            else recent_avg
        )

        slope = self._simple_slope(counts[-14:])
        trend_projection = max(recent_avg + (slope * horizon_days), 0)
        return (0.55 * recent_avg) + (0.30 * weekday_avg) + (0.15 * trend_projection)

    def _ensure_trained_order_model(self, order_series):
        bundle = self._load_order_model_bundle()
        needs_retrain = self._needs_retraining(bundle)
        if not needs_retrain:
            return bundle

        trained = self._train_order_model(order_series)
        if trained:
            self._save_order_model_bundle(trained)
            self._cached_model_bundle = trained
            return trained
        return bundle

    def _needs_retraining(self, bundle):
        if not bundle:
            return True
        if bundle.get("version") != self.model_version:
            return True

        trained_at = bundle.get("trained_at")
        parsed = self._parse_datetime(trained_at)
        if not parsed:
            return True

        return (utcnow() - parsed) >= timedelta(hours=self.retrain_hours)

    def _train_order_model(self, order_series):
        dataset = self._build_training_dataset(order_series)
        features = dataset["X"]
        targets = dataset["y"]

        if len(features) < self.min_train_samples:
            logger.info(
                "Skipping model training due to insufficient samples",
                extra={
                    "required": self.min_train_samples,
                    "available": len(features),
                },
            )
            return None

        trained_at = utcnow().isoformat()
        if RandomForestRegressor is None:
            logger.warning("scikit-learn unavailable; training weekday baseline model")
            return self._train_weekday_baseline(order_series, trained_at, len(features))

        model = RandomForestRegressor(
            n_estimators=180,
            random_state=42,
            min_samples_leaf=2,
        )
        model.fit(features, targets)

        return {
            "version": self.model_version,
            "trained_at": trained_at,
            "trained_samples": len(features),
            "model_type": "sklearn_random_forest",
            "features": list(FORECAST_FEATURES),
            "model": model,
        }

    def _train_weekday_baseline(self, order_series, trained_at, trained_samples):
        weekday_values = {idx: [] for idx in range(7)}
        counts = []
        for day, count in order_series:
            weekday_values[day.weekday()].append(float(count))
            counts.append(float(count))

        global_baseline = (sum(counts) / len(counts)) if counts else 0.0
        weekday_baseline = {
            str(idx): (
                (sum(values) / len(values)) if values else global_baseline
            )
            for idx, values in weekday_values.items()
        }

        return {
            "version": self.model_version,
            "trained_at": trained_at,
            "trained_samples": int(trained_samples),
            "model_type": "weekday_baseline",
            "features": list(FORECAST_FEATURES),
            "model": None,
            "weekday_baseline": weekday_baseline,
            "global_baseline": global_baseline,
        }

    def _build_training_dataset(self, order_series):
        counts = [float(count) for _, count in order_series]
        rows = []
        targets = []

        for idx in range(14, len(order_series)):
            target_date = order_series[idx][0]
            history = counts[:idx]
            if len(history) < 14:
                continue

            rows.append(self._build_feature_vector(history, target_date))
            targets.append(counts[idx])

        return {"X": rows, "y": targets}

    def _predict_with_model(self, model_bundle, order_series, horizon_days):
        if not model_bundle:
            return None

        model_type = model_bundle.get("model_type")
        history = [float(count) for _, count in order_series]
        if not history:
            return None

        last_date = order_series[-1][0]
        predicted = None

        for step in range(1, horizon_days + 1):
            target_date = last_date + timedelta(days=step)
            features = self._build_feature_vector(history, target_date)
            predicted = self._predict_single(model_bundle, model_type, features, target_date)
            if predicted is None:
                return None
            predicted = max(float(predicted), 0.0)
            history.append(predicted)

        return predicted

    def _predict_single(self, model_bundle, model_type, features, target_date):
        if model_type == "sklearn_random_forest":
            model = model_bundle.get("model")
            if model is None:
                return None
            output = model.predict([features])
            if not output:
                return None
            return float(output[0])

        if model_type == "weekday_baseline":
            weekday_map = model_bundle.get("weekday_baseline") or {}
            fallback = float(model_bundle.get("global_baseline") or 0.0)
            return float(weekday_map.get(str(target_date.weekday()), fallback))

        return None

    def _build_feature_vector(self, history_counts, target_date):
        lag_1 = self._lag(history_counts, 1)
        lag_7 = self._lag(history_counts, 7)
        rolling_3 = self._rolling_mean(history_counts, 3)
        rolling_7 = self._rolling_mean(history_counts, 7)
        rolling_14 = self._rolling_mean(history_counts, 14)
        trend_7 = self._simple_slope(history_counts[-7:] if len(history_counts) >= 7 else history_counts)

        day_of_week = float(target_date.weekday())
        angle = (2.0 * math.pi * day_of_week) / 7.0

        return [
            day_of_week,
            math.sin(angle),
            math.cos(angle),
            lag_1,
            lag_7,
            rolling_3,
            rolling_7,
            rolling_14,
            trend_7,
        ]

    @staticmethod
    def _lag(values, days):
        if not values:
            return 0.0
        if len(values) >= days:
            return float(values[-days])
        return float(values[0])

    @staticmethod
    def _rolling_mean(values, window):
        if not values:
            return 0.0
        subset = values[-window:] if len(values) >= window else values
        return float(sum(subset) / len(subset))

    def _resolve_model_store_path(self, path_value):
        path_text = str(path_value or "artifacts/order_forecast_model.pkl").strip()
        if os.path.isabs(path_text):
            return path_text
        return os.path.abspath(path_text)

    def _load_order_model_bundle(self):
        if self._cached_model_bundle is not None:
            return self._cached_model_bundle

        if not os.path.exists(self.model_store_path):
            return None

        try:
            with open(self.model_store_path, "rb") as fp:
                bundle = pickle.load(fp)
        except Exception as exc:
            logger.warning("Failed to load trained model", extra={"error": str(exc)})
            return None

        if not isinstance(bundle, dict):
            return None

        self._cached_model_bundle = bundle
        return bundle

    def _save_order_model_bundle(self, bundle):
        directory = os.path.dirname(self.model_store_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        temp_path = f"{self.model_store_path}.tmp"
        with open(temp_path, "wb") as fp:
            pickle.dump(bundle, fp)
        os.replace(temp_path, self.model_store_path)

    @staticmethod
    def _parse_datetime(value):
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _model_meta(model_bundle):
        if not model_bundle:
            return {"type": "heuristic", "trained_at": None}

        return {
            "type": model_bundle.get("model_type") or "heuristic",
            "trained_at": model_bundle.get("trained_at"),
            "version": model_bundle.get("version"),
            "trained_samples": int(model_bundle.get("trained_samples") or 0),
        }

    def _weather_multiplier(self, weather_day):
        if not weather_day:
            return 1.0

        precipitation = weather_day.get("precipitation_sum")
        temp_max = weather_day.get("temperature_max")

        multiplier = 1.0
        try:
            if precipitation is not None:
                precipitation = float(precipitation)
                if precipitation >= 12:
                    multiplier -= 0.12
                elif precipitation >= 5:
                    multiplier -= 0.06
        except (TypeError, ValueError):
            pass

        try:
            if temp_max is not None:
                temp_max = float(temp_max)
                if 18 <= temp_max <= 34:
                    multiplier += 0.03
                elif temp_max >= 40 or temp_max <= 8:
                    multiplier -= 0.04
        except (TypeError, ValueError):
            pass

        return max(min(multiplier, 1.2), 0.8)

    def _confidence_score(self, counts, slope, weather_available, model_bundle=None, used_model=False):
        if not counts:
            return 0

        tail = counts[-14:] if len(counts) >= 14 else counts
        avg = (sum(tail) / len(tail)) if tail else 0.0
        if avg <= 0:
            volatility = 1.0
        else:
            variance = sum((point - avg) ** 2 for point in tail) / len(tail)
            volatility = math.sqrt(max(variance, 0.0)) / max(avg, 1.0)

        non_zero_ratio = sum(1 for point in counts if point > 0) / max(len(counts), 1)
        data_component = 42 * non_zero_ratio
        trend_component = max(0.0, 18 - min(abs(slope) * 2.2, 18))
        weather_component = 14 if weather_available else 7
        volatility_penalty = min(volatility * 34, 34)

        model_component = 0.0
        if model_bundle:
            samples = int(model_bundle.get("trained_samples") or 0)
            model_component += min(samples / 4.0, 18)
            if model_bundle.get("model_type") == "sklearn_random_forest":
                model_component += 6
        if used_model:
            model_component += 4

        confidence = (
            data_component
            + trend_component
            + weather_component
            + model_component
            + 16
            - volatility_penalty
        )
        return max(0, min(int(round(confidence)), 100))

    def _compute_crop_trends(self, crop_items, horizon_days, weather_available):
        horizon_scale = {1: 1.0, 3: 0.95, 7: 0.88}.get(horizon_days, 0.9)
        trends = []

        for row in crop_items:
            recent_qty = float(row.get("recent_qty") or 0)
            previous_qty = float(row.get("previous_qty") or 0)

            recent_price = self._safe_average(
                row.get("recent_price_sum"),
                row.get("recent_price_count"),
            )
            previous_price = self._safe_average(
                row.get("previous_price_sum"),
                row.get("previous_price_count"),
            )

            recent_supply = float(row.get("recent_supply") or 0)
            previous_supply = float(row.get("previous_supply") or 0)

            demand_growth = self._pct_change(recent_qty, previous_qty)
            price_momentum = self._pct_change(recent_price, previous_price)
            supply_pressure = self._pct_change(recent_supply, previous_supply)

            demand_score = self._normalize_change(demand_growth)
            price_score = self._normalize_change(price_momentum)
            supply_score = 100 - self._normalize_change(supply_pressure)

            trend_score = (0.5 * demand_score) + (0.3 * price_score) + (0.2 * supply_score)
            trend_score *= horizon_scale

            data_points = int(row.get("recent_offer_count") or 0) + int(row.get("previous_offer_count") or 0)
            volatility = abs(demand_growth - price_momentum)
            confidence = 48 + min(data_points * 5, 28) - min(volatility / 8, 24)
            if weather_available:
                confidence += 10
            confidence *= horizon_scale
            confidence = max(0, min(confidence, 100))

            if trend_score < self.min_trend_score or confidence < self.min_confidence:
                continue

            signal = self._trend_signal(demand_growth, price_momentum)
            trends.append(
                {
                    "crop": str(row.get("crop") or "").strip().lower(),
                    "trend_score": round(trend_score, 2),
                    "confidence": round(confidence, 2),
                    "demand_growth_pct": round(demand_growth, 2),
                    "price_momentum_pct": round(price_momentum, 2),
                    "signal": signal,
                }
            )

        trends.sort(key=lambda item: (item["trend_score"], item["confidence"]), reverse=True)
        return trends

    def _send_trend_notifications(self, trends, horizon_days, generated_at):
        sent = 0
        forecast_for_date = (generated_at.date() + timedelta(days=horizon_days)).isoformat()

        for trend in trends:
            if trend.get("signal") != "uptrend":
                continue

            crop = trend.get("crop")
            if not crop:
                continue

            farmer_ids = self.repository.find_relevant_farmer_ids_for_crop(crop)
            if not farmer_ids:
                continue

            priority = "high" if float(trend.get("trend_score") or 0) >= 85 else "medium"
            title = f"Demand alert: {crop.title()}"
            message = (
                f"{crop.title()} is trending for the next {horizon_days} day(s). "
                f"Estimated trend score {trend.get('trend_score')} with confidence {trend.get('confidence')}%."
            )

            payload = {
                "crop": crop,
                "horizon_days": horizon_days,
                "forecast_for_date": forecast_for_date,
                "trend_score": trend.get("trend_score"),
                "confidence": trend.get("confidence"),
                "demand_growth_pct": trend.get("demand_growth_pct"),
                "price_momentum_pct": trend.get("price_momentum_pct"),
            }

            for farmer_id in farmer_ids:
                _, inserted = self.repository.create_notification_if_absent(
                    farmer_id=farmer_id,
                    notification_type="crop_trend_alert",
                    title=title,
                    message=message,
                    payload=payload,
                    priority=priority,
                    created_at=generated_at,
                    expires_at=generated_at + timedelta(days=max(horizon_days, 1) + 2),
                )
                if inserted:
                    sent += 1

        return sent

    @staticmethod
    def _simple_slope(values):
        if not values or len(values) < 2:
            return 0.0
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        numerator = 0.0
        denominator = 0.0
        for idx, val in enumerate(values):
            dx = idx - x_mean
            dy = val - y_mean
            numerator += dx * dy
            denominator += dx * dx
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _safe_average(total, count):
        try:
            count = int(count or 0)
            total = float(total or 0)
        except (TypeError, ValueError):
            return 0.0
        if count <= 0:
            return 0.0
        return total / count

    @staticmethod
    def _pct_change(current, previous):
        current = float(current or 0)
        previous = float(previous or 0)
        if previous <= 0:
            return 100.0 if current > 0 else 0.0
        return ((current - previous) / previous) * 100.0

    @staticmethod
    def _normalize_change(percent):
        percent = float(percent)
        clamped = max(min(percent, 100.0), -100.0)
        return (clamped + 100.0) / 2.0

    @staticmethod
    def _trend_signal(demand_growth, price_momentum):
        combined = float(demand_growth) + float(price_momentum)
        if combined >= 15:
            return "uptrend"
        if combined <= -15:
            return "downtrend"
        return "stable"
