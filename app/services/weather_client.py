import json
import logging
from datetime import datetime
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


logger = logging.getLogger(__name__)


class WeatherClient:
    def __init__(self, provider="open_meteo", latitude=23.0225, longitude=72.5714, timeout_sec=4):
        self.provider = str(provider or "open_meteo").strip().lower()
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.timeout_sec = int(timeout_sec)

    def fetch_daily_forecast(self, days=7):
        if self.provider != "open_meteo":
            logger.warning("Unsupported weather provider", extra={"provider": self.provider})
            return {"provider": self.provider, "available": False, "days": []}

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": "UTC",
            "forecast_days": max(1, min(int(days), 16)),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        }
        url = f"https://api.open-meteo.com/v1/forecast?{urlencode(params)}"

        try:
            with urlopen(url, timeout=self.timeout_sec) as response:  # nosec B310
                payload = json.loads(response.read().decode("utf-8"))
        except (URLError, TimeoutError, ValueError, OSError) as exc:
            logger.warning("Weather fetch failed", extra={"error": str(exc)})
            return {"provider": self.provider, "available": False, "days": []}

        daily = payload.get("daily") or {}
        times = daily.get("time") or []
        tmax = daily.get("temperature_2m_max") or []
        tmin = daily.get("temperature_2m_min") or []
        rain = daily.get("precipitation_sum") or []

        days_out = []
        for idx, day in enumerate(times):
            try:
                datetime.fromisoformat(day)
            except ValueError:
                continue

            days_out.append(
                {
                    "date": day,
                    "temperature_max": self._as_float(tmax, idx),
                    "temperature_min": self._as_float(tmin, idx),
                    "precipitation_sum": self._as_float(rain, idx),
                }
            )

        return {
            "provider": self.provider,
            "available": bool(days_out),
            "days": days_out,
        }

    @staticmethod
    def _as_float(rows, idx):
        if idx >= len(rows):
            return None
        value = rows[idx]
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
