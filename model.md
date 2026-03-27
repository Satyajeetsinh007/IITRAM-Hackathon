# AI Forecasting + Crop Trend Alerts (v1)

## What Was Implemented
This document records the full implementation that was completed for the **AI Forecasting + Crop Trend Alerts** feature in the Farm-to-Market project.

The build adds a production-safe ML-lite intelligence layer that:
- predicts order volume for `1d`, `3d`, and `7d`
- detects trending vegetables from offer + product activity
- sends high-confidence in-app notifications to relevant farmers
- keeps existing admin APIs and analytics features backward compatible

---

## Architecture Added
The implementation follows the existing layered pattern:
- **Routes** -> **Services** -> **Repositories** -> **MongoDB**

### New Core Components
- `app/repositories/market_intelligence_repository.py`
- `app/services/market_intelligence_service.py`
- `app/services/weather_client.py`
- `app/farmer/routes.py`

### Existing Wiring Updated
- `app/admin/dependencies.py` now wires `market_intelligence` service.
- `app/__init__.py` now registers `farmer_bp` and JSON error handling for `/farmer/api/*`.
- `app/admin/routes.py` now exposes new admin AI endpoints.

---

## Database Changes
### New Collections
1. `ml_order_forecasts`
2. `ml_crop_trends`
3. `farmer_notifications`

### New Schema Validators
Added in:
- `app/repositories/schema.py`

### New Indexes
Added in:
- `app/repositories/indexes.py`

Indexes:
- `ml_order_forecasts(forecast_for_date, horizon_days)` unique
- `ml_crop_trends(created_at, crop, horizon_days)`
- `farmer_notifications(farmer_id, is_read, created_at desc)`
- `farmer_notifications(farmer_id, type, created_at)`

---

## Forecasting + Trend Logic
Implemented in:
- `app/services/market_intelligence_service.py`

### Order Forecasting (`1d/3d/7d`)
Inputs:
- last 90 days order counts
- weekday seasonality
- short trend slope
- weather adjustment

Model approach:
- weighted blend of recent average + weekday average + trend projection
- weather multiplier bounded to safe range
- outputs predicted orders + confidence score (0-100)

### Confidence Scoring
Factors:
- data sufficiency
- volatility penalty
- trend stability
- weather availability

### Crop Trend Scoring
Uses 14-day window from offers/products:
- demand growth (offer quantity)
- price momentum (offer price)
- supply pressure (recent product listing trend)

Weights:
- demand: 50%
- price: 30%
- supply: 20%

Filtering:
- only emit trends when both score and confidence pass thresholds:
  - `AI_MIN_TREND_SCORE`
  - `AI_MIN_CONFIDENCE`

### Notification Rules
- only `uptrend` crop signals notify
- only relevant active farmers receive notification
- dedupe by crop + horizon + forecast date + same day
- priority: high for very strong trend scores

---

## Weather Integration
Implemented in:
- `app/services/weather_client.py`

Provider:
- Open-Meteo (`WEATHER_PROVIDER=open_meteo`)

Behavior:
- calls weather forecast API with timeout
- returns normalized daily weather fields
- fails safely to fallback payload if API/network fails
- no crash path (service continues with non-weather mode)

---

## API Endpoints Added

### Admin APIs
Implemented in `app/admin/routes.py`:
- `GET /admin/api/analytics/forecast?horizon=1d|3d|7d`
- `GET /admin/api/analytics/crop-trends?horizon=1d|3d|7d`
- `POST /admin/api/analytics/refresh`

### Farmer APIs
Implemented in `app/farmer/routes.py`:
- `GET /farmer/api/notifications?page=&per_page=&unread_only=`
- `PATCH /farmer/api/notifications/<id>/read`

Security:
- admin APIs use existing `admin_required`
- farmer APIs require `X-User-Id` + `X-User-Role=farmer`

Error format remains:
- `{"error":"...","status":<code>}`

---

## Admin UI Updates
Implemented in:
- `app/templates/admin/analytics.html`

Added above KPI cards:
- **Order Forecast** cards (`1d/3d/7d`) with confidence badge
- **Trending Vegetables** panel with score/confidence/signal chips
- **Run Forecast Now** button (calls refresh API)

Preserved:
- existing KPI charts
- existing 3D analytics chart
- existing analytics range handling

---

## CLI + Config
### New CLI Command
Implemented in `app/cli.py`:
- `flask --app run.py refresh-ai-insights --notify`

### New Config/Env Variables
Implemented in `app/config.py` and `.env.example`:
- `AI_REFRESH_HOURS`
- `AI_MIN_CONFIDENCE`
- `AI_MIN_TREND_SCORE`
- `AI_MODEL_VERSION`
- `WEATHER_PROVIDER`
- `WEATHER_LAT`
- `WEATHER_LON`
- `WEATHER_TIMEOUT_SEC`

### Staleness Policy
- if latest insights are older than `AI_REFRESH_HOURS`, forecast/trend APIs attempt auto-refresh
- if refresh fails, last successful data is returned with warning when available

---

## Tests Added
### Unit Tests
- `tests/unit/test_market_intelligence_service.py`

Covers:
- valid forecast generation
- safe fallback on sparse/empty data
- weather failure fallback
- trend filtering behavior
- confidence/threshold gating
- notification relevance and dedupe

### Integration Tests
- `tests/integration/test_market_intelligence_integration.py`

Covers:
- admin auth protection
- forecast endpoint shape
- invalid horizon handling
- refresh endpoint behavior
- farmer role access control
- list and mark-read notification flow

### Test Result
- full suite passed: `47 passed`

---

## Files Created
- `app/repositories/market_intelligence_repository.py`
- `app/services/market_intelligence_service.py`
- `app/services/weather_client.py`
- `app/farmer/__init__.py`
- `app/farmer/routes.py`
- `tests/unit/test_market_intelligence_service.py`
- `tests/integration/test_market_intelligence_integration.py`
- `model.md` (this file)

## Files Updated
- `.env.example`
- `app/__init__.py`
- `app/admin/dependencies.py`
- `app/admin/routes.py`
- `app/cli.py`
- `app/config.py`
- `app/repositories/indexes.py`
- `app/repositories/schema.py`
- `app/templates/admin/analytics.html`

---

## How To Use
1. Ensure `.env` includes Mongo + AI/weather config.
2. Initialize DB structure:
   - `flask --app run.py init-db`
3. Refresh insights manually:
   - `flask --app run.py refresh-ai-insights --notify`
4. Run server:
   - `flask --app run.py run`
5. Open admin analytics:
   - `http://127.0.0.1:5001/admin/analytics`

---

## Notes
- This is intentionally ML-lite for v1 reliability and easy maintenance.
- Weather is market-level (single configured lat/lon), not per-farm microclimate.
- Notifications are in-app feed only in this version.
