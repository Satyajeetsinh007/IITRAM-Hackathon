from datetime import datetime
from urllib.parse import urlparse

from flask import (
    Blueprint,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from app.admin.decorators import admin_required
from app.admin.dependencies import get_services
from app.admin.forms import LoginForm
from app.services.exceptions import ServiceError
from app.utils.auth import current_admin_id, login_admin, logout_admin


admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.app_context_processor
def inject_template_context():
    return {
        "current_admin": getattr(g, "current_admin", None),
        "current_year": datetime.now().year,
    }


def _is_safe_redirect(target):
    if not target:
        return False
    test_url = urlparse(target)
    return test_url.scheme == "" and test_url.netloc == ""


def _handle_service_error(error):
    return jsonify({"error": error.message, "status": error.status_code}), error.status_code


@admin_bp.route("/")
def admin_home():
    if current_admin_id():
        return redirect(url_for("admin.dashboard"))
    return redirect(url_for("admin.login"))


@admin_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_admin_id():
        return redirect(url_for("admin.dashboard"))

    form = LoginForm()
    if form.validate_on_submit():
        services = get_services()
        admin, error = services["auth"].authenticate(
            form.email.data.strip().lower(), form.password.data
        )
        if error:
            flash(error, "error")
            return render_template("admin/login.html", form=form)

        login_admin(admin)
        flash("Welcome back. Admin access granted.", "success")
        next_target = request.args.get("next")
        if _is_safe_redirect(next_target):
            return redirect(next_target)
        return redirect(url_for("admin.dashboard"))

    return render_template("admin/login.html", form=form)


@admin_bp.post("/logout")
@admin_required
def logout():
    admin_id = current_admin_id()
    services = get_services()
    services["activity"].log(
        admin_id,
        action="admin_logout",
        entity_type="admin_user",
        entity_id=admin_id,
    )
    logout_admin()
    flash("You have been logged out.", "success")
    return redirect(url_for("admin.login"))


@admin_bp.get("/dashboard")
@admin_required
def dashboard():
    return render_template("admin/dashboard.html", active_page="dashboard")


@admin_bp.get("/users")
@admin_required
def users_page():
    return render_template("admin/users.html", active_page="users")


@admin_bp.get("/products")
@admin_required
def products_page():
    return render_template("admin/products.html", active_page="products")


@admin_bp.get("/orders")
@admin_required
def orders_page():
    return render_template("admin/orders.html", active_page="orders")


@admin_bp.get("/transactions")
@admin_required
def transactions_page():
    return render_template("admin/transactions.html", active_page="transactions")


@admin_bp.get("/analytics")
@admin_required
def analytics_page():
    return render_template("admin/analytics.html", active_page="analytics")


@admin_bp.get("/audit-logs")
@admin_required
def audit_logs_page():
    return render_template("admin/audit_logs.html", active_page="audit_logs")


@admin_bp.get("/api/dashboard/kpis")
@admin_required
def dashboard_kpis_api():
    services = get_services()
    services["analytics"].detect_anomalies()
    return jsonify(
        {
            "kpis": services["dashboard"].get_kpis(),
            "recent_activity": services["dashboard"].recent_activity(limit=10),
        }
    )


@admin_bp.get("/api/users")
@admin_required
def list_users_api():
    services = get_services()
    return jsonify(services["users"].list_users(request.args))


@admin_bp.patch("/api/users/<user_id>/status")
@admin_required
def update_user_status_api(user_id):
    try:
        payload = request.get_json(silent=True) or {}
        updated = get_services()["users"].update_status(
            user_id,
            status=(payload.get("status") or "").strip(),
            admin_id=current_admin_id(),
        )
        return jsonify({"item": updated})
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.post("/api/users/bulk-action")
@admin_required
def users_bulk_action_api():
    try:
        payload = request.get_json(silent=True) or {}
        result = get_services()["users"].bulk_action(
            ids=payload.get("ids"),
            action=(payload.get("action") or "").strip(),
            admin_id=current_admin_id(),
        )
        return jsonify(result)
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.get("/api/products")
@admin_required
def list_products_api():
    services = get_services()
    return jsonify(services["products"].list_products(request.args))


@admin_bp.get("/api/products/quality-summary")
@admin_required
def products_quality_summary_api():
    services = get_services()
    return jsonify(services["products"].quality_summary())


@admin_bp.patch("/api/products/<product_id>/review")
@admin_required
def review_product_api(product_id):
    try:
        payload = request.get_json(silent=True) or {}
        updated = get_services()["products"].review_product(
            product_id,
            status=(payload.get("status") or "").strip(),
            review_note=(payload.get("review_note") or "").strip(),
            admin_id=current_admin_id(),
        )
        return jsonify({"item": updated})
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.post("/api/products/bulk-action")
@admin_required
def products_bulk_action_api():
    try:
        payload = request.get_json(silent=True) or {}
        result = get_services()["products"].bulk_action(
            ids=payload.get("ids"),
            action=(payload.get("action") or "").strip(),
            admin_id=current_admin_id(),
        )
        return jsonify(result)
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.get("/api/orders")
@admin_required
def list_orders_api():
    services = get_services()
    return jsonify(services["orders"].list_orders(request.args))


@admin_bp.patch("/api/orders/<order_id>/status")
@admin_required
def update_order_status_api(order_id):
    try:
        payload = request.get_json(silent=True) or {}
        updated = get_services()["orders"].update_status(
            order_id,
            status=(payload.get("status") or "").strip(),
            admin_id=current_admin_id(),
        )
        return jsonify({"item": updated})
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.get("/api/transactions")
@admin_required
def list_transactions_api():
    services = get_services()
    return jsonify(services["transactions"].list_transactions(request.args))


@admin_bp.patch("/api/transactions/<transaction_id>/state")
@admin_required
def update_transaction_state_api(transaction_id):
    try:
        payload = request.get_json(silent=True) or {}
        updated = get_services()["transactions"].update_state(
            transaction_id,
            payment_status=(payload.get("payment_status") or "").strip(),
            review_state=(payload.get("review_state") or "").strip(),
            admin_id=current_admin_id(),
        )
        return jsonify({"item": updated})
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.get("/api/analytics/sales")
@admin_required
def sales_analytics_api():
    try:
        range_key = (request.args.get("range") or "30d").strip().lower()
        data = get_services()["analytics"].sales_trend(range_key)
        return jsonify(data)
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.get("/api/analytics/orders")
@admin_required
def orders_analytics_api():
    try:
        range_key = (request.args.get("range") or "30d").strip().lower()
        data = get_services()["analytics"].orders_trend(range_key)
        return jsonify(data)
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.get("/api/analytics/overview")
@admin_required
def analytics_overview_api():
    try:
        range_key = (request.args.get("range") or "30d").strip().lower()
        data = get_services()["analytics"].overview(range_key)
        return jsonify(data)
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.get("/api/analytics/3d-data")
@admin_required
def analytics_3d_data_api():
    try:
        range_key = (request.args.get("range") or "30d").strip().lower()
        data = get_services()["analytics"].get_3d_data(range_key=range_key, limit=100)
        return jsonify(data)
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.get("/api/analytics/forecast")
@admin_required
def analytics_forecast_api():
    try:
        horizon = (request.args.get("horizon") or "1d").strip().lower()
        data = get_services()["market_intelligence"].get_forecast(horizon)
        return jsonify(data)
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.get("/api/analytics/crop-trends")
@admin_required
def analytics_crop_trends_api():
    try:
        horizon = (request.args.get("horizon") or "1d").strip().lower()
        data = get_services()["market_intelligence"].get_crop_trends(horizon)
        return jsonify(data)
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.post("/api/analytics/refresh")
@admin_required
def analytics_refresh_api():
    try:
        notify = str((request.get_json(silent=True) or {}).get("notify", "true")).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        services = get_services()
        data = services["market_intelligence"].refresh_insights(notify=notify)
        services["activity"].log(
            current_admin_id(),
            action="analytics_refresh",
            entity_type="market_intelligence",
            entity_id=None,
            details={
                "notify": notify,
                "notifications_sent": data.get("notifications_sent", 0),
            },
        )
        return jsonify(data)
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.get("/api/audit-logs")
@admin_required
def audit_logs_api():
    try:
        data = get_services()["activity"].list_logs(request.args)
        return jsonify(data)
    except ServiceError as error:
        return _handle_service_error(error)


@admin_bp.get("/api/alerts")
@admin_required
def list_alerts_api():
    services = get_services()
    items = services["analytics"].list_unresolved_alerts(limit=50)
    return jsonify({"items": items, "count": len(items)})


@admin_bp.patch("/api/alerts/<alert_id>/resolve")
@admin_required
def resolve_alert_api(alert_id):
    try:
        services = get_services()
        updated = services["analytics"].resolve_alert(alert_id, admin_id=current_admin_id())
        services["activity"].log(
            current_admin_id(),
            action="alert_resolved",
            entity_type="admin_alert",
            entity_id=alert_id,
            details={"severity": updated.get("severity"), "type": updated.get("type")},
        )
        return jsonify({"item": updated})
    except ServiceError as error:
        return _handle_service_error(error)
