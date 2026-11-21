from insight_learning.domain_router import DomainRouter


def test_routes_myntra_by_schema_without_keyword():
    decision = DomainRouter("/tmp/adapters").route(
        "show the top 5 brands by rating",
        ["product_name", "brand_name", "rating", "rating_count", "marked_price"],
    )
    assert decision.domain == "myntra"
    assert decision.confidence >= 0.96
    assert decision.fallback is True
    assert decision.reason == "adapter_unavailable"


def test_routes_finance_and_uses_adapter_when_present(tmp_path):
    (tmp_path / "finance").mkdir()
    (tmp_path / "finance" / "adapter_config.json").write_text("{}")
    decision = DomainRouter(tmp_path).route(
        "show monthly expenses",
        ["Date", "Category", "Sub category", "Transaction Type", "Amount"],
    )
    assert decision.domain == "finance"
    assert decision.fallback is False


def test_unknown_request_falls_back(tmp_path):
    decision = DomainRouter(tmp_path).route("summarize this", ["foo", "bar"])
    assert decision.domain == "myntra" or decision.domain == "unknown"
    assert decision.fallback is True


def test_routes_restaurant_schema(tmp_path):
    decision = DomainRouter(tmp_path).route(
        "show the top 5 rated restaurants",
        ["Restaurant Name", "City", "Rating", "Cuisine"],
    )
    assert decision.domain == "restaurant"
    assert decision.fallback is True
