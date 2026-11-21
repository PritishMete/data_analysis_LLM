from insight_learning.adapter_bridge import deterministic_restaurant_plan


def test_top_rated_restaurants():
    assert deterministic_restaurant_plan("Show me top 5 rated restaurants") == {
        "operation": "rank", "column": "rating", "direction": "desc", "limit": 5,
    }


def test_top_rated_restaurants_in_city():
    assert deterministic_restaurant_plan("Show the top 5 rated restaurants in Kolkata") == {
        "operation": "rank", "column": "rating", "direction": "desc", "limit": 5,
        "where": {"column": "city", "operator": "eq", "value": "Kolkata"},
    }


def test_rating_filter_and_aggregate():
    assert deterministic_restaurant_plan("restaurants rated above 4.5") == {
        "operation": "filter", "column": "rating", "operator": "gt", "value": 4.5,
    }
    assert deterministic_restaurant_plan("Show the average rating by city") == {
        "operation": "aggregate", "group_by": "city", "column": "rating", "function": "mean",
    }
