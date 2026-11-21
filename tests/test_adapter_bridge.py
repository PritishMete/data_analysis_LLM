from insight_learning.adapter_bridge import semantic_envelope


def test_semantic_envelope_aliases_safe_fields():
    result = semantic_envelope("myntra", "v1", {"operation": "rank", "group_by": "brand_name", "column": "rating", "function": "mean", "direction": "desc", "limit": 5}, {"brand_name": "FIELD_02", "rating": "FIELD_03"})
    assert result["intent"] == "rank"
    assert result["group_by"] == ["FIELD_02"]
    assert result["metrics"] == [{"column": "FIELD_03", "function": "mean"}]
    assert result["limit"] == 5
