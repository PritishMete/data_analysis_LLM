import asyncio
from io import StringIO

import pandas as pd

import categorization_agent as ca
import main as backend_main
from currency_utils import (
    currency_format,
    detect_currency_from_value,
    extract_target_currency,
    has_currency_conversion_intent,
    normalize_currency,
)
from query_router import _detect_sentiment_intent


SAMPLE = """Country\tRegion\tCity\tGender\tBool\tCurrency\tRating\tRating Count\tLatitude\tLongitude\tRestaurantID\tReviewText
India\tAsia\tNew Delhi\tF\t1\t$1250\t4.2\t728\t28.6139\t77.2090\tR-100\tgood place but need to improve ambience
india\tasia\tDelhi\tFemale\tYes\t₹900\t3.7\t510\t28.7041\t77.1025\tR-101\tworst food quality with high price
Idnia\tAsia\tmumbai\tFemalee\ty\tAed 150\t4.8\t312\t19.0760\t72.8777\tR-102\tAmazing food and excellent service
Uae\tEurope\tDubai\tM\tNOOO\tUsd 75\t2.1\t0\t25.2048\t55.2708\tR-103\t
United Arab Emirates\teu\tAbu Dhabi\tm\t0\tC$45\t4.5\t12\t24.4539\t54.3773\tR-104\tQuick delivery but noisy place
Arab\tMiddle East\tAbu Dhabi\tFemale\tNo\tRubel 2500\t3.9\t80\t24.4539\t54.3773\tR-105\tfriendly staff and tasty food
United Kingdom\tEurope\tLondon\tFemale\tY\tSgd 40\t4.0\t90\t51.5074\t-0.1278\tR-106\tservice was slow but food was great
Uk\tEurope\tLondon\tFemale\tn\tS$25\t2.8\t44\t51.5074\t-0.1278\tR-107\tclean place with good value
Singapore\tAsia\tSingapore\tT\tNo\t₹ 50\t4.4\t66\t1.3521\t103.8198\tR-108\tok
Singapor\tasia\tSingapore\tFemale\tYes\t₽3000\t4.1\t18\t1.3521\t103.8198\tR-109\t 
Bangladesh\tAsia\tDhaka\tMale\tNOOO\t৳1200\t4.7\t101\t23.8103\t90.4125\tR-110\tgreat service and food
bd\tAsia\tDhaka\tFemale\tY\t$20\t3.0\t55\t23.8103\t90.4125\tR-111\t
Russia\tEurope\tMoscow\tMale\tNo\tUsd 75\t4.3\t72\t55.7558\t37.6173\tR-112\tbest ambience
russina\teu\tmumbai\tFemale\tYes\tCad 60\t1.8\t20\t55.7558\t37.6173\tR-113\tbad
Usa\tNorth America\tNew York\tM\tYes\t₹ 100\t4.6\t17\t40.7128\t-74.0060\tR-114\tnice
Us\tNorth America\tNew York\tFemale\tNo\t£35\t4.0\t30\t40.7128\t-74.0060\tR-115\t
Canada\tNorth America\tToronto\tFemale\t0\tUSD 75\t3.6\t28\t43.6532\t-79.3832\tR-116\t
Canad\tNorth America\tnyc\tMale\tNo\tC$45\t4.9\t8\t43.6532\t-79.3832\tR-117\t
India\tAsia\tKolkata\tF\tyess\t₹1,500\t4.1\t88\t22.5726\t88.3639\tR-118\tgood
india\tAsia\tKolkata\tFemale\tNo\t₹ 20\t3.5\t9\t22.5726\t88.3639\tR-119\tbad service
"""


def _load_df() -> pd.DataFrame:
    return pd.read_csv(StringIO(SAMPLE), sep="\t")


def _assert_no_category_suffix(columns: list[str]) -> None:
    assert all(not str(col).endswith("_Category") for col in columns)


def test_currency_helpers_require_explicit_conversion_intent():
    assert has_currency_conversion_intent("Convert Currency to INR")
    assert has_currency_conversion_intent("convert all monetary values to USD")
    assert not has_currency_conversion_intent("Categorize Currency")
    assert not has_currency_conversion_intent("Categorize all columns")
    assert normalize_currency("Indian rupee") == "INR"
    assert normalize_currency("rupees") == "INR"
    assert normalize_currency("dollars") == "USD"
    assert normalize_currency("British pound") == "GBP"
    assert extract_target_currency("convert currency to rupee") == "INR"
    assert extract_target_currency("convert currency to rupees") == "INR"
    assert extract_target_currency("convert currency to Indian rupee") == "INR"
    assert extract_target_currency("convert currency to $") == "USD"
    assert extract_target_currency("convert currency to €") == "EUR"
    assert extract_target_currency("convert currency to £") == "GBP"
    assert extract_target_currency("convert currency to ¥") == "JPY"
    assert currency_format("INR") == "₹#,##0.00"
    assert currency_format("USD") == '$#,##0.00'
    assert currency_format("JPY") == '¥#,##0'
    assert detect_currency_from_value("C$45") == "CAD"
    assert detect_currency_from_value("S$25") == "SGD"


def test_sentiment_requires_explicit_intent():
    columns = ["ReviewText", "Country"]
    assert _detect_sentiment_intent("Analyze sentiment of the reviews.", columns) is not None
    assert _detect_sentiment_intent("Create sentiment categories from ReviewText.", columns) is not None
    assert _detect_sentiment_intent("Show the review text.", columns) is None


def test_gemini_mapping_is_applied_and_not_overwritten(monkeypatch):
    monkeypatch.setattr(ca, "strict_enabled", lambda: False)

    async def fake_ask(user_request, source_column, values, categories, unmatched):
        mapping = {v: f"Gemini::{i}" for i, v in enumerate(values, start=1)}
        return {
            "categories": sorted(set(mapping.values())),
            "unmatchedLabel": unmatched,
            "mapping": mapping,
            "explanation": "fake gemini mapping",
        }

    monkeypatch.setattr(ca, "_ask_agent", fake_ask)
    df = pd.DataFrame({"Signal": ["alpha", "beta", "gamma"]})

    async def run():
        return await ca.categorize_dataframe(df, "Signal", "Signal", "categorize signal")

    out, meta = asyncio.run(run())
    assert meta["execution"]["ai_used"] is True
    assert meta["execution"]["gemini_mapping_used"] is True
    assert meta["execution"]["fallback_used"] is False
    assert out["Signal"].tolist() == ["Gemini::1", "Gemini::2", "Gemini::3"]
    _assert_no_category_suffix(list(out.columns))


def test_categorize_all_columns_preserves_protected_columns_and_currency_unchanged(monkeypatch):
    monkeypatch.setattr(ca, "strict_enabled", lambda: False)

    async def fake_ask(user_request, source_column, values, categories, unmatched):
        mapping = ca._deterministic_special_mapping(values, source_column) or {v: v.title() for v in values}
        return {
            "categories": sorted(set(mapping.values())),
            "unmatchedLabel": unmatched,
            "mapping": mapping,
            "explanation": "fake gemini mapping",
        }

    monkeypatch.setattr(ca, "_ask_agent", fake_ask)
    df = _load_df()

    async def run_all(frame: pd.DataFrame):
        meta = {}
        for column in frame.columns:
            frame, meta[column] = await ca.categorize_dataframe(frame, column, column, "Categorize all columns")
        return frame, meta

    out, meta = asyncio.run(run_all(df))
    assert out["Country"].tolist()[0:4] == ["India", "India", "India", "United Arab Emirates"]
    assert out["Region"].tolist()[0:4] == ["Asia", "Asia", "Asia", "Europe"]
    assert out["Gender"].tolist()[0:4] == ["Female", "Female", "Female", "Male"]
    assert out["Bool"].tolist()[0:4] == ["Yes", "Yes", "Yes", "No"]
    assert out["Currency"].tolist()[0] == "$1250"
    assert out["Currency"].tolist()[1] == "₹900"
    assert out["Currency"].tolist()[2] == "Aed 150"
    assert out["Rating"].tolist() == df["Rating"].tolist()
    assert out["Rating Count"].tolist() == df["Rating Count"].tolist()
    assert out["Latitude"].tolist() == df["Latitude"].tolist()
    assert out["Longitude"].tolist() == df["Longitude"].tolist()
    assert out["RestaurantID"].tolist() == df["RestaurantID"].tolist()
    assert out["ReviewText"].tolist() == df["ReviewText"].tolist()
    _assert_no_category_suffix(list(out.columns))
    assert meta["Rating"]["write_mode"] == "unchanged"
    assert meta["Rating Count"]["write_mode"] == "unchanged"
    assert meta["Latitude"]["write_mode"] == "unchanged"
    assert meta["Longitude"]["write_mode"] == "unchanged"
    assert meta["RestaurantID"]["write_mode"] == "unchanged"
    assert meta["ReviewText"]["write_mode"] == "unchanged"
    assert meta["Currency"]["write_mode"] == "unchanged"
    assert meta["Currency"]["execution"]["column_role"] == "protected_currency"


def test_bool_and_gender_variants_normalize_without_low_medium_high(monkeypatch):
    monkeypatch.setattr(ca, "strict_enabled", lambda: False)

    async def fake_ask(user_request, source_column, values, categories, unmatched):
        mapping = ca._deterministic_special_mapping(values, source_column) or {v: v.title() for v in values}
        return {
            "categories": sorted(set(mapping.values())),
            "unmatchedLabel": unmatched,
            "mapping": mapping,
            "explanation": "fake gemini mapping",
        }

    monkeypatch.setattr(ca, "_ask_agent", fake_ask)
    df = pd.DataFrame({
        "Bool": ["Yes", "No", "Y", "N", "yess", "NOOO", "1", "0"],
        "Gender": ["F", "Female", "Femalee", "M", "m", "Male", "T", "female"],
    })

    async def run():
        df1, meta_bool = await ca.categorize_dataframe(df, "Bool", "Bool", "categorize bool")
        df2, meta_gender = await ca.categorize_dataframe(df1, "Gender", "Gender", "categorize gender")
        return df2, meta_bool, meta_gender

    out, meta_bool, meta_gender = asyncio.run(run())
    assert out["Bool"].tolist() == ["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]
    assert out["Gender"].tolist() == ["Female", "Female", "Female", "Male", "Male", "Male", "Transgender", "Female"]
    assert "Low" not in out["Bool"].tolist()
    assert "Medium" not in out["Bool"].tolist()
    assert "High" not in out["Bool"].tolist()
    assert meta_bool["execution"]["ai_used"] is True
    assert meta_gender["execution"]["ai_used"] is True


def test_country_typo_fallback_is_high_confidence_without_numeric_binning(monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("gemini unavailable")

    monkeypatch.setattr(ca, "_ask_agent", fail)
    df = pd.DataFrame({"Country": ["India", "india", "Idnia", "Uae", "Arab"]})

    async def run():
        return await ca.categorize_dataframe(df, "Country", "Country", "categorize Country")

    out, meta = asyncio.run(run())
    assert meta["execution"]["ai_used"] is False
    assert meta["execution"]["local_fallback_used"] is True
    assert out["Country"].tolist() == [
        "India",
        "India",
        "India",
        "United Arab Emirates",
        "United Arab Emirates",
    ]
    assert "Low" not in out["Country"].tolist()
    assert "Medium" not in out["Country"].tolist()
    assert "High" not in out["Country"].tolist()


def test_semantic_aliases_are_preserved_when_gemini_fails(monkeypatch):
    async def fail(*args, **kwargs):
        raise RuntimeError("gemini unavailable")

    monkeypatch.setattr(ca, "_ask_agent", fail)
    df = pd.DataFrame({
        "Country": ["Arab", "Uae", "Uk"],
        "Region": ["eu", "asia", "Middle East"],
        "City": ["nyc", "moscow", "mumbai"],
        "Gender": ["T", "Femalee", "M"],
    })

    async def run():
        df1, meta_country = await ca.categorize_dataframe(df, "Country", "Country", "categorize country")
        df2, meta_region = await ca.categorize_dataframe(df1, "Region", "Region", "categorize region")
        df3, meta_city = await ca.categorize_dataframe(df2, "City", "City", "categorize city")
        df4, meta_gender = await ca.categorize_dataframe(df3, "Gender", "Gender", "categorize gender")
        return df4, meta_country, meta_region, meta_city, meta_gender

    out, meta_country, meta_region, meta_city, meta_gender = asyncio.run(run())
    assert out["Country"].tolist() == ["United Arab Emirates", "United Arab Emirates", "United Kingdom"]
    assert out["Region"].tolist() == ["Europe", "Asia", "Middle East"]
    assert out["City"].tolist() == ["New York", "Moscow", "Mumbai"]
    assert out["Gender"].tolist() == ["Transgender", "Female", "Male"]
    assert meta_country["execution"]["fallback_used"] is True
    assert meta_region["execution"]["fallback_used"] is True
    assert meta_city["execution"]["fallback_used"] is True
    assert meta_gender["execution"]["fallback_used"] is True


def test_numeric_measurements_remain_unchanged_under_generic_categorization(monkeypatch):
    async def fake_ask(*args, **kwargs):
        raise AssertionError("Gemini should not be called for protected numeric columns")

    monkeypatch.setattr(ca, "_ask_agent", fake_ask)
    df = pd.DataFrame({
        "Rating": [4.2, 3.7, 2.1],
        "Rating Count": [728, 510, 0],
        "Latitude": [28.6139, 28.7041, 19.0760],
        "Longitude": [77.2090, 77.1025, 72.8777],
        "RestaurantID": ["R-100", "R-101", "R-102"],
        "ReviewText": ["good place but need to improve ambience", "worst food quality with high price", "Amazing food"],
    })

    async def run():
        out = df
        metas = {}
        for column in df.columns:
            out, metas[column] = await ca.categorize_dataframe(out, column, column, "Categorize all columns")
        return out, metas

    out, metas = asyncio.run(run())
    assert out["Rating"].tolist() == [4.2, 3.7, 2.1]
    assert out["Rating Count"].tolist() == [728, 510, 0]
    assert out["Latitude"].tolist() == [28.6139, 28.7041, 19.0760]
    assert out["Longitude"].tolist() == [77.2090, 77.1025, 72.8777]
    assert out["RestaurantID"].tolist() == ["R-100", "R-101", "R-102"]
    assert out["ReviewText"].tolist() == df["ReviewText"].tolist()
    assert metas["Rating"]["write_mode"] == "unchanged"
    assert metas["Rating Count"]["write_mode"] == "unchanged"
    assert metas["ReviewText"]["write_mode"] == "unchanged"


def test_multi_column_processing_continues_when_one_column_fails(monkeypatch):
    async def fake_categorize(frame, source_column, new_column, user_request, requested_categories=None, unmatched_label="Other"):
        out = frame.copy()
        if source_column == "City":
            raise ValueError("City failed")
        out[source_column] = out[source_column].astype(str).str.title()
        return out, {
            "source_column": source_column,
            "new_column": source_column,
            "write_mode": "replace_source",
            "mapping": {str(v): str(v).title() for v in frame[source_column].dropna().astype(str).unique()},
            "categories": [],
            "unmatched_label": unmatched_label,
            "explanation": f"{source_column} done",
            "execution": {"column_role": "categorical_normalization", "ai_used": False},
        }

    monkeypatch.setattr(backend_main, "categorize_dataframe", fake_categorize)
    payload = {
        "text": "Categorize Country, Region, City, Gender and Bool",
        "rows": [
            {"Country": "india", "Region": "asia", "City": "mumbai", "Gender": "m", "Bool": "y"},
            {"Country": "Idnia", "Region": "eu", "City": "delhi", "Gender": "f", "Bool": "no"},
        ],
        "categorize": {
            "sourceColumns": ["Country", "Region", "City", "Gender", "Bool"],
            "allColumns": False,
            "newColumnName": "Country",
            "categories": [],
            "unmatchedLabel": "Other",
        },
    }

    result = asyncio.run(backend_main.agentic_categorize(payload))
    assert result["success"] is True
    metadata = result["operation"]["metadata"]
    assert metadata["processed_columns"] == 5
    assert len(metadata["operations"]) == 5
    assert metadata["column_statuses"]
    assert "Country" in result["operation"]["message"]
    assert "Bool" in result["operation"]["message"]


def test_backend_route_accepts_minimal_samples_payload(monkeypatch):
    monkeypatch.setattr(ca, "strict_enabled", lambda: False)

    async def fake_ask(user_request, source_column, values, categories, unmatched):
        if source_column == "Gender":
            mapping = ca._deterministic_special_mapping(values, source_column) or {v: v.title() for v in values}
        else:
            mapping = ca._deterministic_special_mapping(values, source_column) or {v: v.title() for v in values}
        return {
            "categories": sorted(set(mapping.values())),
            "unmatchedLabel": unmatched,
            "mapping": mapping,
            "explanation": "fake gemini mapping",
        }

    monkeypatch.setattr(ca, "_ask_agent", fake_ask)
    payload = {
        "text": "Categorize Country and Gender",
        "categorize": {
            "sourceColumns": ["Country", "Gender"],
            "allColumns": False,
            "newColumnName": "Country",
            "categories": [],
            "unmatchedLabel": "Other",
        },
        "columnSamples": {
            "Country": ["India", "india", "Idnia", "Uae"],
            "Gender": ["F", "Female", "m", "Male"],
        },
    }

    result = asyncio.run(backend_main.agentic_categorize(payload))
    assert result["success"] is True
    operation = result["operation"]
    assert operation["column_mappings"]["Country"]["india"] == "India"
    assert operation["column_mappings"]["Gender"]["m"] == "Male"
    assert operation["data"] is None
    assert operation["metadata"]["column_mappings"]["Country"]["Idnia"] == "India"
