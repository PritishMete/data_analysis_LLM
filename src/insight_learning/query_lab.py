"""Execute the selected domain adapter against the repository dataframe."""
import json
import re
from functools import lru_cache
from pathlib import Path
from threading import Lock

import pandas as pd
from .adapter_bridge import TrainedAdapter

FIELDS = {
    "restaurant": "id,name,city,rating,rating_count,cuisine,cost,link",
    "myntra": "product_name,brand_name,rating,rating_count,marked_price,discounted_price,sizes,product_link",
    "finance": "Date,Mode,Category,Sub category,Transaction Type,Amount",
    "retail": "Date,Product Category,Total Amount,Quantity,Gender,Price per Unit",
}
_lock = Lock()

@lru_cache(maxsize=1)
def adapter(domain):
    root = Path(__file__).resolve().parents[2] / "models" / "adapters" / domain
    version = json.loads((root / "adapter.manifest.json").read_text())["version"]
    return TrainedAdapter(domain, root, version)

def filtered(df, condition):
    series = df[condition["column"]]
    value = condition["value"]
    op = condition["operator"]
    if op in {"gt", "lt", "gte", "lte"}:
        series = pd.to_numeric(series, errors="coerce")
        value = float(value)
    elif op == "eq":
        return df[series.astype(str).str.casefold() == str(value).casefold()]
    else:
        raise ValueError("Unsupported filter")
    return df[getattr(series, {"gt":"gt","lt":"lt","gte":"ge","lte":"le"}[op])(value)]

def query(domain, question, df):
    with _lock:
        model = adapter(domain)
        plan = model.plan(question, FIELDS[domain].split(","))
    if domain == "restaurant":
        cities = [city for city in df["city"].dropna().unique()
                  if re.search(r"\b" + re.escape(str(city)) + r"\b", question, re.I)]
        if len(cities) > 1:
            raise ValueError("Please select one city")
        if cities:
            df = filtered(df, {"column": "city", "operator": "eq", "value": cities[0]})
    op = plan["operation"]
    if "where" in plan:
        df = filtered(df, plan["where"])
    col = plan["column"]
    group = plan.get("group_by")
    func = plan.get("function")
    if op == "filter":
        df = filtered(df, plan)
    else:
        if func != "count":
            df = df.assign(**{col: pd.to_numeric(df[col], errors="coerce")}).dropna(subset=[col])
        if op == "trend":
            if plan["frequency"] != "month":
                raise ValueError("Unsupported trend interval")
            df = df.assign(**{group: pd.to_datetime(df[group], errors="raise").dt.to_period("M").astype(str)})
        if func:
            if func not in {"mean","sum","count","max","min"} or not group:
                raise ValueError("Unsupported aggregation")
            df = df.groupby(group)[col].agg(func).reset_index()
        elif op != "rank":
            raise ValueError("Missing aggregation")
        if op == "rank":
            direction, limit = plan["direction"], plan["limit"]
            if direction not in {"asc","desc"} or type(limit) is not int or limit < 1:
                raise ValueError("Invalid ranking")
            df = df.sort_values(col, ascending=direction=="asc", kind="stable")
            if group and not func:
                df = df.drop_duplicates(group)
            df = df.head(limit)
    rows = json.loads(df.to_json(orient="records"))
    return {"result": rows, "display": rows, "operation": op, "model_version": model.version}
