"""Agentic location enrichment.

The LLM is used as an orchestrator/column-mapping agent. It never invents
coordinates or administrative areas. Factual location values come from the
configured geocoder (Google if GOOGLE_MAPS_API_KEY is set, otherwise
OpenStreetMap Nominatim with a conservative rate limit).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from functools import lru_cache
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MODEL = os.getenv("LOCATION_AGENT_MODEL", "gemini-3.5-flash")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org").rstrip("/")
USER_AGENT = os.getenv("GEOCODING_USER_AGENT", "EnterpriseDataAnalytics/1.0 location-enrichment")

_FIELD_ALIASES = {
    "city": ["city", "town", "municipality", "locality"],
    "country": ["country", "country_name", "nation"],
    "region": ["region", "state", "state_province", "province", "administrative_area", "state_name"],
    "latitude": ["latitude", "lat", "latitude_center", "lat_center", "y"],
    "longitude": ["longitude", "long", "lon", "longitude_center", "lng", "lon_center", "x"],
    "location_id": ["locationid", "location_id", "location id", "id"],
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def infer_location_columns(columns: list[str]) -> dict[str, str | None]:
    normalized = {_norm(c): c for c in columns}
    result: dict[str, str | None] = {k: None for k in _FIELD_ALIASES}
    for field, aliases in _FIELD_ALIASES.items():
        for alias in aliases:
            key = _norm(alias)
            if key in normalized:
                result[field] = normalized[key]
                break
        if result[field] is None:
            for c in columns:
                nc = _norm(c)
                if any(_norm(a) in nc or nc in _norm(a) for a in aliases if len(_norm(a)) >= 3):
                    result[field] = c
                    break
    return result


def _blank(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip() == ""


def _valid_coord(v: Any, latitude: bool) -> bool:
    if _blank(v):
        return False
    try:
        x = float(v)
        return -90 <= x <= 90 if latitude else -180 <= x <= 180
    except Exception:
        return False


def _http_json(url: str) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en"})
    with urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


class Geocoder:
    def __init__(self) -> None:
        self._last_nominatim = 0.0

    def _nominatim(self, params: dict[str, Any]) -> dict[str, Any] | None:
        # Nominatim's public service is intentionally throttled. For serious
        # production volumes set GOOGLE_MAPS_API_KEY or point NOMINATIM_URL at
        # an organization-controlled Nominatim instance.
        wait = 1.05 - (time.monotonic() - self._last_nominatim)
        if wait > 0:
            time.sleep(wait)
        url = f"{NOMINATIM_URL}/search?{urlencode(params)}"
        self._last_nominatim = time.monotonic()
        data = _http_json(url)
        return data[0] if isinstance(data, list) and data else None

    def _nominatim_reverse(self, lat: float, lon: float) -> dict[str, Any] | None:
        wait = 1.05 - (time.monotonic() - self._last_nominatim)
        if wait > 0:
            time.sleep(wait)
        url = f"{NOMINATIM_URL}/reverse?{urlencode({'lat': lat, 'lon': lon, 'format': 'jsonv2', 'zoom': 10})}"
        self._last_nominatim = time.monotonic()
        return _http_json(url)

    def resolve(self, *, city: str | None, region: str | None, country: str | None,
                latitude: Any, longitude: Any) -> dict[str, Any]:
        has_coords = _valid_coord(latitude, True) and _valid_coord(longitude, False)
        if has_coords:
            lat, lon = float(latitude), float(longitude)
            if GOOGLE_MAPS_API_KEY:
                params = {"latlng": f"{lat},{lon}", "key": GOOGLE_MAPS_API_KEY, "language": "en"}
                data = _http_json(f"https://maps.googleapis.com/maps/api/geocode/json?{urlencode(params)}")
                results = data.get("results") or []
                if results:
                    return self._google_to_location(results[0], lat, lon, "reverse_geocode")
            item = self._nominatim_reverse(lat, lon)
            if item:
                return self._nominatim_to_location(item, lat, lon, "reverse_geocode")
            return {"confidence": 0.0, "method": "reverse_geocode", "reason": "No geocoder result."}

        if city:
            query = ", ".join(x for x in [city, region, country] if x)
            if GOOGLE_MAPS_API_KEY:
                params = {"address": query, "key": GOOGLE_MAPS_API_KEY, "language": "en"}
                data = _http_json(f"https://maps.googleapis.com/maps/api/geocode/json?{urlencode(params)}")
                results = data.get("results") or []
                if len(results) == 1:
                    return self._google_to_location(results[0], None, None, "forward_geocode")
                if len(results) > 1:
                    return {"confidence": 0.45, "method": "forward_geocode", "reason": "Multiple possible locations; coordinates left blank."}
            item = self._nominatim({"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1})
            if item:
                return self._nominatim_to_location(item, None, None, "forward_geocode")
        return {"confidence": 0.0, "method": "none", "reason": "Insufficient location information."}

    @staticmethod
    def _google_to_location(item: dict[str, Any], lat: float | None, lon: float | None, method: str) -> dict[str, Any]:
        address = {"city": None, "region": None, "country": None}
        for component in item.get("address_components", []):
            types = component.get("types", [])
            if "country" in types:
                address["country"] = component.get("long_name")
            elif "administrative_area_level_1" in types:
                address["region"] = component.get("long_name")
            elif any(t in types for t in ["locality", "postal_town", "administrative_area_level_2"]):
                address["city"] = component.get("long_name")
        loc = item.get("geometry", {}).get("location", {})
        return {**address, "latitude": lat if lat is not None else loc.get("lat"),
                "longitude": lon if lon is not None else loc.get("lng"),
                "confidence": 0.95, "method": method, "reason": "Geocoder matched a unique result."}

    @staticmethod
    def _nominatim_to_location(item: dict[str, Any], lat: float | None, lon: float | None, method: str) -> dict[str, Any]:
        a = item.get("address", {})
        city = a.get("city") or a.get("town") or a.get("village") or a.get("municipality")
        region = a.get("state") or a.get("state_district") or a.get("county")
        out_lat = lat if lat is not None else item.get("lat")
        out_lon = lon if lon is not None else item.get("lon")
        return {"city": city, "region": region, "country": a.get("country"),
                "latitude": float(out_lat) if out_lat is not None else None,
                "longitude": float(out_lon) if out_lon is not None else None,
                "confidence": 0.90 if method == "reverse_geocode" else 0.82,
                "method": method, "reason": "Geocoder matched a location."}


async def agent_map_columns(columns: list[str]) -> dict[str, Any]:
    """Ask the existing Gemini agent to identify location columns.

    The response is constrained to a small JSON object. Actual factual
    enrichment is performed by the geocoder, never by the language model.
    """
    instruction = (
        "You are a spreadsheet location-enrichment planning agent. Identify which "
        "provided column names represent LocationID, City, Country, Region/State, "
        "Latitude/Lat and Longitude/Lon. Return ONLY JSON with keys "
        "location_id,city,country,region,latitude,longitude. Values must be exact "
        "column names from the provided list or null. Never invent a column."
    )
    try:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        agent = LlmAgent(name="location_agent", model=MODEL, instruction=instruction)
        session_service = InMemorySessionService()
        sid = str(time.time_ns())
        await session_service.create_session(app_name="location_agent_app", user_id="api_user", session_id=sid)
        runner = Runner(agent=agent, app_name="location_agent_app", session_service=session_service)
        content = types.Content(role="user", parts=[types.Part(text=json.dumps({"columns": columns}))])
        final = None
        async for event in runner.run_async(user_id="api_user", session_id=sid, new_message=content):
            if event.is_final_response() and event.content and event.content.parts:
                final = next((p.text for p in event.content.parts if getattr(p, "text", None)), None)
        if final:
            start, end = final.find("{"), final.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(final[start:end + 1])
                valid = set(columns)
                return {k: (v if v in valid else None) for k, v in parsed.items()}
    except Exception:
        pass
    return infer_location_columns(columns)


def _apply_result(row: dict[str, Any], cols: dict[str, str | None], resolved: dict[str, Any]) -> int:
    filled = 0
    for key in ("city", "country", "region", "latitude", "longitude"):
        col = cols.get(key)
        if not col or _blank(resolved.get(key)):
            continue
        if not _blank(row.get(col)):
            continue
        row[col] = resolved[key]
        filled += 1
    return filled


async def enrich_rows(rows: list[dict[str, Any]], columns: list[str], preview_only: bool = False) -> dict[str, Any]:
    cols = infer_location_columns(columns)
    agent_cols = await agent_map_columns(columns)
    # Prefer the AI mapping when it is valid; deterministic inference remains
    # the fallback for anything the model leaves unresolved.
    for key, value in agent_cols.items():
        if value in columns:
            cols[key] = value

    required = [cols.get(k) for k in ("city", "country", "region", "latitude", "longitude")]
    if not any(required):
        return {"success": False, "error": "No location columns were detected.", "columns": cols}

    geocoder = Geocoder()
    output = [dict(r) for r in rows]
    filled_cells = 0
    resolved_rows = 0
    unresolved_rows = 0
    row_results = []
    geocode_cache: dict[tuple[str | None, str | None, str | None, str, str], dict[str, Any]] = {}

    for index, row in enumerate(output):
        city = str(row.get(cols["city"], "")).strip() if cols["city"] else None
        region = str(row.get(cols["region"], "")).strip() if cols["region"] else None
        country = str(row.get(cols["country"], "")).strip() if cols["country"] else None
        lat = row.get(cols["latitude"]) if cols["latitude"] else None
        lon = row.get(cols["longitude"]) if cols["longitude"] else None
        missing = any([
            cols["city"] and _blank(row.get(cols["city"])),
            cols["region"] and _blank(row.get(cols["region"])),
            cols["country"] and _blank(row.get(cols["country"])),
            cols["latitude"] and _blank(row.get(cols["latitude"])),
            cols["longitude"] and _blank(row.get(cols["longitude"])),
        ])
        if not missing:
            continue
        cache_key = (city or None, region or None, country or None, str(lat or ""), str(lon or ""))
        if cache_key in geocode_cache:
            resolved = geocode_cache[cache_key]
        else:
            try:
                resolved = await asyncio.to_thread(
                    geocoder.resolve, city=city or None, region=region or None, country=country or None,
                    latitude=lat, longitude=lon,
                )
            except Exception as exc:
                resolved = {"confidence": 0.0, "method": "error", "reason": str(exc)}
            geocode_cache[cache_key] = resolved
        confidence = float(resolved.get("confidence") or 0)
        filled = 0
        if confidence >= 0.80:
            filled = _apply_result(row, cols, resolved)
        if filled:
            filled_cells += filled
            resolved_rows += 1
        else:
            unresolved_rows += 1
        row_results.append({"row": index + 2, "filled": filled, "confidence": confidence,
                            "method": resolved.get("method"), "reason": resolved.get("reason")})

    return {
        "success": True,
        "columns": cols,
        "output_columns": columns,
        "rows": output,
        "preview_only": preview_only,
        "filled_cells": filled_cells,
        "resolved_rows": resolved_rows,
        "unresolved_rows": unresolved_rows,
        "total_rows": len(rows),
        "row_results": row_results[:500],
        "message": f"Location agent resolved {resolved_rows} rows and proposed {filled_cells} cell fills."
    }
