import asyncio

from location_agent import enrich_rows, infer_location_columns


def test_infer_location_columns():
    cols = ["LocationID", "City", "Country", "Latitude_Center", "Longitude_Center", "Region"]
    got = infer_location_columns(cols)
    assert got["city"] == "City"
    assert got["country"] == "Country"
    assert got["region"] == "Region"
    assert got["latitude"] == "Latitude_Center"
    assert got["longitude"] == "Longitude_Center"


def test_reverse_coordinate_fill(monkeypatch):
    async def fake_agent(columns):
        return infer_location_columns(columns)

    monkeypatch.setattr("location_agent.agent_map_columns", fake_agent)

    def fake_reverse(self, lat, lon):
        return {
            "lat": str(lat), "lon": str(lon),
            "address": {"city": "Kolkata", "state": "West Bengal", "country": "India"},
        }

    monkeypatch.setattr("location_agent.Geocoder._nominatim_reverse", fake_reverse)
    monkeypatch.setattr("location_agent.GOOGLE_MAPS_API_KEY", "")

    result = asyncio.run(enrich_rows([
        {"LocationID": 1, "City": "", "Country": "", "Latitude_Center": 22.5726, "Longitude_Center": 88.3639, "Region": ""}
    ], ["LocationID", "City", "Country", "Latitude_Center", "Longitude_Center", "Region"]))

    row = result["rows"][0]
    assert row["City"] == "Kolkata"
    assert row["Region"] == "West Bengal"
    assert row["Country"] == "India"
    assert result["filled_cells"] == 3


def test_forward_city_can_fill_coordinates(monkeypatch):
    async def fake_agent(columns):
        return infer_location_columns(columns)

    monkeypatch.setattr("location_agent.agent_map_columns", fake_agent)

    def fake_search(self, params):
        return {
            "lat": "22.5726", "lon": "88.3639",
            "address": {"city": "Kolkata", "state": "West Bengal", "country": "India"},
        }

    monkeypatch.setattr("location_agent.Geocoder._nominatim", fake_search)
    monkeypatch.setattr("location_agent.GOOGLE_MAPS_API_KEY", "")

    result = asyncio.run(enrich_rows([
        {"LocationID": 1, "City": "Kolkata", "Country": "India", "Latitude_Center": "", "Longitude_Center": "", "Region": "West Bengal"}
    ], ["LocationID", "City", "Country", "Latitude_Center", "Longitude_Center", "Region"]))

    row = result["rows"][0]
    assert row["Latitude_Center"] == 22.5726
    assert row["Longitude_Center"] == 88.3639
    assert result["filled_cells"] == 2
