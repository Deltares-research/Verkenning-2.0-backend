from app.datasets.unit_cost_to_json import csv_to_nested_json, parse_price, parse_percentage
import json

from app.unit_costs_and_surcharges import KostenCatalogus, KostenItem, load_kosten_catalogus

import pytest

@pytest.mark.parametrize("raw,expected", [
    ("€ 414.87", 414.87),
    ("414.87", 414.87),
    ("1,270.52", 1270.52),
    ("€ 1.270,52", 1270.52),
    ("12,694.90", 12694.90),
    ("€12.694,90", 12694.90),
])
def test_parse_price(raw, expected):
    assert parse_price(raw) == expected


def test_csv_to_nested_json_parses_correctly(tmp_path):
    csv_content = """Code,Omschrijving,Eenheid,€/Eenheid
,Damwandconstructies:,,
Q-GC1A.100,Leveren en aanbrengen damwand 2m,m¹,€ 414.87
Q-GC1A.110,Leveren en aanbrengen damwand 4m,m¹,€ 788.01
,Grondverzet:,,
Q-GV010,Ontgraven teelaarde,m³,€ 3.71
Q-GV020,Ontgraven klei,m³,€ 9.45
"""

    csv_path = tmp_path / "test.csv"
    json_path = tmp_path / "out.json"

    csv_path.write_text(csv_content, encoding="utf-8")

    catalog = csv_to_nested_json(str(csv_path), str(json_path))

    # ### Structural tests ###
    assert "Damwandconstructies" in catalog
    assert "Grondverzet" in catalog

    assert len(catalog["Damwandconstructies"]) == 2
    assert len(catalog["Grondverzet"]) == 2

    # ### Content tests ###
    item = catalog["Damwandconstructies"][0]
    assert item["code"] == "Q-GC1A.100"
    assert item["omschrijving"].startswith("Leveren en aanbrengen")
    assert item["eenheid"] == "m¹"
    assert item["prijs"] == 414.87

    item2 = catalog["Grondverzet"][1]
    assert item2["prijs"] == 9.45  # Converted from € 9,45

    # ### JSON file was written ###
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert "Damwandconstructies" in written
    assert written["Grondverzet"][0]["prijs"] == 3.71




def test_all_items_have_required_fields(tmp_path):
    csv_content = """Code,Omschrijving,Eenheid,€/Eenheid
,Damwandconstructies:,,
Q-GC1A.100,Damwand 2m,m¹,€ 414.87
"""

    csv_path = tmp_path / "input.csv"
    json_path = tmp_path / "output.json"

    csv_path.write_text(csv_content, encoding="utf-8")

    catalog = csv_to_nested_json(str(csv_path), str(json_path))

    for category, items in catalog.items():
        for item in items:
            assert "code" in item
            assert "omschrijving" in item
            assert "eenheid" in item
            assert "prijs" in item
            assert isinstance(item["prijs"], float)


def test_no_empty_categories(tmp_path):
    csv_content = """Code,Omschrijving,Eenheid,€/Eenheid
,Damwandconstructies:,,
Q-GC1A.100,Damwand 2m,m¹,€ 414.87
"""

    csv_path = tmp_path / "x.csv"
    json_path = tmp_path / "y.json"

    csv_path.write_text(csv_content, encoding="utf-8")
    catalog = csv_to_nested_json(str(csv_path), str(json_path))

    for category, items in catalog.items():
        assert len(items) > 0

#tests for the dataclass values
#verify it has the right number of categories and items
def test_load_kosten_catalogus():
    catalogus = load_kosten_catalogus()
    assert isinstance(catalogus, KostenCatalogus)
    assert "Damwandconstructies (vereenvoudigd)" in catalogus.categorieen
    assert "Grondverzet" in catalogus.categorieen
    assert len(catalogus.categorieen["Damwandconstructies (vereenvoudigd)"]) > 0
    assert len(catalogus.categorieen["Grondverzet"]) > 0

    # Check that items are of type KostenItem
    for categorie, items in catalogus.categorieen.items():
        for item in items:
            assert isinstance(item, KostenItem)
            assert isinstance(item.code, str)
            assert isinstance(item.omschrijving, str)
            assert isinstance(item.eenheid, str)
            assert isinstance(item.prijs, float)

def test_number_of_items_in_categories():
    catalogus = load_kosten_catalogus()
    damwand_items = catalogus.categorieen.get("Damwandconstructies (vereenvoudigd)", [])
    grondverzet_items = catalogus.categorieen.get("Grondverzet", [])
    opslagfactoren_constructies = catalogus.categorieen.get("Percentages ter bepaling Opslagfactor investeringskosten / benoemde directe bouwkosten Grondversterkingen", [])
    assert len(damwand_items) == 9  # Expecting 20 items in Damwandconstructies
    assert len(grondverzet_items) == 8  # Expecting at least 8 items in Grondverzet
    assert len(opslagfactoren_constructies) == 6  # Expecting 6 percentage items in opslagfactoren_constructies
