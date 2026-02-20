import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from app.datasets.unit_cost_to_json import (
    parse_percentage,
    parse_price,
    csv_to_nested_json,
)


def test_parse_percentage():
    assert parse_percentage("12,5%") == 12.5
    assert parse_percentage("12.5 %") == 12.5
    assert parse_percentage("0.075") == 0.075
    assert parse_percentage(None) is None

def test_parse_price_european_format():
    assert parse_price("€ 1.270,52") == 1270.52
    assert parse_price("1.270,52") == 1270.52
    assert parse_price("12.694,90") == 12694.90


def test_parse_price_us_format():
    assert parse_price("1,234.56") == 1234.56


def test_parse_price_simple_float():
    assert parse_price("1270.52") == 1270.52


def test_csv_to_nested_json(tmp_path):
    # Create a temporary CSV file
    csv_content = textwrap.dedent(
        """\
        Code,Omschrijving,Eenheid,€/Eenheid,Percentage
        ,Hoofdgroep:
        101,Beton,m3,"€ 1.270,52",,
        102,Wapening,kg,1.25,, 
        201,Opslag,,,"12,5%",
        """
    )

    csv_file = tmp_path / "input.csv"
    json_file = tmp_path / "output.json"
    csv_file.write_text(csv_content, encoding="utf-8")

    # Run the converter
    catalog = csv_to_nested_json(csv_file, json_file)

    # Check returned structure
    assert "Hoofdgroep" in catalog
    assert len(catalog["Hoofdgroep"]) == 3

    # Item 1: price
    assert catalog["Hoofdgroep"][0] == {
        "code": "101",
        "omschrijving": "Beton",
        "eenheid": "m3",
        "prijs": 1270.52,
    }

    # Item 2: float price without formatting
    assert catalog["Hoofdgroep"][1]["prijs"] == 1.25

    # Item 3: percentage
    assert catalog["Hoofdgroep"][2]["percentage"] == 12.5

    # Check JSON file written
    assert json_file.exists()
    loaded = json.loads(json_file.read_text(encoding="utf-8"))
    assert loaded == catalog


def test_run_as_main(tmp_path):
    script = Path(__file__).parent.parent.parent.joinpath("app", "datasets", "unit_cost_to_json.py").resolve()

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True
    )

    # Script should run successfully
    assert result.returncode == 0