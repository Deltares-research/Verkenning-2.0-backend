
import csv
import json
import re
from pathlib import Path


#Note: this function should be run when updating the eenheidsprijzen.csv file and/or the opslagfactoren csv are updated.
def parse_percentage(value: str) -> float:
    """
    Clean and convert percentage strings to float.
    Supports values like:
    - "12,5%"
    - "12.5 %"
    - "0.075"
    """
    if value is None:
        return None

    # Remove percentage symbol and whitespace
    v = value.replace("%", "").strip()

    # Replace comma with dot if it's the decimal
    if "," in v:
        v = v.replace(",", ".")

    return float(v)

def parse_price(value: str) -> float:
    """
    Convert European and US formatted price strings to floats.
    Examples supported:
    € 1.270,52  → 1270.52
    € 1,270.52  → 1270.52
    € 1270,52   → 1270.52
    € 1270.52   → 1270.52
    € 12.694,90 → 12694.90
    € 12,694.90 → 12694.90
    1.270,52    → 1270.52
    1270,52     → 1270.52
    """
    if value is None:
        return None

    # Remove Euro sign and whitespace
    v = value.replace("€", "").strip()

    # Remove non-breaking spaces or weird unicode
    v = v.replace("\u00A0", "").replace(" ", "")

    # Case 1: European format with . as thousands and , as decimal
    # e.g. "1.234,56"
    if re.match(r"^\d{1,3}(\.\d{3})+,\d{1,2}$", v):
        v = v.replace(".", "").replace(",", ".")
        return float(v)

    # Case 2: European format without thousands: "1234,56"
    if "," in v and v.count(",") == 1 and "." not in v:
        v = v.replace(",", ".")
        return float(v)

    # Case 3: US format: thousands comma, decimal dot → "1,234.56"
    if re.match(r"^\d{1,3}(,\d{3})+\.\d{1,2}$", v):
        v = v.replace(",", "")
        return float(v)

    # Finally: just try normal float parsing
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"Could not parse price value: {value}")

def csv_to_nested_json(csv_path: str, json_path: str):
    """
    Convert a CSV with category headers and item rows into nested JSON.
    Expects headers: Code, Omschrijving, Eenheid, €/Eenheid
    Category rows have empty 'Code' but a value in 'Omschrijving'.
    """
    catalog = {}
    current_category = None

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=",")  # adjust if comma separated
        reader.fieldnames = [name.lstrip("\ufeff") for name in reader.fieldnames]

        for row in reader:
            code = row["Code"].strip() if row["Code"] else ""
            oms = row["Omschrijving"].strip() if row["Omschrijving"] else ""
            eenheid = row["Eenheid"].strip() if "Eenheid" in row and row["Eenheid"] else ""
            prijs_raw = row["€/Eenheid"] if "€/Eenheid" in row and row["€/Eenheid"] else ""
            percentage_raw = row["Percentage"] if "Percentage" in row and row["Percentage"] else ""

            # Detect category header
            # Condition: row has NO code but *does* have omschrijving text
            if code == "" and oms != "":
                current_category = oms.rstrip(":")
                catalog[current_category] = []
                continue

            # Skip empty lines or lines before the first category
            if not current_category or code == "":
                continue
            
            if prijs_raw != "":
                # Parse the price
                prijs = parse_price(prijs_raw)

                # Add item
                catalog[current_category].append({
                    "code": code,
                    "omschrijving": oms,
                    "eenheid": eenheid,
                    "prijs": prijs
                })
            elif percentage_raw != "":
                # Parse the percentage
                percentage = parse_percentage(percentage_raw)

                # Add item as percentage type
                catalog[current_category].append({
                    "code": code,
                    "omschrijving": oms,
                    "percentage": percentage
                })

    # Save to JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    return catalog

if __name__ == "__main__":
    csv_path = Path(__file__).parent.joinpath("eenheidsprijzen.csv")
    json_path = Path(__file__).parent.joinpath("eenheidsprijzen.json")
    csv_to_nested_json(csv_path, json_path)

    csv_path_opslag = Path(__file__).parent.joinpath("opslagfactoren.csv")
    json_path_opslag = Path(__file__).parent.joinpath("opslagfactoren.json")
    csv_to_nested_json(csv_path_opslag, json_path_opslag)