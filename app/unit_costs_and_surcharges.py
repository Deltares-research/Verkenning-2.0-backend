
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class KostenItem:
    code: str
    omschrijving: str
    eenheid: str
    prijs: float

@dataclass
class KostenCatalogus:
    categorieen: Dict[str, List[KostenItem]]

#fill data classes from dataset/eenheidsprijzen.json
import json
def load_kosten_catalogus(eenheidsprijzen: str = "app/datasets/eenheidsprijzen.json", opslagfactoren: str = "app/datasets/opslagfactoren.json") -> KostenCatalogus:
    with open(eenheidsprijzen, "r", encoding="utf-8") as f:
        eenheidsprijsdata = json.load(f)
    
    with open(opslagfactoren, "r", encoding="utf-8") as f:
        opslagfactorendata = json.load(f)

    categorieen = {}
    for categorie, items in eenheidsprijsdata.items():
        kosten_items = []
        for item in items:
            kosten_item = KostenItem(
                code=item["code"],
                omschrijving=item["omschrijving"],
                eenheid=item["eenheid"],
                prijs=item["prijs"]
            )
            kosten_items.append(kosten_item)
        categorieen[categorie] = kosten_items

    for categorie, items in opslagfactorendata.items():
        kosten_items = []
        for item in items:
            kosten_item = KostenItem(
                code=item["code"],
                omschrijving=item["omschrijving"],
                eenheid="percentage",
                prijs=item["percentage"]
            )
            kosten_items.append(kosten_item)
        categorieen[categorie] = kosten_items
    
    return KostenCatalogus(categorieen=categorieen)