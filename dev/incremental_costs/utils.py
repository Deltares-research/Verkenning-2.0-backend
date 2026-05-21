from __future__ import annotations
import pandas as pd
from pathlib import Path
import copy

from app.dike_components.dike_model import DikeModel
from app.cost_calculator import CostCalculator, StructureCosts, CostItem, SummedCostItem
from app.unit_costs_and_surcharges import KostenCatalogus, load_kosten_catalogus, get_price


path_cost = Path("app/datasets/eenheidsprijzen.json")
path_opslag_factor = Path("app/datasets/opslagfactoren.json")   


def get_dimensions_dict_from_df(df: pd.DataFrame) -> dict:
    #set Kostenpost as index
    df_modified = df.set_index('Kostenpost')

    ground = {
        'V1b': df_modified.loc['Afgraven grasbekleding', 'Hoeveelheid'],
        'V2b': df_modified.loc['Afgraven kleilaag', 'Hoeveelheid'],
        'V3': max(df_modified.loc['Aanvullen teelaarde', 'Hoeveelheid'],df_modified.loc['Hergebruik teelaarde', 'Hoeveelheid']),
        'V4': df_modified.loc['Aanbrengen nieuwe kleilaag', 'Hoeveelheid'],
        'V5': df_modified.loc['Aanvullen kern', 'Hoeveelheid'] - df_modified.loc['Afgraven grasbekleding', 'Hoeveelheid'],
        'full_AHN_surface': df_modified.loc['Opruimen terrein', 'Hoeveelheid'],
        'envelop_AHN_surface': 0, #not used
        'full_design_surface': df_modified.loc['Profieleren nieuwe graslaag', 'Hoeveelheid'],
        'envelop_design_surface': df_modified.loc['Profieleren dijkkern', 'Hoeveelheid']
    }
    
    #get (if they exist) the vaklengte and unit_costs of the structure 
    #check if Heavescherm, Damwand onverankerd or Damwand verankerd are in the df index
    if 'Heavescherm' in df_modified.index:
        structure_type = 'Heavescherm'
        structure_vaklengte = df_modified.loc['Heavescherm', 'Hoeveelheid']
        structure_unit_cost = df_modified.loc['Heavescherm', 'Eenheidsprijs']
    elif 'Onverankerde damwand' in df_modified.index:
        structure_type = 'Onverankerde damwand'
        structure_vaklengte = df_modified.loc['Onverankerde damwand', 'Hoeveelheid']
        structure_unit_cost = df_modified.loc['Onverankerde damwand', 'Eenheidsprijs']
    elif 'Verankerde damwand' in df_modified.index:
        structure_type = 'Verankerde damwand'
        structure_vaklengte = df_modified.loc['Verankerde damwand', 'Hoeveelheid']
        structure_unit_cost = df_modified.loc['Verankerde damwand', 'Eenheidsprijs']
    else:
        structure_type = 'Geen constructie'
        structure_vaklengte = 0.0
        structure_unit_cost = 0.0

    structure = {'Type': structure_type,
                 'Vaklengte': structure_vaklengte,
                 'Eenheidsprijs': structure_unit_cost,
    }

    infrastructure = {
        'Weg': df_modified.loc['Verwijderen weg', 'Hoeveelheid'],
        'Fietspad': df_modified.loc['Verwijderen fietspad', 'Hoeveelheid']
    }

    return {'ground': ground, 'structure': structure, 'infrastructure': infrastructure}


def find_structure_cost_from_catalog(structure_type: str, cost_catalog: KostenCatalogus) -> dict:
    return {'c': get_price(cost_catalog, f'c_{structure_type}'),
            'd': get_price(cost_catalog, f'd_{structure_type}'),
            'z': get_price(cost_catalog, f'z_{structure_type}')}


def modified_cost_computation(dike_model, dimensions, second_structure=None, nb_houses=0, reuse_clay_as_top=False):
    ground = dimensions['ground']
    structure = dimensions['structure']
    infrastructure = dimensions['infrastructure']

    cat = load_kosten_catalogus(eenheidsprijzen=str(path_cost), opslagfactoren=str(path_opslag_factor))
    calculator = CostCalculator(cat, dike_model.complexity)

    groundwork_cost = calculator.calc_direct_cost_ground_work(volumes=ground, reuse_clay_as_top=reuse_clay_as_top)

    infrastructure_cost = calculator.calc_direct_cost_infrastructure(road_area=infrastructure['Weg'], bike_path_area=infrastructure['Fietspad'])

    if second_structure is not None:
        cost_function_pararams = {'c': get_price(cat, f"c_{second_structure['Type']}"),
                                  'd': get_price(cat, f"d_{second_structure['Type']}"),
                                  'z': get_price(cat, f"z_{second_structure['Type']}")
                                  }
        second_structure_cost = calculator.calc_direct_cost_structure(vaklengte=second_structure['Vaklengte'],
                                                                      wandlengte=second_structure['Wandlengte'],
                                                                      structure_type=second_structure['Type'],
                                                                      cost_function_parameters=cost_function_pararams
                                                                      )
        if structure['Eenheidsprijs'] != 0:
            structure['Type'] = f"{structure['Type']} en {second_structure['Type']}"
        structure['Eenheidsprijs'] = (structure['Vaklengte'] * structure['Eenheidsprijs']) + second_structure_cost.totale_BDBK_constructie.value_excl_BTW
        structure['Vaklengte'] = 1
    structure_cost = StructureCosts(directe_bouwkosten=CostItem(quantity=structure['Vaklengte'],
                                                                unit_cost=structure['Eenheidsprijs'],
                                                                unit='m'
                                                                )
                                    )

    extra = 0 if 'extra' not in dimensions else dimensions['extra']
    extra_cost = SummedCostItem(description="Extra kosten", value_excl_BTW=extra, value_incl_BTW=extra*1.21)
    
    total_construction_cost = calculator.calc_construction_costs(groundwork_cost=groundwork_cost.totale_BDBK_grondwerk,
                                                                 structure_cost=structure_cost.totale_BDBK_constructie + extra_cost,
                                                                 infrastructure_cost=infrastructure_cost.totale_BDBK_infrastructuur)

    engineering_cost = calculator.calc_all_engineering_costs(construction_cost=total_construction_cost.totale_bouwkosten)

    general_cost = calculator.calc_general_costs(construction_cost=total_construction_cost.totale_bouwkosten)

    _investering_cost = total_construction_cost.totale_bouwkosten + engineering_cost.total_engineering_costs + general_cost.total_general_costs

    risk_cost = calculator.calc_risk_cost(investering_cost=_investering_cost, construction_costs=total_construction_cost)

    real_estate_costs = calculator.calc_real_estate_costs(nb_houses=nb_houses)

    full_cost_dict = \
        {"Bouwkosten":
            {"Directe benoemde bouwkosten": {
                "Directe kosten grondwerk": groundwork_cost.to_dict(),
                "Directe kosten constructies": structure_cost.to_dict(),
                "Directe kosten infrastructuur": infrastructure_cost.to_dict()
                },
            "Indirecte bouwkosten": total_construction_cost.to_dict()
            },
            "Engineeringkosten" : engineering_cost.to_dict(),
            "Overige bijkomende kosten": general_cost.to_dict(),
            "Risicoreservering": risk_cost.to_dict(),
            "Vastgoedkosten": real_estate_costs.to_dict(),
        }

    costs_summary = {"Directe kosten grondwerk": groundwork_cost.totale_BDBK_grondwerk.value_excl_BTW,
                    "Directe kosten constructies": structure_cost.totale_BDBK_constructie.value_excl_BTW,
                    "Directe kosten infrastructuur": infrastructure_cost.totale_BDBK_infrastructuur.value_excl_BTW,
                    "Directe niet-benoemde bouwkosten":
                         total_construction_cost.directe_niet_benoemde_bouwkosten_grondwerk.value +
                         total_construction_cost.directe_niet_benoemde_bouwkosten_constructie.value +
                         total_construction_cost.directe_niet_benoemde_bouwkosten_infrastructuur.value,
                    "Indirecte bouwkosten": total_construction_cost.indirecte_bouwkosten.value_excl_BTW,
                    "Engineeringkosten": engineering_cost.total_engineering_costs.value_excl_BTW,
                    "Overige bijkomende kosten": general_cost.total_general_costs.value_excl_BTW,
                    "Objectoverstijgende risicoreservering": risk_cost.value,
                    "Vastgoedkosten": real_estate_costs.total_real_estate_costs.value_excl_BTW
                    }
    costs_summary = pd.DataFrame.from_dict(costs_summary, orient='index', columns=['Kosten excl. BTW'])

    new_dimensions = {
        'ground': {
            'V1b': groundwork_cost.afgraven_toplaag.quantity,
            'V2b': groundwork_cost.afgraven_oud_materiaal.quantity,
            'V3': groundwork_cost.hergebruik_toplaag.quantity + groundwork_cost.aanvullen_toplaag.quantity,
            'V4': groundwork_cost.aanbrengen_nieuwe_kleilaag.quantity,
            'V5': groundwork_cost.aanvullen_kern.quantity - groundwork_cost.afgraven_toplaag.quantity,
            'full_AHN_surface': groundwork_cost.kosten_opruimen.quantity,
            'envelop_AHN_surface': 0,  # niet gebruikt
            'full_design_surface': groundwork_cost.profileren_nieuwe_toplaag.quantity,
            'envelop_design_surface': groundwork_cost.profileren_dijkkern.quantity,
            },
        'structure': {
            'type': structure['Type'],
            'quantity': structure['Vaklengte'],
            'unit_cost': structure['Eenheidsprijs']
            },
        'infrastructure': {
            'Weg': infrastructure_cost.verwijderen_weg.quantity,
            'Fietspad': infrastructure_cost.verwijderen_fietspad.quantity,
            },
        }

    return costs_summary, full_cost_dict, new_dimensions


def compute_incremental_volumes(vol_orig_to_A: dict, vol_orig_to_B: dict):
    """
    vol_orig_to_A and vol_orig_to_B are dicts with the same keys as your 'volumes' dict.
    Returns the incremental volumes for the A→B step.
    """
    A = vol_orig_to_A
    B = vol_orig_to_B

    return {
        # After measure A, the new toplaag (V3_A) becomes the "old" toplaag (V1b) for step A→B.
        # What needs to be removed in A→B is the toplaag that was placed in A,
        # only where it falls within the envelope that needs to change.
        'V1b': A['V3'],  # The toplaag placed in step A is the one removed in step A→B

        # Same logic: the kleilaag placed in A becomes the "old" kleilaag for A→B
        'V2b': A['V4'],  # The kleilaag placed in step A is the one removed in step A→B

        # New toplaag, kleilaag, kern for the B profile
        'V3': B['V3'],
        'V4': B['V4'],

        # The kern in B minus the kern already built in A.
        # V2b from A→B step (= A['V4']) is already reused, so net new fill is:
        'V5': B['V5'] - A['V5'],  # only the additional kern volume needed

        # Surfaces: use the B surfaces for profiling, but subtract A surfaces
        # for work that was already done and needs to be redone
        'full_AHN_surface':     A['full_design_surface'],
        'envelop_AHN_surface':  A['envelop_design_surface'],
        'full_design_surface':  B['full_design_surface'],
        'envelop_design_surface': B['envelop_design_surface'],
    }


def make_reinforcement_incremental(dimensions_first_increment: dict,
                                   dimensions_second_increment: dict,
                                   complexity: str='gemiddelde maatregel',
                                   nb_houses: int=0,
                                   additional_costs: float=0.0,
                                   reuse_clay: bool=True
                                   ) -> tuple:

    #copy second increment and adjust volumes
    incremental_reinforcement = copy.deepcopy(dimensions_second_increment)
    incremental_reinforcement['ground'] = compute_incremental_volumes(dimensions_first_increment['ground'], dimensions_second_increment['ground'])

    #add any additional costs
    incremental_reinforcement['extra'] = additional_costs

    #recompute the costs for the incremental reinforcement
    increment_cost_summary, increment_costs_detailed, increment_dimensions = (
        modified_cost_computation(DikeModel(complexity=complexity),
                                            dimensions=incremental_reinforcement,
                                            second_structure=None,
                                            nb_houses=nb_houses,
                                            reuse_clay_as_top=reuse_clay
                                  )
    )
    return increment_cost_summary, increment_costs_detailed, increment_dimensions


def compute_incremental_costs(alternative_order: list[str],
                              dimensions: dict[str, dict],
                              dike: DikeModel,
                              reuse_clay: bool=True,
                              complexity: str='gemiddelde maatregel',
                              extra_costs: float=0.0
                              ) -> tuple:
    incremental_cost_summaries = {}
    incremental_detailed_costs = {}
    incremental_dimensions = {}

    #compute cost of first alternative in the order
    (incremental_cost_summaries[alternative_order[0]],
     incremental_detailed_costs[alternative_order[0]],
     incremental_dimensions[alternative_order[0]]
     ) = modified_cost_computation(dike, dimensions[alternative_order[0]])

    print(f"Kosten van {alternative_order[0]}:")
    print(f"€{incremental_cost_summaries[alternative_order[0]]['Kosten excl. BTW'].sum():,.0f}")
    print("\n")

    for i in range(len(alternative_order)-1):
        alt_1 = alternative_order[i]
        alt_2 = alternative_order[i+1]
        (incremental_cost_summaries[f"{alt_1} to {alt_2}"],
         incremental_detailed_costs[f"{alt_1} to {alt_2}"],
         incremental_dimensions[f"{alt_1} to {alt_2}"]
         ) = make_reinforcement_incremental(dimensions[alt_1], dimensions[alt_2],
                                            complexity=complexity, reuse_clay=reuse_clay,
                                            additional_costs=extra_costs
                                            )

        #print summarized incremental costs per incremental step
        print(f"Incrementele kosten van {alt_1} naar {alt_2}:")
        print(f"€{incremental_cost_summaries[f'{alt_1} to {alt_2}']['Kosten excl. BTW'].sum():,.0f}")
        print("\n")

    return incremental_cost_summaries, incremental_detailed_costs, incremental_dimensions