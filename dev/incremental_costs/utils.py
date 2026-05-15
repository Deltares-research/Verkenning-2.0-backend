from __future__ import annotations
import pandas as pd
from pathlib import Path
import copy

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union
import math
import matplotlib.pyplot as plt
import numpy as np

#switch dir and load packages
import os
# os.chdir(r"C:\Users\tao\Local_Documents\GitHub\Verkenning-2.0-backend")
from app.dike_components.dike_model import DikeModel
from app.cost_calculator import CostCalculator, DirectCostGroundWork, StructureCosts, InfrastructureCosts
from app.unit_costs_and_surcharges import load_kosten_catalogus
from app.cost_calculator import CostItem


path_cost = Path("app/datasets/eenheidsprijzen.json")
path_opslag_factor = Path("app/datasets/opslagfactoren.json")   


def get_dimensions_dict_from_df(df: pd.DataFrame) -> Dict[str, float]:
    #set Kostenpost as index
    df_modified = df.set_index('Kostenpost')

    volume_dict = {
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
    road_surface = df_modified.loc['Verwijderen weg', 'Hoeveelheid']
    
    structure_and_infra_cost = df.loc[(df['Pad'] == 'Bouwkosten (BK) > Directe Bouwkosten (DBK)') & (df['Type'] == 'Subtotaal')]['Totaal excl. BTW'].item() - df.loc[(df['Pad'] == 'Bouwkosten (BK) > Directe Bouwkosten (DBK) > Grondversterking') & (df['Type'] == 'Subtotaal')]['Totaal excl. BTW'].item()
    return {'volumes': volume_dict, 'road_surface': road_surface, 'structure_and_infra_cost': structure_and_infra_cost}

#modified cost computation
def modified_cost_computation(dike_model, dimensions, nb_houses=  0, reuse_clay_as_top=False):
    volumes = dimensions['volumes']
    structure_and_infra_cost = dimensions['structure_and_infra_cost']
    road_surface = dimensions['road_surface']

    cat = load_kosten_catalogus(eenheidsprijzen=str(path_cost), opslagfactoren=str(path_opslag_factor))

    calculator = CostCalculator(cat, dike_model.complexity)
    groundwork_cost = calculator.calc_direct_cost_ground_work(volumes=volumes, reuse_clay_as_top=reuse_clay_as_top)
    infrastructure_cost = calculator.calc_direct_cost_infrastructure(road_surface)
    structure_cost_value = structure_and_infra_cost - infrastructure_cost.totale_BDBK_infrastructuur.value_excl_BTW

    structure_costs = StructureCosts(directe_bouwkosten=CostItem(unit_cost = 1000, quantity = structure_cost_value/1000, unit = 'm2' ))

    total_construction_cost = calculator.calc_construction_costs(groundwork_cost = groundwork_cost.totale_BDBK_grondwerk,
                                                                 structure_cost = structure_costs.totale_BDBK_constructie,
                                                                 infrastructure_cost=infrastructure_cost.totale_BDBK_infrastructuur)
    engineering_cost = calculator.calc_all_engineering_costs(
        construction_cost=total_construction_cost.totale_bouwkosten)

    general_cost = calculator.calc_general_costs(construction_cost=total_construction_cost.totale_bouwkosten)

    _investering_cost = total_construction_cost.totale_bouwkosten + engineering_cost.total_engineering_costs + general_cost.total_general_costs

    risk_cost = calculator.calc_risk_cost(investering_cost=_investering_cost,
                                            construction_costs=total_construction_cost)
    real_estate_costs = calculator.calc_real_estate_costs(nb_houses=nb_houses)

    full_cost_dict = {"Bouwkosten":
                {"Directe Bouwkosten": {
                    "Directe kosten grondwerk": groundwork_cost.to_dict(),
                    "Directe kosten constructies": structure_costs.to_dict(),
                    "Directe kosten infrastructuur": infrastructure_cost.to_dict(),},
                "Indirecte Bouwkosten": total_construction_cost.to_dict()},
            "Engineeringkosten" : engineering_cost.to_dict(),  
            "Overige bijkomende kosten": general_cost.to_dict(), 
                "Risicoreservering": risk_cost.to_dict(),
                "Vastgoedkosten": real_estate_costs.to_dict(),
        }
    costs_summary = {"Directe kosten grondwerk": groundwork_cost.totale_BDBK_grondwerk.value_excl_BTW,
                    "Directe kosten constructies": structure_costs.totale_BDBK_constructie.value_excl_BTW,
                    "Directe kosten infrastructuur": infrastructure_cost.totale_BDBK_infrastructuur.value_excl_BTW,
                    "Indirecte Bouwkosten": total_construction_cost.indirecte_bouwkosten.value_excl_BTW,
                    "Engineeringkosten": engineering_cost.total_engineering_costs.value_excl_BTW,
                    "Overige bijkomende kosten": general_cost.total_general_costs.value_excl_BTW,
                    "Objectoverstijgende risicoreservering": risk_cost.value,
                    "Vastgoedkosten": real_estate_costs.total_real_estate_costs.value_excl_BTW
                    }
    #make dataframe from costs summary with keys as index and column header Kosten excl. BTW
    costs_summary = pd.DataFrame.from_dict(costs_summary, orient='index', columns=['Kosten excl. BTW'])

    return costs_summary, full_cost_dict

def compute_incremental_volumes(vol_orig_to_A, vol_orig_to_B):
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

def make_reinforcement_incremental(dimensions_first_increment, dimensions_second_increment, complexity = 'gemiddelde maatregel', additional_costs = 0, reuse_clay = True):

    #Aangevulde volume moet worden aangepast naar het verschil tussen de 2 increments.
    dV5 = dimensions_second_increment['volumes']['V5'] - dimensions_first_increment['volumes']['V5']

    #copy second increment and adjust V5
    incremental_reinforcement = copy.deepcopy(dimensions_second_increment)
    incremental_reinforcement['volumes']['V5'] = dV5
    incremental_reinforcement['volumes'] = compute_incremental_volumes(dimensions_first_increment['volumes'], dimensions_second_increment['volumes'])

    #add any additional costs
    incremental_reinforcement['structure_and_infra_cost'] += additional_costs

    #recompute the costs for the incremental reinforcement
    increment_cost_summary, increment_costs_detailed = modified_cost_computation(DikeModel(complexity = complexity), incremental_reinforcement, reuse_clay_as_top=reuse_clay)
    return increment_cost_summary, increment_costs_detailed

def compute_incremental_costs(alternative_order: list[str], dimensions: dict[str, dict], dike: DikeModel, reuse_clay = True, complexity = 'gemiddelde maatregel'):
    incremental_cost_summaries = {}
    incremental_detailed_costs = {}
    #compute cost of first alternative in the order
    incremental_cost_summaries[alternative_order[0]], incremental_detailed_costs[alternative_order[0]] = modified_cost_computation(dike, dimensions[alternative_order[0]])
    print(f"Kosten van {alternative_order[0]}:")
    print(f"€{incremental_cost_summaries[alternative_order[0]]['Kosten excl. BTW'].sum():,.0f}")
    print("\n")

    for i in range(len(alternative_order)-1):
        alt_1 = alternative_order[i]
        alt_2 = alternative_order[i+1]
        incremental_cost_summaries[f"{alt_1} to {alt_2}"], incremental_detailed_costs[f"{alt_1} to {alt_2}"] = make_reinforcement_incremental(dimensions[alt_1], dimensions[alt_2], complexity = complexity, reuse_clay=reuse_clay)
        #print summarized incremental costs per incremental step
        print(f"Incrementele kosten van {alt_1} naar {alt_2}:")
        print(f"€{incremental_cost_summaries[f'{alt_1} to {alt_2}']['Kosten excl. BTW'].sum():,.0f}")
        print("\n")


    return incremental_cost_summaries, incremental_detailed_costs