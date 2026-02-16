"""
Test 3D surface area calculation locally without API
"""
import sys
import time

import numpy as np
import shapely

from app.dike_components.dike_model import DikeModel

sys.path.insert(0, '..')

import json
import geopandas as gpd
from shapely.geometry import shape

# Test GeoJSON input
geojson_input = {
    "type": "FeatureCollection",
    "crs": {
        "type": "name",
        "properties": {
            "name": "EPSG:4326"
        }
    },
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            5.5880257283351,
                            51.89317402584419,
                            8.5
                        ],
                        [
                            5.5894349916198305,
                            51.893210674409914,
                            8.5
                        ],
                        [
                            5.590070111617253,
                            51.893406527779895,
                            8.5
                        ],
                        [
                            5.590225087768538,
                            51.893546975044146,
                            8.5
                        ],
                        [
                            5.59016749740108,
                            51.893571301269766,
                            10.3
                        ],
                        [
                            5.59002275777441,
                            51.89344013074613,
                            10.3
                        ],
                        [
                            5.589417200422898,
                            51.893253393625294,
                            10.3
                        ],
                        [
                            5.5880227993282805,
                            51.893217131470266,
                            10.3
                        ],
                        [
                            5.5880257283351,
                            51.89317402584419,
                            8.5
                        ]
                    ]
                ]
            },
            "properties": {
                "name": "-27.2m_-22.4m"
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            5.5880227993282805,
                            51.893217131470266,
                            10.3
                        ],
                        [
                            5.589417200422898,
                            51.893253393625294,
                            10.3
                        ],
                        [
                            5.59002275777441,
                            51.89344013074613,
                            10.3
                        ],
                        [
                            5.59016749740108,
                            51.893571301269766,
                            10.3
                        ],
                        [
                            5.589998325336215,
                            51.893642759394,
                            11
                        ],
                        [
                            5.589883655451839,
                            51.89353883934842,
                            11
                        ],
                        [
                            5.589364938586728,
                            51.893378881303704,
                            11
                        ],
                        [
                            5.588014195338595,
                            51.893343754245045,
                            11
                        ],
                        [
                            5.5880227993282805,
                            51.893217131470266,
                            10.3
                        ]
                    ]
                ]
            },
            "properties": {
                "name": "-22.4m_-8.3m"
            }
        }
    ]
}

print("Converting GeoJSON to GeoDataFrame...")
features = []
for feature in geojson_input['features']:
    geom = shape(feature['geometry'])
    features.append({'geometry': geom, **feature['properties']})

gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
# print(f"Created GeoDataFrame with {len(gdf)} features")
#
# print("\nInitializing DikeModel...")
# dike_model = DikeModel(gdf)
#
#
# print("\nCalculating volumes and direct cost...")
# costs = dike_model.compute_cost(10, 10)
# print(costs)

dike_model = DikeModel(_3d_ground_polygon=gdf, complexity='makkelijke maatregel')
costs = dike_model.compute_cost(nb_houses=0, road_area=0)
print(costs)

a = {'Directe kosten grondwerk': {
    'kosten_opruimen': {'value': 853.1034128614323, 'unit_cost': 0.28, 'quantity': 3046.7979030765437, 'unit': 'm2',
                        'description': 'Opruimen terrein en afvoeren naar stort', 'dimension': None},
    'kosten_maaien': {'value': 60.935958061530876, 'unit_cost': 0.02, 'quantity': 3046.7979030765437, 'unit': 'm2',
                      'description': 'Maaien terreinen en afvoeren maaisel naar stort', 'dimension': None},
    'afgraven_toplaag': {'value': 968.8817331783408, 'unit_cost': 3.71, 'quantity': 261.15410597798945, 'unit': 'm3',
                         'description': 'Ontgraven teelaarde en in tijdelijk depot zetten', 'dimension': None},
    'afgraven_oud_materiaal': {'value': 3206.97242140971, 'unit_cost': 3.07, 'quantity': 1044.6164239119578,
                               'unit': 'm3', 'description': 'Ontgraven zand en in tijdelijk depot zetten',
                               'dimension': None},
    'hergebruik_oud_materiaal': {'value': 3029.3876293446774, 'unit_cost': 2.9, 'quantity': 1044.6164239119578,
                                 'unit': 'm3', 'description': 'Opnemen uit depot en aanbrengen van zand (kern)',
                                 'dimension': None},
    'aanvullen_kern': {'value': 15291.223992375042, 'unit_cost': 14.54, 'quantity': 1051.6660242348723, 'unit': 'm3',
                       'description': 'Leveren en aanbrengen (verwerken) zand (kern)', 'dimension': None},
    'profileren_dijkkern': {'value': 2188.1033046948514, 'unit_cost': 0.73, 'quantity': 2997.4017872532213,
                            'unit': 'm2', 'description': 'Profileren dijkprofiel kern', 'dimension': None},
    'aanbrengen_nieuwe_kleilaag': {'value': 40786.03803649607, 'unit_cost': 21.06, 'quantity': 1936.6589760919314,
                                   'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) klei',
                                   'dimension': None},
    'profileren_nieuwe_kleilaag': {'value': 2397.921429802577, 'unit_cost': 0.8, 'quantity': 2997.4017872532213,
                                   'unit': 'm2', 'description': 'Profileren dijkprofiel afdeklaag', 'dimension': None},
    'hergebruik_toplaag': {'value': 1026.3356364934987, 'unit_cost': 3.93, 'quantity': 261.15410597798945, 'unit': 'm3',
                           'description': 'Opnemen uit depot en aanbrengen toplaag (teelaarde)', 'dimension': None},
    'aanvullen_toplaag': {'value': 5392.096125332834, 'unit_cost': 16.9, 'quantity': 319.05894232738666, 'unit': 'm3',
                          'description': 'Leveren en aanbrengen (verwerken) teelaarde', 'dimension': None},
    'profileren_nieuwe_toplaag': {'value': 2307.9993761849805, 'unit_cost': 0.77, 'quantity': 2997.4017872532213,
                                  'unit': 'm2', 'description': 'Profileren dijkprofiel teelaarde (eindprofiel)',
                                  'dimension': None},
    'inzaaien_nieuwe_toplaag': {'value': 1019.1166076660953, 'unit_cost': 0.34, 'quantity': 2997.4017872532213,
                                'unit': 'm2', 'description': 'Inzaaien dijkprofiel', 'dimension': None},
    'totale_BDBK_grondwerk': 78528.11566390163},

    'Directe kosten constructies': {
        'directe_bouwkosten': {'value': 0.0, 'unit_cost': 0.0, 'quantity': 0.0, 'unit': '', 'description': '',
                               'dimension': None}, 'totale_BDBK_constructie': 0.0},

    'Directe kosten infrastructuur': {
        'verwijderen_weg': {'value': 0.0, 'unit_cost': 15.0, 'quantity': 0, 'unit': 'm2',
                            'description': 'Opbreken en afvoeren regionale weg (B=4-7m) (incl. stort-/recyclingskosten)',
                            'dimension': None},
        'aanleggen_weg': {'value': 0.0, 'unit_cost': 54.79, 'quantity': 0, 'unit': 'm2',
                          'description': 'Leveren en aanbrengen regionale weg (B=4-7m) exclusief bebording verlichting bermen en sloten',
                          'dimension': None},
        'verwijderen_fietspad': {'value': 0.0, 'unit_cost': 10.84, 'quantity': 0, 'unit': 'm2',
                                 'description': 'Opbreken en afvoeren fietspad (B<2m) (incl. stort-/recyclingskosten)',
                                 'dimension': None},
        'aanleggen_fietspad': {'value': 0.0, 'unit_cost': 38.61, 'quantity': 0, 'unit': 'm2',
                               'description': 'Leveren en aanbrengen fietspad (B<2m) exclusief bebording en verlichting',
                               'dimension': None}},


    'Indirecte bouwkosten': {'totale_BDBK_grondwerk': 78528.11566390163, 'totale_BDBK_constructie': 0.0,
                             'totale_BDBK_infrastructuur': 0.0,
                             'directe_niet_benoemde_bouwkosten_grondwerk': {'code': 'Q-GGMAKNTD',
                                                                            'surcharge_percentage': 1.0,
                                                                            'base_cost': 78528.11566390163,
                                                                            'value': 785.2811566390162,
                                                                            'description': ''},
                             'directe_niet_benoemde_bouwkosten_constructie': {'code': 'Q-GCMAKNTD',
                                                                              'surcharge_percentage': 5.0,
                                                                              'base_cost': 0.0, 'value': 0.0,
                                                                              'description': ''},
                             'directe_niet_benoemde_bouwkosten_infrastructuur': {'code': 'Q-GCMAKNTD',
                                                                                 'surcharge_percentage': 5.0,
                                                                                 'base_cost': 0.0, 'value': 0.0,
                                                                                 'description': ''},

                             'pm_kosten': {'code': 'Q-EKABKUKMAN', 'surcharge_percentage': 20.0,
                                           'base_cost': 79313.39682054064, 'value': 15862.67936410813,
                                           'description': ''},
                             'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2,
                                                 'base_cost': 95176.07618464877, 'value': 6852.677485294712,
                                                 'description': ''},
                             'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1,
                                                 'base_cost': 102028.75366994349, 'value': 5203.466437167117,
                                                 'description': ''}, 'totale_directe_bouwkosten': 79313.39682054064,
                             'indirecte_bouwkosten': 27918.823286569957, 'totale_bouwkosten': 107232.2201071106},

    'Engineeringkosten': {
        'engineering_opdrachtgever': {'code': 'Q-ENGOG1', 'surcharge_percentage': 8.0, 'base_cost': 107232.2201071106,
                                      'value': 8578.577608568849,
                                      'description': 'Engineeringskosten opdrachtgever (EPK) - makkelijk'},
        'engineering_opdrachtnemer': {'code': 'Q-ENGON1', 'surcharge_percentage': 5.9, 'base_cost': 107232.2201071106,
                                      'value': 6326.700986319525,
                                      'description': 'Engineeringskosten opdrachtnemer (schets-, voor-, definitief ontwerp, e.d.) - makkelijk'},
        'onderzoekskosten': {'code': 'Q-OND', 'surcharge_percentage': 1.0, 'base_cost': 107232.2201071106,
                             'value': 1072.322201071106,
                             'description': 'Onderzoeken (archeologie, explosieven, LNC, e.d.))'},
        'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 15977.600795959479,
                            'value': 1150.3872573090825, 'description': 'Algemene kosten (AK)'},
        'winst_en_risico': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 17127.98805326856,
                            'value': 873.5273907166966, 'description': 'Winst & risico'},
        'direct_engineering_cost': 15977.600795959479, 'indirect_engineering_cost': 2023.9146480257791,
        'total_engineering_costs': 18001.51544398526}, 'Overige bijkomende kosten': {
        'vergunningen_verzekeringen': {'code': 'Q-VERG', 'surcharge_percentage': 3.0, 'base_cost': 107232.2201071106,
                                       'value': 3216.966603213318,
                                       'description': 'Vergunningen, heffingen en verzekeringen'},
        'kabels_leidingen': {'code': 'Q-KL', 'surcharge_percentage': 1.0, 'base_cost': 107232.2201071106,
                             'value': 1072.322201071106, 'description': 'Kabels & leidingen'},
        'planschade_inpassingsmaatregelen': {'code': 'Q-PLAN', 'surcharge_percentage': 4.0,
                                             'base_cost': 107232.2201071106,
                                             'value': 4289.288804284424,
                                             'description': 'Planschade & inpassingsmaatregelen'},
        'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 8578.577608568849,
                            'value': 617.6575878169571, 'description': 'Algemene kosten (AK)'},
        'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 9196.235196385805,
                            'value': 469.00799501567604, 'description': 'Winst & risico'},
        'direct_general_costs': 8578.577608568849, 'indirect_general_costs': 1086.6655828326332,
        'total_general_costs': 9665.243191401481},
    'Risicoreservering': {'code': '', 'surcharge_percentage': 10.0, 'base_cost': 134898.97874249733,
                          'value': 13489.897874249733,
                          'description': 'Objectoverstijgende risicoreservering (makkelijk)'},
    'Vastgoedkosten': {
        'direct_benoemd_real_estate_cost': {'value': 0.0, 'unit_cost': 700000.0, 'quantity': 0, 'unit': 'panden',
                                            'description': '', 'dimension': None},
        'direct_niet_benoemd_real_estate_cost': {'code': 'Q-GVMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0,
                                                 'value': 0.0,
                                                 'description': 'GV - makkelijk: Nader te detailleren directe bouwkosten'},
        'indirect_real_estate_cost': {'code': 'Q-GVMAKIND', 'surcharge_percentage': 7.0, 'base_cost': 0.0, 'value': 0.0,
                                      'description': 'GV - makkelijk: Indirecte vastgoedkosten'},
        'real_estate_risk_cost': {'code': 'Q-GVMAKNBO', 'surcharge_percentage': 15.0, 'base_cost': 0.0, 'value': 0.0,
                                  'description': 'GV - makkelijk: Niet benoemd objectrisico vastgoed'},
        'total_real_estate_costs': 0.0}}

EXPECTED_COST_DECOMPOSITION = {
    "Directe kosten grondwerk": {
        "preparation_cost": {
            "value": 914.0393709229633,
            "unit_cost": 0.30000000000000004,
            "quantity": 3046.7979030765437,
            "unit": "m2",
        },
        "afgraven_grasbekleding_cost": {
            "value": 968.8817331783408,
            "unit_cost": 3.71,
            "quantity": 261.15410597798945,
            "unit": "m3",
        },
        "afgraven_kleilaag_cost": {
            "value": 3206.97242140971,
            "unit_cost": 3.07,
            "quantity": 1044.6164239119578,
            "unit": "m3",
        },
        "herkeuren_kleilaag_cost": {
            "value": 3029.3876293446774,
            "unit_cost": 2.9,
            "quantity": 1044.6164239119578,
            "unit": "m3",
        },
        "aanvullen_kern_cost": {
            "value": 15291.223992375042,
            "unit_cost": 14.54,
            "quantity": 1051.6660242348723,
            "unit": "m3",
        },
        "profieleren_dijkkern_cost": {
            "value": 2188.1033046948514,
            "unit_cost": 0.73,
            "quantity": 2997.4017872532213,
            "unit": "m2",
        },
        "aanbregen_nieuwe_kleilaag_cost": {
            "value": 40786.03803649607,
            "unit_cost": 21.06,
            "quantity": 1936.6589760919314,
            "unit": "m3",
        },
        "profieleren_vannieuwe_kleilaag_cost": {
            "value": 2397.921429802577,
            "unit_cost": 0.8,
            "quantity": 2997.4017872532213,
            "unit": "m3",
        },
        "hergebruik_teelaarde_cost": {
            "value": 1026.3356364934987,
            "unit_cost": 3.93,
            "quantity": 261.15410597798945,
            "unit": "m3",
        },
        "aanvullen_teelaarde_cost": {
            "value": 5392.096125332834,
            "unit_cost": 16.9,
            "quantity": 319.05894232738666,
            "unit": "m3",
        },
        "profieleren_nieuwe_graslaag_cost": {
            "value": 1288.882768518885,
            "unit_cost": 0.43,
            "quantity": 2997.4017872532213,
            "unit": "m2",
        },
        "totale_BDBK_grondwerk": 76489.88244856945,
    },
    "Directe kosten constructies": {
        "totale_BDBK_constructie": 0.0,
    },
    "Engineeringkosten": {
        "epk_cost": 8355.916696941656,
    },
    "Vastgoedkosten": {
        "house_cost": 0,
        "road_cost": 0.0,
        "total_real_estate_costs": 0.0,
    },
}

# --- Full comparison ---
for category, expected_items in EXPECTED_COST_DECOMPOSITION.items():
    assert category in costs

    for key, expected in expected_items.items():
        assert key in costs[category]

        actual = costs[category][key]

        if isinstance(expected, dict):
            np.testing.assert_allclose(actual["value"], expected["value"], rtol=1e-6)
            np.testing.assert_allclose(actual["unit_cost"], expected["unit_cost"], rtol=1e-6)
            np.testing.assert_allclose(actual["quantity"], expected["quantity"], rtol=1e-6)
            assert actual["unit"] == expected["unit"]
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-6)
