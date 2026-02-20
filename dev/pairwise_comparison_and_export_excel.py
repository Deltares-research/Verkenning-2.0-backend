import pandas as pd
from itertools import product

# Replace these with your full NEW and OLD dictionaries.
new_data =  {'Bouwkosten': {'Directe Bouwkosten': {'Directe kosten grondwerk': {'kosten_opruimen': {'value': 820.8411496489607, 'value_incl_BTW': 993.2177910752424, 'unit_cost': 0.28, 'quantity': 2931.5755344605736, 'unit': 'm2', 'description': 'Opruimen terrein en afvoeren naar stort', 'dimension': None}, 'kosten_maaien': {'value': 58.631510689211474, 'value_incl_BTW': 70.94412793394588, 'unit_cost': 0.02, 'quantity': 2931.5755344605736, 'unit': 'm2', 'description': 'Maaien terreinen en afvoeren maaisel naar stort', 'dimension': None}, 'afgraven_toplaag': {'value': 2172.904398249233, 'value_incl_BTW': 2629.214321881572, 'unit_cost': 3.71, 'quantity': 585.6885170483108, 'unit': 'm3', 'description': 'Ontgraven teelaarde en in tijdelijk depot zetten', 'dimension': None}, 'afgraven_oud_materiaal': {'value': 7192.254989353257, 'value_incl_BTW': 8702.62853711744, 'unit_cost': 3.07, 'quantity': 2342.7540681932433, 'unit': 'm3', 'description': 'Ontgraven zand en in tijdelijk depot zetten', 'dimension': None}, 'hergebruik_oud_materiaal': {'value': 6793.986797760405, 'value_incl_BTW': 8220.72402529009, 'unit_cost': 2.9, 'quantity': 2342.7540681932433, 'unit': 'm3', 'description': 'Opnemen uit depot en aanbrengen van zand (kern)', 'dimension': None}, 'aanvullen_kern': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 14.54, 'quantity': 0, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) zand (kern)', 'dimension': None}, 'profileren_dijkkern': {'value': 2153.3102, 'value_incl_BTW': 2605.505342, 'unit_cost': 0.73, 'quantity': 2949.74, 'unit': 'm2', 'description': 'Profileren dijkprofiel kern', 'dimension': None}, 'aanbrengen_nieuwe_kleilaag': {'value': 38750.569727331036, 'value_incl_BTW': 46888.189370070555, 'unit_cost': 21.06, 'quantity': 1840.0080592274949, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) klei', 'dimension': None}, 'profileren_nieuwe_kleilaag': {'value': 2359.792, 'value_incl_BTW': 2855.3483199999996, 'unit_cost': 0.8, 'quantity': 2949.74, 'unit': 'm2', 'description': 'Profileren dijkprofiel afdeklaag', 'dimension': None}, 'hergebruik_toplaag': {'value': 2301.7558719998615, 'value_incl_BTW': 2785.1246051198323, 'unit_cost': 3.93, 'quantity': 585.6885170483108, 'unit': 'm3', 'description': 'Opnemen uit depot en aanbrengen toplaag (teelaarde)', 'dimension': None}, 'aanvullen_toplaag': {'value': 2363.131208865239, 'value_incl_BTW': 2859.388762726939, 'unit_cost': 16.9, 'quantity': 139.83024904528042, 'unit': 'm3', 'description': 'Leveren en aanbrengen (verwerken) teelaarde', 'dimension': None}, 'profileren_nieuwe_toplaag': {'value': 2307.9993761849805, 'value_incl_BTW': 2792.6792451838264, 'unit_cost': 0.77, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Profileren dijkprofiel teelaarde (eindprofiel)', 'dimension': None}, 'inzaaien_nieuwe_toplaag': {'value': 1019.1166076660953, 'value_incl_BTW': 1233.1310952759752, 'unit_cost': 0.34, 'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Inzaaien dijkprofiel', 'dimension': None}, 'totale_BDBK_grondwerk': {'description': 'Benoemde directe bouwkosten grondwerk', 'value_excl_BTW': 68294.29383774828, 'value_incl_BTW': 82636.09554367544}}, 'Directe kosten constructies': {'directe_bouwkosten': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 0.0, 'quantity': 0.0, 'unit': '', 'description': '', 'dimension': None}, 'totale_BDBK_constructie': {'description': 'Benoemde directe bouwkosten constructie', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}}, 'Directe kosten infrastructuur': {'verwijderen_weg': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 15.0, 'quantity': 0, 'unit': 'm2', 'description': 'Opbreken en afvoeren regionale weg (B=4-7m) (incl. stort-/recyclingskosten)', 'dimension': None}, 'aanleggen_weg': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 54.79, 'quantity': 0, 'unit': 'm2', 'description': 'Leveren en aanbrengen regionale weg (B=4-7m) exclusief bebording verlichting bermen en sloten', 'dimension': None}, 'verwijderen_fietspad': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 10.84, 'quantity': 0, 'unit': 'm2', 'description': 'Opbreken en afvoeren fietspad (B<2m) (incl. stort-/recyclingskosten)', 'dimension': None}, 'aanleggen_fietspad': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 38.61, 'quantity': 0, 'unit': 'm2', 'description': 'Leveren en aanbrengen fietspad (B<2m) exclusief bebording en verlichting', 'dimension': None}}}, 'Indirecte Bouwkosten': {'totale_BDBK_grondwerk': {'description': 'Benoemde directe bouwkosten grondwerk', 'value_excl_BTW': 68294.29383774828, 'value_incl_BTW': 82636.09554367544}, 'totale_BDBK_constructie': {'description': 'Benoemde directe bouwkosten constructie', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}, 'totale_BDBK_infrastructuur': {'description': 'Benoemde directe bouwkosten infrastructuur', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}, 'directe_niet_benoemde_bouwkosten_grondwerk': {'code': 'Q-GGMAKNTD', 'surcharge_percentage': 1.0, 'base_cost': 68294.29383774828, 'value': 682.9429383774827, 'description': ''}, 'directe_niet_benoemde_bouwkosten_constructie': {'code': 'Q-GCMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': ''}, 'directe_niet_benoemde_bouwkosten_infrastructuur': {'code': 'Q-GCMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': ''}, 'pm_kosten': {'code': 'Q-EKABKUKMAN', 'surcharge_percentage': 20.0, 'base_cost': 68977.23677612576, 'value': 13795.447355225151, 'description': ''}, 'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 82772.68413135092, 'value': 5959.633257457267, 'description': ''}, 'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 88732.3173888082, 'value': 4525.348186829217, 'description': ''}, 'totale_directe_bouwkosten': {'description': 'Totale directe bouwkosten', 'value_excl_BTW': 68977.23677612576, 'value_incl_BTW': 83462.4564991122}, 'indirecte_bouwkosten': {'description': 'Indirecte bouwkosten', 'value_excl_BTW': 24280.428799511636, 'value_incl_BTW': 29379.318847409075}, 'totale_bouwkosten': {'description': 'Totale bouwkosten', 'value_excl_BTW': 93257.6655756374, 'value_incl_BTW': 112841.77534652127}}}, 'Engineeringkosten': {'engineering_opdrachtgever': {'code': 'Q-ENGOG1', 'surcharge_percentage': 8.0, 'base_cost': 93257.6655756374, 'value': 7460.613246050992, 'description': 'Engineeringskosten opdrachtgever (EPK) - makkelijk'}, 'engineering_opdrachtnemer': {'code': 'Q-ENGON1', 'surcharge_percentage': 5.9, 'base_cost': 93257.6655756374, 'value': 5502.2022689626065, 'description': 'Engineeringskosten opdrachtnemer (schets-, voor-, definitief ontwerp, e.d.) - makkelijk'}, 'onderzoekskosten': {'code': 'Q-OND', 'surcharge_percentage': 1.0, 'base_cost': 93257.6655756374, 'value': 932.576655756374, 'description': 'Onderzoeken (archeologie, explosieven, LNC, e.d.))'}, 'algemene_kosten': {'code': '', 'surcharge_percentage': 7.200000000000001, 'base_cost': 13895.392170769972, 'value': 1000.4682362954381, 'description': ''}, 'winst_en_risico': {'code': '', 'surcharge_percentage': 5.1, 'base_cost': 14895.86040706541, 'value': 759.6888807603358, 'description': ''}, 'direct_engineering_cost': {'description': 'Totale directe engineeringkosten', 'value_excl_BTW': 13895.392170769972, 'value_incl_BTW': 15246.695744960958}, 'indirect_engineering_cost': {'description': 'Totale indirecte engineeringkosten', 'value_excl_BTW': 1760.157117055774, 'value_incl_BTW': 1931.3294434056943}, 'total_engineering_costs': {'description': 'Totale engineeringkosten', 'value_excl_BTW': 15655.549287825746, 'value_incl_BTW': 17178.025188366653}}, 'Overige bijkomende kosten': {'vergunningen_verzekeringen': {'code': 'Q-VERG', 'surcharge_percentage': 3.0, 'base_cost': 93257.6655756374, 'value': 2797.729967269122, 'description': 'Vergunningen, heffingen en verzekeringen'}, 'kabels_leidingen': {'code': 'Q-KL', 'surcharge_percentage': 1.0, 'base_cost': 93257.6655756374, 'value': 932.576655756374, 'description': 'Kabels & leidingen'}, 'planschade_inpassingsmaatregelen': {'code': 'Q-PLAN', 'surcharge_percentage': 4.0, 'base_cost': 93257.6655756374, 'value': 3730.306623025496, 'description': 'Planschade & inpassingsmaatregelen'}, 'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2, 'base_cost': 7460.613246050992, 'value': 537.1641537156714, 'description': 'Algemene kosten (AK)'}, 'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1, 'base_cost': 7997.777399766663, 'value': 407.88664738809973, 'description': 'Winst & risico'}, 'direct_general_costs': {'description': 'Totale directe algemene kosten', 'value_excl_BTW': 7460.613246050992, 'value_incl_BTW': 9027.3420277217}, 'indirect_general_costs': {'description': 'Totale indirecte algemene kosten', 'value_excl_BTW': 945.0508011037712, 'value_incl_BTW': 1143.511469335563}, 'total_general_costs': {'description': 'Totale algemene kosten', 'value_excl_BTW': 8405.664047154763, 'value_incl_BTW': 10170.853497057262}}, 'Risicoreservering': {'code': '', 'surcharge_percentage': 10.0, 'base_cost': 117318.8789106179, 'value': 11731.88789106179, 'description': 'Objectoverstijgende risicoreservering (makkelijk)'}, 'Vastgoedkosten': {'direct_benoemd_real_estate_cost': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 700000.0, 'quantity': 0, 'unit': 'panden', 'description': '', 'dimension': None}, 'direct_niet_benoemd_real_estate_cost': {'code': 'Q-GVMAKNTD', 'surcharge_percentage': 5.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Nader te detailleren directe bouwkosten'}, 'indirect_real_estate_cost': {'code': 'Q-GVMAKIND', 'surcharge_percentage': 7.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Indirecte vastgoedkosten'}, 'real_estate_risk_cost': {'code': 'Q-GVMAKNBO', 'surcharge_percentage': 15.0, 'base_cost': 0.0, 'value': 0.0, 'description': 'GV - makkelijk: Niet benoemd objectrisico vastgoed'}, 'total_real_estate_costs': {'description': 'Totale vastgoedkosten', 'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}}}

EXPECTED_COST_DECOMPOSITION = {'Bouwkosten': {'Directe Bouwkosten': {'Directe kosten grondwerk': {
    'kosten_opruimen': {'value': 820.8411496489607, 'value_incl_BTW': 993.2177910752424, 'unit_cost': 0.28,
                        'quantity': 2931.5755344605736, 'unit': 'm2',
                        'description': 'Opruimen terrein en afvoeren naar stort', 'dimension': None},
    'kosten_maaien': {'value': 58.631510689211474, 'value_incl_BTW': 70.94412793394588, 'unit_cost': 0.02,
                      'quantity': 2931.5755344605736, 'unit': 'm2',
                      'description': 'Maaien terreinen en afvoeren maaisel naar stort', 'dimension': None},
    'afgraven_toplaag': {'value': 2175.2290465697456, 'value_incl_BTW': 2632.0271463493923, 'unit_cost': 3.71,
                         'quantity': 586.3151068921147, 'unit': 'm3',
                         'description': 'Ontgraven teelaarde en in tijdelijk depot zetten', 'dimension': None},
    'afgraven_oud_materiaal': {'value': 7199.9495126351685, 'value_incl_BTW': 8711.938910288554, 'unit_cost': 3.07,
                               'quantity': 2345.260427568459, 'unit': 'm3',
                               'description': 'Ontgraven zand en in tijdelijk depot zetten', 'dimension': None},
    'hergebruik_oud_materiaal': {'value': 6801.25523994853, 'value_incl_BTW': 8229.518840337721, 'unit_cost': 2.9,
                                 'quantity': 2345.260427568459, 'unit': 'm3',
                                 'description': 'Opnemen uit depot en aanbrengen van zand (kern)', 'dimension': None},
    'aanvullen_kern': {'value': 20019.064945666425, 'value_incl_BTW': 24223.068584256373, 'unit_cost': 14.54,
                       'quantity': 1376.8270251489976, 'unit': 'm3',
                       'description': 'Leveren en aanbrengen (verwerken) zand (kern)', 'dimension': None},
    'profileren_dijkkern': {'value': 2188.1033046948514, 'value_incl_BTW': 2647.60499868077, 'unit_cost': 0.73,
                            'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Profileren dijkprofiel kern',
                            'dimension': None},
    'aanbrengen_nieuwe_kleilaag': {'value': 40786.038036496066, 'value_incl_BTW': 49351.10602416024, 'unit_cost': 21.06,
                                   'quantity': 1936.658976091931, 'unit': 'm3',
                                   'description': 'Leveren en aanbrengen (verwerken) klei', 'dimension': None},
    'profileren_nieuwe_kleilaag': {'value': 2397.921429802577, 'value_incl_BTW': 2901.4849300611186, 'unit_cost': 0.8,
                                   'quantity': 2997.4017872532213, 'unit': 'm2',
                                   'description': 'Profileren dijkprofiel afdeklaag', 'dimension': None},
    'hergebruik_toplaag': {'value': 2304.218370086011, 'value_incl_BTW': 2788.104227804073, 'unit_cost': 3.93,
                           'quantity': 586.3151068921147, 'unit': 'm3',
                           'description': 'Opnemen uit depot en aanbrengen toplaag (teelaarde)', 'dimension': None},
    'aanvullen_toplaag': {'value': 0.0, 'value_incl_BTW': 0.0, 'unit_cost': 16.9, 'quantity': 0, 'unit': 'm3',
                          'description': 'Leveren en aanbrengen (verwerken) teelaarde', 'dimension': None},
    'profileren_nieuwe_toplaag': {'value': 2307.9993761849805, 'value_incl_BTW': 2792.6792451838264, 'unit_cost': 0.77,
                                  'quantity': 2997.4017872532213, 'unit': 'm2',
                                  'description': 'Profileren dijkprofiel teelaarde (eindprofiel)', 'dimension': None},
    'inzaaien_nieuwe_toplaag': {'value': 1019.1166076660953, 'value_incl_BTW': 1233.1310952759752, 'unit_cost': 0.34,
                                'quantity': 2997.4017872532213, 'unit': 'm2', 'description': 'Inzaaien dijkprofiel',
                                'dimension': None},
    'totale_BDBK_grondwerk': {'description': 'Benoemde directe bouwkosten grondwerk',
                              'value_excl_BTW': 88078.3685300886, 'value_incl_BTW': 106574.82592140725}},
                                                                     'Directe kosten constructies': {
                                                                         'directe_bouwkosten': {'value': 0.0,
                                                                                                'value_incl_BTW': 0.0,
                                                                                                'unit_cost': 0.0,
                                                                                                'quantity': 0.0,
                                                                                                'unit': '',
                                                                                                'description': '',
                                                                                                'dimension': None},
                                                                         'totale_BDBK_constructie': {
                                                                             'description': 'Benoemde directe bouwkosten constructie',
                                                                             'value_excl_BTW': 0.0,
                                                                             'value_incl_BTW': 0.0}},
                                                                     'Directe kosten infrastructuur': {
                                                                         'verwijderen_weg': {'value': 0.0,
                                                                                             'value_incl_BTW': 0.0,
                                                                                             'unit_cost': 15.0,
                                                                                             'quantity': 0,
                                                                                             'unit': 'm2',
                                                                                             'description': 'Opbreken en afvoeren regionale weg (B=4-7m) (incl. stort-/recyclingskosten)',
                                                                                             'dimension': None},
                                                                         'aanleggen_weg': {'value': 0.0,
                                                                                           'value_incl_BTW': 0.0,
                                                                                           'unit_cost': 54.79,
                                                                                           'quantity': 0, 'unit': 'm2',
                                                                                           'description': 'Leveren en aanbrengen regionale weg (B=4-7m) exclusief bebording verlichting bermen en sloten',
                                                                                           'dimension': None},
                                                                         'verwijderen_fietspad': {'value': 0.0,
                                                                                                  'value_incl_BTW': 0.0,
                                                                                                  'unit_cost': 10.84,
                                                                                                  'quantity': 0,
                                                                                                  'unit': 'm2',
                                                                                                  'description': 'Opbreken en afvoeren fietspad (B<2m) (incl. stort-/recyclingskosten)',
                                                                                                  'dimension': None},
                                                                         'aanleggen_fietspad': {'value': 0.0,
                                                                                                'value_incl_BTW': 0.0,
                                                                                                'unit_cost': 38.61,
                                                                                                'quantity': 0,
                                                                                                'unit': 'm2',
                                                                                                'description': 'Leveren en aanbrengen fietspad (B<2m) exclusief bebording en verlichting',
                                                                                                'dimension': None}}},
                                              'Indirecte Bouwkosten': {'totale_BDBK_grondwerk': {
                                                  'description': 'Benoemde directe bouwkosten grondwerk',
                                                  'value_excl_BTW': 88078.3685300886,
                                                  'value_incl_BTW': 106574.82592140725}, 'totale_BDBK_constructie': {
                                                  'description': 'Benoemde directe bouwkosten constructie',
                                                  'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0},
                                                                       'totale_BDBK_infrastructuur': {
                                                                           'description': 'Benoemde directe bouwkosten infrastructuur',
                                                                           'value_excl_BTW': 0.0,
                                                                           'value_incl_BTW': 0.0},
                                                                       'directe_niet_benoemde_bouwkosten_grondwerk': {
                                                                           'code': 'Q-GGMAKNTD',
                                                                           'surcharge_percentage': 1.0,
                                                                           'base_cost': 88078.3685300886,
                                                                           'value': 880.783685300886,
                                                                           'description': ''},
                                                                       'directe_niet_benoemde_bouwkosten_constructie': {
                                                                           'code': 'Q-GCMAKNTD',
                                                                           'surcharge_percentage': 5.0,
                                                                           'base_cost': 0.0, 'value': 0.0,
                                                                           'description': ''},
                                                                       'directe_niet_benoemde_bouwkosten_infrastructuur': {
                                                                           'code': 'Q-GCMAKNTD',
                                                                           'surcharge_percentage': 5.0,
                                                                           'base_cost': 0.0, 'value': 0.0,
                                                                           'description': ''},
                                                                       'pm_kosten': {'code': 'Q-EKABKUKMAN',
                                                                                     'surcharge_percentage': 20.0,
                                                                                     'base_cost': 88959.15221538949,
                                                                                     'value': 17791.8304430779,
                                                                                     'description': ''},
                                                                       'algemene_kosten': {'code': 'Q-AK',
                                                                                           'surcharge_percentage': 7.2,
                                                                                           'base_cost': 106750.98265846739,
                                                                                           'value': 7686.070751409652,
                                                                                           'description': ''},
                                                                       'risico_en_winst': {'code': 'Q-WR',
                                                                                           'surcharge_percentage': 5.1,
                                                                                           'base_cost': 114437.05340987704,
                                                                                           'value': 5836.289723903729,
                                                                                           'description': ''},
                                                                       'totale_directe_bouwkosten': {
                                                                           'description': 'Totale directe bouwkosten',
                                                                           'value_excl_BTW': 88959.15221538949,
                                                                           'value_incl_BTW': 107640.57418062132},
                                                                       'indirecte_bouwkosten': {
                                                                           'description': 'Indirecte bouwkosten',
                                                                           'value_excl_BTW': 31314.190918391283,
                                                                           'value_incl_BTW': 37890.17101125345},
                                                                       'totale_bouwkosten': {
                                                                           'description': 'Totale bouwkosten',
                                                                           'value_excl_BTW': 120273.34313378077,
                                                                           'value_incl_BTW': 145530.74519187477}}},
                               'Engineeringkosten': {
                                   'engineering_opdrachtgever': {'code': 'Q-ENGOG1', 'surcharge_percentage': 8.0,
                                                                 'base_cost': 120273.34313378077,
                                                                 'value': 9621.867450702463,
                                                                 'description': 'Engineeringskosten opdrachtgever (EPK) - makkelijk'},
                                   'engineering_opdrachtnemer': {'code': 'Q-ENGON1', 'surcharge_percentage': 5.9,
                                                                 'base_cost': 120273.34313378077,
                                                                 'value': 7096.127244893066,
                                                                 'description': 'Engineeringskosten opdrachtnemer (schets-, voor-, definitief ontwerp, e.d.) - makkelijk'},
                                   'onderzoekskosten': {'code': 'Q-OND', 'surcharge_percentage': 1.0,
                                                        'base_cost': 120273.34313378077, 'value': 1202.7334313378078,
                                                        'description': 'Onderzoeken (archeologie, explosieven, LNC, e.d.))'},
                                   'algemene_kosten': {'code': '', 'surcharge_percentage': 7.199999999999999,
                                                       'base_cost': 17920.728126933336, 'value': 1290.2924251392,
                                                       'description': ''},
                                   'winst_en_risico': {'code': '', 'surcharge_percentage': 5.099999999999999,
                                                       'base_cost': 19211.020552072536, 'value': 979.7620481556992,
                                                       'description': ''},
                                   'direct_engineering_cost': {'description': 'Totale directe engineeringkosten',
                                                               'value_excl_BTW': 17920.728126933336,
                                                               'value_incl_BTW': 19663.48886894182},
                                   'indirect_engineering_cost': {'description': 'Totale indirecte engineeringkosten',
                                                                 'value_excl_BTW': 2270.0544732948993,
                                                                 'value_incl_BTW': 2490.8134620065975},
                                   'total_engineering_costs': {'description': 'Totale engineeringkosten',
                                                               'value_excl_BTW': 20190.782600228235,
                                                               'value_incl_BTW': 22154.302330948416}},
                               'Overige bijkomende kosten': {
                                   'vergunningen_verzekeringen': {'code': 'Q-VERG', 'surcharge_percentage': 3.0,
                                                                  'base_cost': 120273.34313378077,
                                                                  'value': 3608.2002940134234,
                                                                  'description': 'Vergunningen, heffingen en verzekeringen'},
                                   'kabels_leidingen': {'code': 'Q-KL', 'surcharge_percentage': 1.0,
                                                        'base_cost': 120273.34313378077, 'value': 1202.7334313378078,
                                                        'description': 'Kabels & leidingen'},
                                   'planschade_inpassingsmaatregelen': {'code': 'Q-PLAN', 'surcharge_percentage': 4.0,
                                                                        'base_cost': 120273.34313378077,
                                                                        'value': 4810.933725351231,
                                                                        'description': 'Planschade & inpassingsmaatregelen'},
                                   'algemene_kosten': {'code': 'Q-AK', 'surcharge_percentage': 7.2,
                                                       'base_cost': 9621.867450702463, 'value': 692.7744564505773,
                                                       'description': 'Algemene kosten (AK)'},
                                   'risico_en_winst': {'code': 'Q-WR', 'surcharge_percentage': 5.1,
                                                       'base_cost': 10314.64190715304, 'value': 526.046737264805,
                                                       'description': 'Winst & risico'},
                                   'direct_general_costs': {'description': 'Totale directe algemene kosten',
                                                            'value_excl_BTW': 9621.867450702463,
                                                            'value_incl_BTW': 11642.459615349979},
                                   'indirect_general_costs': {'description': 'Totale indirecte algemene kosten',
                                                              'value_excl_BTW': 1218.8211937153824,
                                                              'value_incl_BTW': 1474.7736443956123},
                                   'total_general_costs': {'description': 'Totale algemene kosten',
                                                           'value_excl_BTW': 10840.688644417845,
                                                           'value_incl_BTW': 13117.233259745592}},
                               'Risicoreservering': {'code': '', 'surcharge_percentage': 10.0,
                                                     'base_cost': 151304.81437842685, 'value': 15130.481437842685,
                                                     'description': 'Objectoverstijgende risicoreservering (makkelijk)'},
                               'Vastgoedkosten': {
                                   'direct_benoemd_real_estate_cost': {'value': 0.0, 'value_incl_BTW': 0.0,
                                                                       'unit_cost': 700000.0, 'quantity': 0,
                                                                       'unit': 'panden', 'description': '',
                                                                       'dimension': None},
                                   'direct_niet_benoemd_real_estate_cost': {'code': 'Q-GVMAKNTD',
                                                                            'surcharge_percentage': 5.0,
                                                                            'base_cost': 0.0, 'value': 0.0,
                                                                            'description': 'GV - makkelijk: Nader te detailleren directe bouwkosten'},
                                   'indirect_real_estate_cost': {'code': 'Q-GVMAKIND', 'surcharge_percentage': 7.0,
                                                                 'base_cost': 0.0, 'value': 0.0,
                                                                 'description': 'GV - makkelijk: Indirecte vastgoedkosten'},
                                   'real_estate_risk_cost': {'code': 'Q-GVMAKNBO', 'surcharge_percentage': 15.0,
                                                             'base_cost': 0.0, 'value': 0.0,
                                                             'description': 'GV - makkelijk: Niet benoemd objectrisico vastgoed'},
                                   'total_real_estate_costs': {'description': 'Totale vastgoedkosten',
                                                               'value_excl_BTW': 0.0, 'value_incl_BTW': 0.0}}}

old_data = EXPECTED_COST_DECOMPOSITION


def extract_items_recursive(data, path_prefix=''):
    items = []
    for key, value in data.items():
        current_path = f"{path_prefix}/{key}" if path_prefix else key
        if isinstance(value, dict):
            if 'value' in value:
                items.append({
                    'Path': current_path,
                    'Item': key,
                    'Description': value.get('description', ''),
                    'Unit': value.get('unit', ''),
                    'Quantity': value.get('quantity', 0),
                    'Unit Cost': value.get('unit_cost', 0),
                    'Value excl BTW': value.get('value', 0),
                    'Value incl BTW': value.get('value_incl_BTW', 0)
                })
            else:
                items.extend(extract_items_recursive(value, current_path))
    return items


def safe_pct_diff(new_value, old_value):
    if old_value == 0:
        return None
    return (new_value - old_value) / old_value


# --- Extract all leaf cost items recursively ---
new_items = extract_items_recursive(new_data)
old_items = extract_items_recursive(old_data)

# --- Matched comparison (same path) ---
matched_rows = []
old_by_path = {item['Path']: item for item in old_items}
for new_item in new_items:
    old_item = old_by_path.get(new_item['Path'])
    if old_item is None:
        continue

    matched_rows.append({
        'Path': new_item['Path'],
        'Item': new_item['Item'],
        'Description': new_item['Description'],
        'Unit': new_item['Unit'],
        'Quantity (Old)': old_item['Quantity'],
        'Quantity (New)': new_item['Quantity'],
        'Unit Cost (Old)': old_item['Unit Cost'],
        'Unit Cost (New)': new_item['Unit Cost'],
        'Value excl BTW (Old)': old_item['Value excl BTW'],
        'Value excl BTW (New)': new_item['Value excl BTW'],
        'Difference excl BTW': new_item['Value excl BTW'] - old_item['Value excl BTW'],
        'Difference excl BTW (%)': safe_pct_diff(new_item['Value excl BTW'], old_item['Value excl BTW']),
        'Value incl BTW (Old)': old_item['Value incl BTW'],
        'Value incl BTW (New)': new_item['Value incl BTW'],
        'Difference incl BTW': new_item['Value incl BTW'] - old_item['Value incl BTW'],
        'Difference incl BTW (%)': safe_pct_diff(new_item['Value incl BTW'], old_item['Value incl BTW'])
    })

# --- All pairwise comparison (all NEW x all OLD) ---
pairwise_rows = []
for old_item, new_item in product(old_items, new_items):
    pairwise_rows.append({
        'Old Path': old_item['Path'],
        'Old Item': old_item['Item'],
        'Old Value excl BTW': old_item['Value excl BTW'],
        'Old Value incl BTW': old_item['Value incl BTW'],
        'New Path': new_item['Path'],
        'New Item': new_item['Item'],
        'New Value excl BTW': new_item['Value excl BTW'],
        'New Value incl BTW': new_item['Value incl BTW'],
        'Difference excl BTW': new_item['Value excl BTW'] - old_item['Value excl BTW'],
        'Difference excl BTW (%)': safe_pct_diff(new_item['Value excl BTW'], old_item['Value excl BTW']),
        'Difference incl BTW': new_item['Value incl BTW'] - old_item['Value incl BTW'],
        'Difference incl BTW (%)': safe_pct_diff(new_item['Value incl BTW'], old_item['Value incl BTW'])
    })

# --- Convert to DataFrames ---
matched_df = pd.DataFrame(matched_rows)
pairwise_df = pd.DataFrame(pairwise_rows)

# --- Export to Excel (2 sheets) ---
output_file = 'pairwise_comparison_bouwkosten.xlsx'
with pd.ExcelWriter(output_file) as writer:
    matched_df.to_excel(writer, sheet_name='Matched_by_Path', index=False)
    pairwise_df.to_excel(writer, sheet_name='All_Pairwise', index=False)

print(f"Excel file created: {output_file}")
print(f"Matched rows: {len(matched_df)}")
print(f"All pairwise rows: {len(pairwise_df)}")
