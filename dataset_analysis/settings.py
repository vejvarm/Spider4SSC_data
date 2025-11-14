import pathlib
import json


WEIGHTS = {
    'Distinct Variables': 1, # Num Tables 
    'Tokens': 0.25,          #
    'Keywords': 1,           #
    'Joins/Traversals': 3,   # 25% Join
    'Nesting Depth': 3,      # 20% nest
    'Filters': 1.5,          # 10% Variable In
    'Aggregations': 1.5,
    'Sorting/Limiting': 1,
    'Projections': 1         # 15% Variable Out
}

QUERY_ORDER = ('sparql', 'sql', 'cypher')
QUERY_LANGUAGES = ('Cypher', 'SPARQL', 'SQL', 'MQL')
QUERY_KEYWORDS = json.load(open("./dataset_analysis/query-keywords.json"))

# QUERY_LANGUAGE_COLORS = {
#     'Cypher': [0.984375, 0.7265625, 0],
#     'SPARQL': 'firebrick',
#     'SQL': 'darkolivegreen'
# }

QUERY_LANGUAGE_COLORS = {
    'sparql': '#2F7B3B',
    'sql': '#BCAE55',
    'cypher': '#51779C',
    'total': 'gray'
}

RES_FOLDER = pathlib.Path("./plot_results")
RES_FOLDER.mkdir(exist_ok=True, parents=True)
RESCALED_FOLDER = RES_FOLDER.joinpath('rescaled')
RESCALED_FOLDER.mkdir(exist_ok=True, parents=True)
PLOT_SUFFIXES = ['.png', '.svg', '.pdf']
DEFAULT_FACTOR_PLOT_LIST = ["Tokens", "Keywords", 
                            "Joins/Traversals", "Nesting Depth", 
                            "Distinct Variables", "Aggregations"]
DEFAULT_FACTOR_YMAX = [32, 9, 3.5, 1.1, 4, 0.42]
DEFAULT_COMPLEXITY_YMAX = 3.5