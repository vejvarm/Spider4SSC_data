import re
import json

import numpy as np

QUERY_KEYWORDS = json.load(open("dataset_analysis/query-keywords.json"))
WEIGHTS = {
    'Distinct Variables': 1, # Num Tables 
    'Tokens': 0.5,           #
    'Keywords': 1,           #
    'Joins/Traversals': 2,   # 25% Join
    'Nesting Depth': 2,      # 20% nest
    'Filters': 1.5,          # 10% Variable In
    'Aggregations': 1.5,
    'Sorting/Limiting': 1,
    'Projections': 1         # 15% Variable Out
}


# Color helpers
def get_colors():
    return np.array([
        [0.1, 0.1, 0.1],          # Schwarz
        [0.4, 0.4, 0.4],          # Sehr dunkles Grau
        [0.7, 0.7, 0.7],          # Dunkles Grau
        [0.9, 0.9, 0.9],          # Helles Grau
        [0.984375, 0.7265625, 0], # Dunkelgelb
        [1, 1, 0.9]               # Hellgelb
    ])

def get_colors_bright():
    return np.array([
        [0x22/255, 0x88/255, 0x33/255],  # '228833' (SPARQL)
        [0xCC/255, 0xBB/255, 0x44/255],  # 'CCBB44' (SQL)
        [0x44/255, 0x77/255, 0xAA/255],  # '4477AA' (Cypher)
    ])

def color_bars(ax, colors):
    dark_color = colors[2]
    for p in ax.patches:
        p.set_edgecolor(dark_color)




# Count analysis helpers
def analyze_sql_query(q: str):
    """Returns (num_output_columns, num_distinct_alias_vars) for a single SQL query."""
    # 1) Count output columns (handles optional DISTINCT, AS aliases, ignores commas in funcs)
    m = re.search(r'(?si)\bselect\b\s*(?:distinct\b\s*)?(.*?)\bfrom\b', q)
    if not m:
        num_cols = 0
    else:
        body = m.group(1)
        # grab “table.col or col” + optional AS alias, skipping stray DISTINCT/AS tokens
        pairs = re.findall(
            r'''(?xi)
            \b(?!DISTINCT\b|AS\b)
            ([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)
            (?:\s+AS\s+([A-Za-z_]\w*))?
            ''',
            body
        )
        # alias if present else bare name after any dot
        cols = { alias or col.rsplit('.',1)[-1] for col,alias in pairs }
        num_cols = len(cols)

    # 2) Count qualified variables anywhere in the SQL
    pattern = r'\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b'
    num_vars = len(set(re.findall(pattern, q)))

    return num_cols + num_vars

def count_distinct_vars(row):
    query = row['query']
    query_language = row['query_language']
    
    if query_language == 'cypher':
        # distinct_vars = len(set(re.findall(r'\((\w):\[A-Z\]', query)))
        node_vars = set(re.findall(r'\(([a-zA-Z_][a-zA-Z0-9_]*)', query))    # Node variables
        rel_vars = set(re.findall(r'\[([a-zA-Z_][a-zA-Z0-9_]*)', query))    # Relationship variables
        distinct_vars = len(node_vars | rel_vars)
        distinct_vars = len(set(re.findall(r'\(([a-zA-Z][a-zA-Z0-9_]*)', query)))
    elif query_language == 'sparql':
        distinct_vars = len(set(re.findall(r'\?\w+', query)))
    elif query_language == 'sql':
        # match any IDENT.IDENT anywhere in the SQL
        # pattern = r'\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b'
        # distinct_vars = len(set(re.findall(pattern, query)))
        # distinct_vars = len(set(re.findall(r'\b[A-Za-z\_]+\.\b[A-Za-z\_]+\b', query)))
        # distinct_vars = len(set(re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b', query)))
        # This will return tuples (table, column), so use something like:
        # distinct_vars = len(set(['.'.join(match) for match in re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b', query)]))
        distinct_vars = analyze_sql_query(query)
    elif query_language == 'mql':
        distinct_vars = len(set(re.findall(r'\$\w+', query)))
    else:
        raise
    
    return distinct_vars


    
def count_keywords(row):
    sparql_keywords = rf'\b({"|".join(QUERY_KEYWORDS["SPARQL"])})'
    sqlite_keywords = rf'\b({"|".join(QUERY_KEYWORDS["SQL"])})'
    cypher_keywords = rf'\b({"|".join(QUERY_KEYWORDS["Cypher"])})'
    # sqlite_keywords = r'\b(ABORT|ACTION|ADD|AFTER|ALL|ALTER|ALWAYS|ANALYZE|AND|AS|ASC|ATTACH|AUTOINCREMENT|BEFORE|BEGIN|BETWEEN|BY|CASCADE|CASE|CAST|CHECK|COLLATE|COLUMN|COMMIT|CONFLICT|CONSTRAINT|CREATE|CROSS|CURRENT|CURRENT_DATE|CURRENT_TIME|CURRENT_TIMESTAMP|DATABASE|DEFAULT|DEFERRABLE|DEFERRED|DELETE|DESC|DETACH|DISTINCT|DO|DROP|EACH|ELSE|END|ESCAPE|EXCEPT|EXCLUDE|EXCLUSIVE|EXISTS|EXPLAIN|FAIL|FILTER|FIRST|FOLLOWING|FOR|FOREIGN|FROM|FULL|GENERATED|GLOB|GROUP|GROUPS|HAVING|IF|IGNORE|IMMEDIATE|IN|INDEX|INDEXED|INITIALLY|INNER|INSERT|INSTEAD|INTERSECT|INTO|IS|ISNULL|JOIN|KEY|LAST|LEFT|LIKE|LIMIT|MATCH|MATERIALIZED|NATURAL|NO|NOT|NOTHING|NOTNULL|NULL|NULLS|OF|OFFSET|ON|OR|ORDER|OTHERS|OUTER|OVER|PARTITION|PLAN|PRAGMA|PRECEDING|PRIMARY|QUERY|RAISE|RANGE|RECURSIVE|REFERENCES|REGEXP|REINDEX|RELEASE|RENAME|REPLACE|RESTRICT|RETURNING|RIGHT|ROLLBACK|ROW|ROWS|SAVEPOINT|SELECT|SET|TABLE|TEMP|TEMPORARY|THEN|TIES|TO|TRANSACTION|TRIGGER|UNBOUNDED|UNION|UNIQUE|UPDATE|USING|VACUUM|VALUES|VIEW|VIRTUAL|WHEN|WHERE|WINDOW|WITH|WITHOUT)\b' # https://sqlite.org/lang_keywords.html
    # cypher_keywords = r'\b(MATCH|RETURN|WHERE|AND|OR|CREATE|DELETE|SET|WITH|LIMIT|COUNT|DISTINCT|OPTIONAL|SKIP|ORDER BY|UNWIND|MERGE|CASE|WHEN|THEN|ELSE|END|AS|IS NULL|IS NOT NULL|STARTS WITH|ENDS WITH|CONTAINS)\b'
    # sparql_keywords = r'\b(SELECT|WHERE|FILTER|LIMIT|OFFSET|ORDER BY|GROUP BY|UNION|OPTIONAL|GRAPH|COUNT|PREFIX|DISTINCT|ASK|CONSTRUCT|DESCRIBE|FROM|NAMED|BASE|PREFIX|REDUCED|BIND|VALUES|SERVICE|MINUS|EXISTS|NOT EXISTS|IF|COALESCE|SAMPLE|GROUP_CONCAT|HAVING)\b'
    # sql_keywords = r'\b(SELECT|FROM|WHERE|JOIN|INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL OUTER JOIN|CROSS JOIN|ON|GROUP BY|HAVING|ORDER BY|LIMIT|OFFSET|INSERT|UPDATE|DELETE|COUNT|DISTINCT|AS|UNION|INTERSECT|EXCEPT|IN|NOT IN|LIKE|BETWEEN|IS NULL|IS NOT NULL|EXISTS|ANY|ALL|CASE|WHEN|THEN|ELSE|END|ASC|DESC|AND|OR|OVER|PARTITION BY)\b'
    mql_keywords = r'\b(find|findOne|insert|insertOne|insertMany|update|updateOne|updateMany|replace|replaceOne|delete|deleteOne|deleteMany|count|distinct|sort|limit|skip|aggregate|group|match|project|unwind|lookup|out|indexStats|geoNear|geoSearch|text|where|expr|jsonSchema|mod|regex|options|size|all|elemMatch|slice|bitsAllClear|bitsAllSet|bitsAnyClear|bitsAnySet|comment|meta|explain|hint|maxTimeMS|max|min|returnKey|showDiskLoc|snapshot|natural|caseSensitive|diacriticSensitive)\b'

    query = row['query']
    query_language = row['query_language']
    
    if query_language == 'cypher':
        keywords = len(re.findall(cypher_keywords, query, re.IGNORECASE))
    elif query_language == 'sparql':
        keywords = len(re.findall(sparql_keywords, query, re.IGNORECASE))
    elif query_language == 'sql':
        keywords = len(re.findall(sqlite_keywords, query, re.IGNORECASE))
    elif query_language == 'mql':
        keywords = len(re.findall(mql_keywords, query, re.IGNORECASE))
    else:
        raise
    
    return keywords


import re

def parse_sparql_triples(query):
    # Remove PREFIX and SELECT stuff
    body = re.sub(r'PREFIX [^\{]*\{', '{', query, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r'SELECT[^{]*\{', '{', body, flags=re.IGNORECASE | re.DOTALL)
    where_content = re.search(r'\{(.*)\}', body, flags=re.DOTALL)
    if not where_content:
        return []
    triples_section = where_content.group(1)
    # Split lines by period
    triple_lines = []
    for line in triples_section.split('.'):
        line = line.strip()
        if line:
            triple_lines.append(line)
    # Expand semicolons and commas
    expanded_triples = []
    for line in triple_lines:
        chunks = [c.strip() for c in line.split(';') if c.strip()]
        if not chunks: continue
        subject_predicate = chunks[0].split(None, 2)
        if len(subject_predicate) == 3:
            subj, pred, obj = subject_predicate
            expanded_triples.append((subj, pred, obj))
            for extra in chunks[1:]:
                if extra:
                    predobj = extra.split(None, 1)
                    if len(predobj) == 2:
                        pred, obj = predobj
                        expanded_triples.append((subj, pred, obj))
    return expanded_triples

def count_sparql_logical_joins(query):
    triples = parse_sparql_triples(query)
    if not triples: return 0
    var_to_triples = {}
    for idx, (s, p, o) in enumerate(triples):
        for v in (s, o):  # only count subject and object variables (or both if you prefer)
            if v.startswith('?'):
                var_to_triples.setdefault(v, set()).add(idx)
    return sum(len(triples_idxs)-1 for triples_idxs in var_to_triples.values() if len(triples_idxs) > 1)


def count_joins_traversals(row):
    query = row['query']
    query_language = row['query_language'].lower()

    if query_language == 'sql':
        join_regex = re.compile(r'\b(JOIN|INNER JOIN|LEFT JOIN|RIGHT JOIN|FULL OUTER JOIN|CROSS JOIN)\b', re.IGNORECASE)
        explicit_joins = len(join_regex.findall(query))
        
        # Find comma-separated FROM *clause* and count
        from_match = re.search(r'FROM\s+(.*?)(?:\s+WHERE|\s+GROUP BY|\s+ORDER BY|$)', query, re.IGNORECASE | re.DOTALL)
        implicit_joins = 0
        if from_match:
            from_section = from_match.group(1).strip()
            # Remove potential JOIN patterns to avoid splitting inside them
            from_section = re.sub(join_regex, '', from_section)
            # Split by ',' not inside parentheses (for future-proofing)
            tables = [t.strip() for t in re.split(r',(?![^(]*\))', from_section) if t.strip()]
            if explicit_joins == 0 and len(tables) > 1:
                implicit_joins = len(tables) - 1
        return explicit_joins + implicit_joins

    elif query_language == 'sparql':
        return count_sparql_logical_joins(query)

    elif query_language == 'cypher':
        traversal_regex = re.compile(r'-\[[^\]]*\]->|<-\[[^\]]*\]-|-\[[^\]]*\]-')
        traversals = len(traversal_regex.findall(query))
        return traversals
    elif query_language == 'mql':
        mongodb_joins = r'\$lookup'
        joins_traversals = len(re.findall(mongodb_joins, query, re.IGNORECASE))
    else:
        raise ValueError("Unsupported query language: " + query_language)
    return joins_traversals


import re

def count_nested_levels2(
    text, count_open_token_prefix, count_open_token, uncount_open_token, close_token
):
    stack = []
    max_depth = 0

    # Construct regex patterns
    count_open_pattern = re.compile(rf"\{count_open_token_prefix}\s*{count_open_token}")
    uncount_open_pattern = re.compile(rf"\{uncount_open_token}")
    close_pattern = re.compile(rf"{re.escape(close_token)}")

    # Tokenize the text to find all relevant tokens
    tokens = re.findall(
        rf"\{count_open_token_prefix}\s*{count_open_token}|\{uncount_open_token}|{re.escape(close_token)}",
        text,
    )

    for token in tokens:
        if count_open_pattern.match(token):
            stack.append(token)
            max_depth = max(max_depth, len(stack))
        elif uncount_open_pattern.match(token):
            stack.append(token)
        elif close_pattern.match(token):
            if stack and count_open_pattern.match(stack[-1]):
                stack.pop()
            elif stack and uncount_open_pattern.match(stack[-1]):
                stack.pop()

    return max_depth


def count_nesting_level2(row):
    query = row['query']
    query_language = row['query_language']
    query = strip_strings_and_comments(query)  # Helper, see below

    tokens = []
    if query_language == 'cypher':
        # Handles both 'CALL {' and '{ <whitespace> MATCH'
        open_patterns = [
            re.compile(r'CALL\s*{', re.IGNORECASE),
            re.compile(r'{\s*MATCH', re.IGNORECASE)
        ]
        close_pattern = re.compile(r'}')
        # Build a flat token list for all open/close occurrences
        pos = 0
        while pos < len(query):
            found = False
            for pattern in open_patterns:
                m = pattern.match(query, pos)
                if m:
                    tokens.append('OPEN')
                    pos = m.end()
                    found = True
                    break
            if not found:
                m = close_pattern.match(query, pos)
                if m:
                    tokens.append('CLOSE')
                    pos = m.end()
                else:
                    pos += 1
    elif query_language == 'sparql':
        # '{ <whitespace> SELECT' (arbitrary whitespace)
        open_pattern = re.compile(r'{\s*SELECT', re.IGNORECASE)
        close_pattern = re.compile(r'}')
        pos = 0
        while pos < len(query):
            if open_pattern.match(query, pos):
                tokens.append('OPEN')
                pos = open_pattern.match(query, pos).end()
            elif close_pattern.match(query, pos):
                tokens.append('CLOSE')
                pos = close_pattern.match(query, pos).end()
            else:
                pos += 1
    elif query_language == 'sql':
        # '(<whitespace>SELECT', typical for subqueries
        open_pattern = re.compile(r'\(\s*SELECT', re.IGNORECASE)
        close_pattern = re.compile(r'\)')
        pos = 0
        while pos < len(query):
            if open_pattern.match(query, pos):
                tokens.append('OPEN')
                pos = open_pattern.match(query, pos).end()
            elif close_pattern.match(query, pos):
                tokens.append('CLOSE')
                pos = close_pattern.match(query, pos).end()
            else:
                pos += 1
    else:
        raise ValueError(f"Unsupported language: {query_language}")

    # Now count nesting
    stack = []
    max_depth = 1 # DEFAULT DEPTH == 1
    for token in tokens:
        if token == 'OPEN':
            stack.append('subquery')
            max_depth = max(max_depth, len(stack))
        elif token == 'CLOSE':
            if stack:
                stack.pop()
    return max_depth

# Helper function: Remove comments and strings for cleaner matching (optional but recommended)
def strip_strings_and_comments(query):
    # Remove single/multi-line comments and strings (not bulletproof for all dialects!)
    # Remove /* ... */ (multiline comments, SQL)
    query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
    # Remove -- ... (single-line comments, SQL)
    query = re.sub(r'--.*', '', query)
    # Remove # ... (SPARQL comments)
    query = re.sub(r'#.*', '', query)
    # Remove single/double quoted strings
    query = re.sub(r"(?:'(?:[^'\\]|\\.)*')|(?:\"(?:[^\"\\]|\\.)*\")", '', query)
    return query


def count_filters(row):
    query = row['query']
    query_language = row['query_language']
    
    if query_language == 'cypher':
        filters = len(re.findall(r'\bWHERE\b', query, re.IGNORECASE))
    elif query_language == 'sparql':
        filters = len(re.findall(r'\bFILTER\b', query, re.IGNORECASE))
    elif query_language == 'sql':
        filters = len(re.findall(r'\bWHERE\b|\bCASE\b|\bIF\b', query, re.IGNORECASE))
    elif query_language == 'mql':
        # filters = len(re.findall(r'\b\$match\b', query, re.IGNORECASE))
        filters = len(re.findall(r'\b\$match\b|\b\$cond\b', query, re.IGNORECASE))
    else:
        raise ValueError(f"Unsupported query language: {query_language}")
    
    return filters



def count_aggregations(row):
    query = row['query']
    query_language = row['query_language']
    
    if query_language == 'cypher':
        aggregations = len(re.findall(r'\bDISTINCT\b', query, re.IGNORECASE))
    elif query_language == 'sparql':
        aggregations = len(re.findall(r'\bGROUP BY\b|\bAGGREGATE\b|\bDISTINCT\b', query, re.IGNORECASE))
    elif query_language == 'sql':
        aggregations = len(re.findall(r'\bGROUP BY\b|\bDISTINCT\b', query, re.IGNORECASE))
    elif query_language == 'mql':
        aggregations = len(re.findall(r'\b\$group\b', query, re.IGNORECASE))
    else:
        raise ValueError(f"Unsupported query language: {query_language}")
    
    return aggregations



def count_sorting_limiting(row):
    query = row['query']
    query_language = row['query_language']
    
    if query_language == 'cypher':
        sorting_limiting = len(re.findall(r'\bORDER BY\b|\bLIMIT\b|\bSKIP\b', query, re.IGNORECASE))
    elif query_language == 'sparql':
        sorting_limiting = len(re.findall(r'\bORDER BY\b|\bLIMIT\b|\bOFFSET\b', query, re.IGNORECASE))
    elif query_language == 'sql':
        sorting_limiting = len(re.findall(r'\bORDER BY\b|\bLIMIT\b', query, re.IGNORECASE))
    elif query_language == 'mql':
        sorting_limiting = len(re.findall(r'\b\$sort\b|\b\$limit\b|\b\$skip\b', query, re.IGNORECASE))
    else:
        raise ValueError(f"Unsupported query language: {query_language}")
    
    return sorting_limiting



def count_projections(row):
    query = row['query']
    query_language = row['query_language']
    
    if query_language == 'cypher':
        projections = len(re.findall(r'\bRETURN\b', query, re.IGNORECASE))
    elif query_language == 'sparql':
        projections = len(re.findall(r'\bSELECT\b', query, re.IGNORECASE))
    elif query_language == 'sql':
        projections = len(re.findall(r'\bSELECT\b', query, re.IGNORECASE))
    elif query_language == 'mql':
        projections = len(re.findall(r'\b\$project\b', query, re.IGNORECASE))
    else:
        raise ValueError(f"Unsupported query language: {query_language}")
    
    return projections


def calculate_complexity_score(row, weights=WEIGHTS):
    distinct_vars = row['Distinct Variables']
    tokens = row['Tokens']
    keywords = row['Keywords']
    joins_traversals = row['Joins/Traversals']
    nesting_level = row['Nesting Depth']
    filters = row['Filters']
    aggregations = row['Aggregations']
    sorting_limiting = row['Sorting/Limiting']
    projections = row['Projections']
    
    weighted_sum = (
        (distinct_vars * weights['Distinct Variables']) +
        (tokens * weights['Tokens']) +
        (keywords * weights['Keywords']) +
        (joins_traversals * weights['Joins/Traversals']) +
        (nesting_level * weights['Nesting Depth']) +
        (filters * weights['Filters']) +
        (aggregations * weights['Aggregations']) +
        (sorting_limiting * weights['Sorting/Limiting']) +
        (projections * weights['Projections'])
    )
    
    total_weights = sum(weights.values())
    
    complexity_score = weighted_sum / total_weights
    
    return complexity_score