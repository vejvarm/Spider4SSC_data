#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rdflib import Graph

# ----------------------------
# Normalization helpers
# ----------------------------

def norm_name(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.lower()

def _canon_key(s: str) -> str:
    # Normalize for case/spacing/punctuation-insensitive lookup.
    return norm_name(s)

def sparql_str_lit(val: str) -> str:
    """
    Convert SQL string tokens like "\"Nike\"" or "'Nike'" to a SPARQL literal.

    IMPORTANT: your TTL uses xsd:string typed literals (e.g. "Monitor"^^xsd:string),
    so we emit typed literals to ensure semantic equivalence in comparisons.
    """
    if not isinstance(val, str):
        raise TypeError("Expected string")
    v = val.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    v = v.replace("\\", "\\\\").replace('"', '\\"')
    return f"\"{v}\"^^xsd:string"

def is_null_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip().lower() in {"null", "none"}:
        return True
    return False

def is_numeric_like(v: Any) -> bool:
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        s = v.strip()
        if re.fullmatch(r"[-+]?\d+", s):
            return True
        if re.fullmatch(r"[-+]?\d*\.\d+", s):
            return True
    return False

# ----------------------------
# Schema structures (from test_tables.json)
# ----------------------------

@dataclass
class DbSchema:
    db_id: str
    table_names_original: List[str]
    column_names_original: List[Tuple[int, str]]  # (table_id, col_name) with [-1,"*"] at idx0
    primary_keys: List[int]
    foreign_keys: List[Tuple[int, int]]  # (child_col_id, parent_col_id)

    def col_ref(self, col_id: int) -> Tuple[int, str, str]:
        """Return (table_id, table_name, col_name_original) for a Spider col_id."""
        t_id, c_name = self.column_names_original[col_id]
        if t_id < 0:
            return (-1, "*", "*")
        t_name = self.table_names_original[t_id]
        return (t_id, t_name, c_name)

# ----------------------------
# TTL grounding: prefixes + predicate IRIs
# ----------------------------

@dataclass
class TtlGrounding:
    g: Graph
    # Keys are prefix labels WITHOUT trailing ":" (except the default prefix label stored as ":"), e.g.:
    # {":": "http://valuenet/ontop/", "products": ".../Products#", "xsd": "http://www.w3.org/2001/XMLSchema#"}
    prefix_map: Dict[str, str]
    root_prefix: str = ":"  # use ":" for classes per your data
    pred_local_map: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    pred_obj_map: Dict[str, bool] = field(default_factory=dict)
    ns_to_prefix: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure ":" exists (fallback to your valuennet root if absent)
        if ":" not in self.prefix_map:
            if "" in self.prefix_map:
                self.prefix_map[":"] = self.prefix_map[""]
            else:
                self.prefix_map[":"] = "http://valuenet/ontop/"

        # Ensure xsd points to standard namespace when missing
        if "xsd" not in self.prefix_map:
            self.prefix_map["xsd"] = "http://www.w3.org/2001/XMLSchema#"

        self.ns_to_prefix = {
            ns: p
            for p, ns in self.prefix_map.items()
            if p not in {":", "xsd", ""}
        }
        self._build_pred_maps()

    @staticmethod
    def _split_uri(uri: str) -> Tuple[str, str]:
        if "#" in uri:
            ns, local = uri.rsplit("#", 1)
            return ns + "#", local
        ns, local = uri.rsplit("/", 1)
        return ns + "/", local

    def _build_pred_maps(self) -> None:
        try:
            from rdflib.term import URIRef
        except Exception:
            URIRef = None  # type: ignore

        pred_local_map: Dict[str, Dict[str, List[str]]] = {}
        pred_obj_map: Dict[str, bool] = {}

        for _, p, o in self.g:
            if URIRef is not None and not isinstance(p, URIRef):
                continue
            uri = str(p)
            ns, local = self._split_uri(uri)
            pfx = self.ns_to_prefix.get(ns)
            if not pfx:
                continue

            key = _canon_key(local)
            pred_local_map.setdefault(pfx, {}).setdefault(key, []).append(local)

            if URIRef is not None:
                is_obj = isinstance(o, URIRef)
                pred_obj_map[uri] = pred_obj_map.get(uri, False) or is_obj

        self.pred_local_map = pred_local_map
        self.pred_obj_map = pred_obj_map

    def find_table_prefix(self, table_name_original: str) -> str:
        """
        Try to find a declared prefix whose namespace looks like .../<Table>#.
        Fallback: use normalized table name as prefix (still valid in SPARQL if declared).
        """
        t = table_name_original.strip()
        candidates = {
            t,
            _canon_key(t),
            re.sub(r"\s+", "_", t),
            re.sub(r"[^0-9A-Za-z_]", "_", t),
        }
        for p, ns in self.prefix_map.items():
            if p in {":", "xsd", ""}:
                continue
            for cand in candidates:
                if not cand:
                    continue
                if re.search(rf"/{re.escape(cand)}#?$", ns, flags=re.IGNORECASE) or re.search(
                    rf"/{re.escape(cand)}#", ns, flags=re.IGNORECASE
                ):
                    return p

        t_key = _canon_key(t)
        for p in self.prefix_map:
            if p in {":", "xsd", ""}:
                continue
            if _canon_key(p) == t_key:
                return p

        return norm_name(t)

    def class_iri(self, table_name_original: str) -> str:
        return f"{self.root_prefix}{norm_name(table_name_original)}"

    def pred_iri(self, table_name_original: str, col_name_original: str) -> Tuple[str, str]:
        pfx = self.find_table_prefix(table_name_original)
        local = self._resolve_pred_local(pfx, col_name_original)
        if pfx == ":":
            return pfx, f":{local}"
        return pfx, f"{pfx}:{local}"

    def _resolve_pred_local(self, pfx: str, col_name_original: str) -> str:
        locals_by_key = self.pred_local_map.get(pfx)
        if locals_by_key:
            # Prefer exact/case-insensitive matches when multiple locals share a canonical key.
            all_locals = [l for lst in locals_by_key.values() for l in lst]
            for local in all_locals:
                if local == col_name_original:
                    return local
            for local in all_locals:
                if local.lower() == col_name_original.lower():
                    return local
            key = _canon_key(col_name_original)
            if key in locals_by_key:
                return locals_by_key[key][0]

        return norm_name(col_name_original)

    def pred_uri(self, table_name_original: str, col_name_original: str) -> Optional[str]:
        _, pred = self.pred_iri(table_name_original, col_name_original)
        try:
            return str(self.g.namespace_manager.expand_curie(pred))
        except Exception:
            return None

    def predicate_is_object(self, table_name_original: str, col_name_original: str) -> bool:
        uri = self.pred_uri(table_name_original, col_name_original)
        if not uri:
            return False
        return self.pred_obj_map.get(uri, False)

    def prefixes_sparql(self, used_prefixes: List[str]) -> str:
        """
        Emit valid SPARQL PREFIX lines:
          - default prefix label ":" emits: PREFIX : <...>
          - normal labels emit: PREFIX products: <...>
        """
        out = []
        seen = set()
        for p in used_prefixes:
            if p in seen:
                continue
            seen.add(p)
            ns = self.prefix_map.get(p)
            if ns is None:
                if p == ":":
                    ns = "http://valuenet/ontop/"
                else:
                    ns = f"http://valuenet/ontop/{p.capitalize()}#"

            if p == ":":
                out.append(f"PREFIX : <{ns}>")
            else:
                out.append(f"PREFIX {p}: <{ns}>")
        return "\n".join(out)

# ----------------------------
# Spider SQL AST decoding
# ----------------------------

OP_MAP = {
    2: "=",   # equality
    3: ">",   # greater
    4: "<",   # less
    5: ">=",  # greater eq
    6: "<=",  # less eq
    7: "!=",  # not equal
}

AGG_MAP = {
    0: None,
    1: "MAX",
    2: "MIN",
    3: "COUNT",
    4: "SUM",
    5: "AVG",
}

def parse_col_unit(col_unit: List[Any]) -> Tuple[Optional[int], bool]:
    """
    Spider col_unit typically: [agg_id_or_0, col_id, isDistinct]
    Return (col_id, is_distinct)
    """
    if not isinstance(col_unit, list) or len(col_unit) < 2:
        return (None, False)
    col_id = col_unit[1]
    is_dist = bool(col_unit[2]) if len(col_unit) > 2 else False
    return (col_id, is_dist)

def parse_val_unit(val_unit: Any) -> Tuple[Optional[str], Optional[int], bool]:
    """
    val_unit often: [agg_id, col_unit, _]
    Example: [0, [0, 12, false], null]
    Returns (agg_func, col_id, isDistinctCol)
    """
    if not isinstance(val_unit, list) or len(val_unit) < 2:
        return (None, None, False)
    agg_id = val_unit[0]
    col_unit = val_unit[1]
    agg = AGG_MAP.get(agg_id)
    if isinstance(col_unit, list):
        col_id, is_dist = parse_col_unit(col_unit)
        return (agg, col_id, is_dist)
    return (agg, None, False)

# ----------------------------
# SPARQL builder
# ----------------------------

class SparqlBuilder:
    def __init__(self, schema: DbSchema, ttl: TtlGrounding, var_prefix: str = "t"):
        self.schema = schema
        self.ttl = ttl
        self.var_prefix = var_prefix
        self.used_prefixes: List[str] = [":", "xsd"]
        self.patterns: List[str] = []
        self.optionals: List[str] = []
        self.filters: List[str] = []
        self.group_by: List[str] = []
        self.having: List[str] = []
        self.order_by: List[str] = []
        self.limit: Optional[int] = None
        self.select_vars: List[str] = []
        self.select_exprs: List[str] = []
        self.distinct: bool = False

        self.table_vars: Dict[int, str] = {}   # table_id -> ?t1
        self.table_name_by_var: Dict[str, str] = {}  # ?t1 -> table_name_original
        self.colvar_cache: Dict[Tuple[str, str], str] = {}  # (tvar, pred) -> ?var

        self.fk_pairs = set(schema.foreign_keys)  # (child_col_id, parent_col_id)

        # PATCH: track whether SQL asked for SELECT * explicitly (Spider encodes as col_id == 0)
        self.select_star: bool = False
        self.subquery_counter: int = 0

    def _default_tvar(self, idx: int) -> str:
        return f"?{self.var_prefix}{idx}"

    def _ensure_table(self, table_name: str, tvar: str):
        self.patterns.append(f"{tvar} a {self.ttl.class_iri(table_name)} .")
        self.table_name_by_var[tvar] = table_name

    def _ensure_col(self, tvar: str, table_name: str, col_name: str, optional: bool = False) -> str:
        pfx, pred = self.ttl.pred_iri(table_name, col_name)
        if pfx not in self.used_prefixes:
            self.used_prefixes.append(pfx)

        key = (tvar, pred)
        if key in self.colvar_cache:
            return self.colvar_cache[key]

        v = f"{tvar}_{norm_name(col_name)}"
        triple = f"{tvar} {pred} {v} ."
        if optional:
            self.optionals.append(f"OPTIONAL {{ {triple} }}")
        else:
            self.patterns.append(triple)

        self.colvar_cache[key] = v
        return v

    def _merge_prefixes(self, prefixes: List[str]) -> None:
        for p in prefixes:
            if p not in self.used_prefixes:
                self.used_prefixes.append(p)

    def _detect_object_fk(self, child_table: str, child_col: str) -> bool:
        """Heuristic: predicate resolves to any triple with IRI object in TTL."""
        return self.ttl.predicate_is_object(child_table, child_col)

    @staticmethod
    def _is_interleaved_where(where_clause):
        return any(isinstance(x, str) and x.lower() in {"and", "or"} for x in where_clause)

    @staticmethod
    def _extract_conditions(interleaved: List[Any]) -> List[List[Any]]:
        """
        Spider can encode cond lists interleaved with 'and'/'or' strings:
        [cond, 'and', cond, ...]
        Return only the list-conditions.
        """
        out = []
        for x in interleaved:
            if isinstance(x, str) and x.lower() in {"and", "or"}:
                continue
            if isinstance(x, list):
                out.append(x)
        return out

    # ---- FROM / JOIN ----

    def apply_from(self, sql_from: Dict[str, Any]):
        # table_units: [["table_unit", table_id], ...]
        for i, tu in enumerate(sql_from.get("table_units", []), start=1):
            _, table_id = tu
            table_name = self.schema.table_names_original[table_id]
            tvar = f"?{self.var_prefix}{i}"
            self.table_vars[table_id] = tvar
            self._ensure_table(table_name, tvar)

        raw_conds = sql_from.get("conds", [])
        conds = self._extract_conditions(raw_conds) if self._is_interleaved_where(raw_conds) else raw_conds

        for cond in conds:
            # expected: [not, op_id, val_unit, col_unit, null]
            if not isinstance(cond, list) or len(cond) < 5:
                continue

            not_flag, op_id, val_unit, col_unit, _ = cond
            if op_id != 2:
                continue  # JOIN should be "="

            _, left_col_id, _ = parse_val_unit(val_unit)
            right_col_id, _ = parse_col_unit(col_unit) if isinstance(col_unit, list) else (None, False)

            if left_col_id is None or right_col_id is None:
                continue

            lt_id, lt_name, lc_name = self.schema.col_ref(left_col_id)
            rt_id, rt_name, rc_name = self.schema.col_ref(right_col_id)

            left_tvar = self.table_vars.get(lt_id, self._default_tvar(1))
            right_tvar = self.table_vars.get(rt_id, self._default_tvar(2))

            # Prefer object-property join when FK direction + TTL supports it
            if (left_col_id, right_col_id) in self.fk_pairs:
                # left is child, right is parent
                if self._detect_object_fk(lt_name, lc_name):
                    pfx, pred = self.ttl.pred_iri(lt_name, lc_name)
                    if pfx not in self.used_prefixes:
                        self.used_prefixes.append(pfx)
                    self.patterns.append(f"{left_tvar} {pred} {right_tvar} .")
                else:
                    lv = self._ensure_col(left_tvar, lt_name, lc_name)
                    rv = self._ensure_col(right_tvar, rt_name, rc_name)
                    self.filters.append(f"FILTER({lv} = {rv}) .")
            elif (right_col_id, left_col_id) in self.fk_pairs:
                # right is child, left is parent
                if self._detect_object_fk(rt_name, rc_name):
                    pfx, pred = self.ttl.pred_iri(rt_name, rc_name)
                    if pfx not in self.used_prefixes:
                        self.used_prefixes.append(pfx)
                    self.patterns.append(f"{right_tvar} {pred} {left_tvar} .")
                else:
                    lv = self._ensure_col(left_tvar, lt_name, lc_name)
                    rv = self._ensure_col(right_tvar, rt_name, rc_name)
                    self.filters.append(f"FILTER({lv} = {rv}) .")
            else:
                lv = self._ensure_col(left_tvar, lt_name, lc_name)
                rv = self._ensure_col(right_tvar, rt_name, rc_name)
                self.filters.append(f"FILTER({lv} = {rv}) .")

    # ---- SELECT ----

    def apply_select(self, sql_select: List[Any]):
        self.distinct = bool(sql_select[0])
        items = sql_select[1]
        agg_counter = 0

        for item in items:
            agg_id, val_unit = item
            agg = AGG_MAP.get(agg_id)

            agg_in_val, col_id, col_is_distinct = parse_val_unit(val_unit)

            # PATCH: Spider encodes SELECT * as col_id == 0 (column_names_original[0] is [-1,"*"])
            # We do NOT expand properties; we simply emit SELECT * later, and avoid creating any _:_ triple.
            if agg is None and (col_id == 0 or col_id is None) and agg_in_val is None:
                self.select_star = True
                continue

            if agg == "COUNT":
                # COUNT(*) or COUNT(col) or COUNT(DISTINCT col)
                if col_id is None or col_id == 0:
                    self.select_exprs.append("(COUNT(*) AS ?aggregation_all)")
                else:
                    t_id, t_name, c_name = self.schema.col_ref(col_id)
                    tvar = self.table_vars.get(t_id, self._default_tvar(1))
                    v = self._ensure_col(tvar, t_name, c_name)
                    agg_counter += 1
                    if col_is_distinct:
                        self.select_exprs.append(
                            f"(COUNT(DISTINCT {v}) AS ?aggregation_count_{agg_counter})"
                        )
                    else:
                        self.select_exprs.append(
                            f"(COUNT({v}) AS ?aggregation_count_{agg_counter})"
                        )

            elif agg in {"AVG", "MIN", "MAX", "SUM"}:
                if col_id is None or col_id == 0:
                    continue
                t_id, t_name, c_name = self.schema.col_ref(col_id)
                tvar = self.table_vars.get(t_id, self._default_tvar(1))
                v = self._ensure_col(tvar, t_name, c_name)
                agg_counter += 1
                self.select_exprs.append(
                    f"({agg}({v}) AS ?aggregation_{norm_name(c_name)}_{agg_counter})"
                )

            else:
                # plain column
                if col_id is None or col_id == 0:
                    # if col_id==0 here, it's really "*", handled by select_star
                    self.select_star = True
                    continue
                t_id, t_name, c_name = self.schema.col_ref(col_id)
                tvar = self.table_vars.get(t_id, self._default_tvar(1))
                v = self._ensure_col(tvar, t_name, c_name)
                self.select_vars.append(v)

    # ---- WHERE (supports interleaved boolean) ----

    @staticmethod
    def _is_atomic_cond(x: Any) -> bool:
        return (
            isinstance(x, list)
            and len(x) == 5
            and isinstance(x[0], (bool, int))
            and isinstance(x[1], int)
        )

    def _rhs_literal(self, value: Any) -> str:
        if isinstance(value, str):
            return sparql_str_lit(value)
        if is_numeric_like(value):
            return str(value)
        return str(value)

    def _rhs_term(self, value: Any) -> Optional[str]:
        if isinstance(value, list):
            col_id, _ = parse_col_unit(value) if len(value) >= 2 else (None, False)
            if col_id is None:
                _, col_id, _ = parse_val_unit(value)
            if col_id is not None and col_id != 0:
                t_id, t_name, c_name = self.schema.col_ref(col_id)
                tvar = self.table_vars.get(t_id, self._default_tvar(1))
                return self._ensure_col(tvar, t_name, c_name)
            return None
        if isinstance(value, dict):
            return None
        return self._rhs_literal(value)

    @staticmethod
    def _like_pattern_to_regex(raw: Any) -> Optional[str]:
        if not isinstance(raw, str):
            return None
        v = raw.strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        # Escape regex metacharacters except SQL LIKE wildcards.
        out = []
        for ch in v:
            if ch == "%":
                out.append(".*")
            elif ch == "_":
                out.append(".")
            else:
                out.append(re.escape(ch))
        return "^" + "".join(out) + "$"

    def _cond_to_expr(self, cond: List[Any]) -> Optional[str]:
        try:
            not_flag, op_id, val_unit, value, value2 = cond
        except ValueError:
            raise ValueError(f"Invalid where condition structure: {cond}")

        _, col_id, _ = parse_val_unit(val_unit)
        if col_id is None or col_id == 0:
            return None

        t_id, t_name, c_name = self.schema.col_ref(col_id)
        tvar = self.table_vars.get(t_id, self._default_tvar(1))

        if op_id == 1:
            low = self._rhs_term(value)
            high = self._rhs_term(value2)
            if low is None or high is None:
                return None
            v = self._ensure_col(tvar, t_name, c_name)
            expr = f"({v} >= {low} && {v} <= {high})"
            return f"!{expr}" if not_flag else expr

        if op_id == 9:
            v = self._ensure_col(tvar, t_name, c_name)
            regex = self._like_pattern_to_regex(value)
            if regex is None:
                return None
            expr = f"REGEX(STR({v}), \"{regex}\")"
            return f"!{expr}" if not_flag else expr

        op = OP_MAP.get(op_id)
        if op is None:
            return None

        if is_null_value(value):
            v = self._ensure_col(tvar, t_name, c_name, optional=True)
            return f"BOUND({v})" if not_flag else f"!BOUND({v})"

        v = self._ensure_col(tvar, t_name, c_name)
        rhs = self._rhs_term(value)
        if rhs is None:
            return None

        expr = f"({v} {op} {rhs})"
        if not_flag:
            expr = f"!{expr}"
        return expr

    def _cond_to_in_filter(self, cond: List[Any]) -> Optional[str]:
        try:
            not_flag, _op_id, val_unit, value, _ = cond
        except ValueError:
            return None

        _, col_id, _ = parse_val_unit(val_unit)
        if col_id is None or col_id == 0:
            return None

        t_id, t_name, c_name = self.schema.col_ref(col_id)
        tvar = self.table_vars.get(t_id, self._default_tvar(1))
        outer_v = self._ensure_col(tvar, t_name, c_name)

        # IN (subquery)
        if isinstance(value, dict):
            sub = self._build_subquery(value, outer_v)
            if sub is None:
                return None
            sub_query, _sub_var, sub_prefixes = sub
            self._merge_prefixes(sub_prefixes)
            indent = "  "
            sub_query = "\n".join(f"{indent}{line}" for line in sub_query.splitlines())
            exists_kw = "NOT EXISTS" if not_flag else "EXISTS"
            return f"FILTER {exists_kw} {{\n{sub_query}\n}}"

        # IN (literal list)
        if isinstance(value, list):
            items = [self._rhs_literal(v) for v in value if not is_null_value(v)]
            if not items:
                return None
            expr = f"{outer_v} IN ({', '.join(items)})"
            if not_flag:
                expr = f"!({expr})"
            return f"FILTER({expr}) ."

        rhs = self._rhs_literal(value)
        expr = f"({outer_v} = {rhs})"
        if not_flag:
            expr = f"!{expr}"
        return f"FILTER({expr}) ."

    def _build_subquery(
        self, sub_sql: Dict[str, Any], outer_var: Optional[str]
    ) -> Optional[Tuple[str, str, List[str]]]:
        self.subquery_counter += 1
        sub_prefix = f"s{self.subquery_counter}"
        sub = SparqlBuilder(self.schema, self.ttl, var_prefix=sub_prefix)
        try:
            sub.apply_from(sub_sql.get("from", {}))
            sub.apply_select(sub_sql.get("select", [False, []]))
            sub.apply_where(sub_sql.get("where", []))
            sub.apply_group_by(sub_sql.get("groupBy", []))
            sub.apply_having(sub_sql.get("having", []))
            sub.apply_order_by(sub_sql.get("orderBy", []))
            sub.limit = sub_sql.get("limit", None)
        except Exception:
            return None

        if not sub.select_vars:
            return None
        sub_var = sub.select_vars[0]

        extra_filters: List[str] = []
        if outer_var:
            extra_filters.append(f"FILTER({sub_var} = {outer_var}) .")

        sub_query = sub.to_subquery(extra_filters=extra_filters, override_select=[sub_var])
        return sub_query, sub_var, sub.used_prefixes

    def _build_interleaved_bool_expr(self, where_clause: List[Any]) -> Optional[str]:
        def tok_to_expr(tok: Any) -> Optional[str]:
            if self._is_atomic_cond(tok):
                return self._cond_to_expr(tok)
            if isinstance(tok, list) and self._is_interleaved_where(tok):
                inner = self._build_interleaved_bool_expr(tok)
                return f"({inner})" if inner else None
            if isinstance(tok, list) and len(tok) == 1 and self._is_atomic_cond(tok[0]):
                return self._cond_to_expr(tok[0])
            return None

        def op_to_sparql(op: str) -> str:
            op_l = op.strip().lower()
            if op_l == "and":
                return "&&"
            if op_l == "or":
                return "||"
            raise ValueError(f"Unknown boolean operator in WHERE: {op}")

        tokens = where_clause[:]
        if not tokens:
            return None

        first = tok_to_expr(tokens[0])
        if first is None:
            return None

        expr = first
        i = 1
        while i < len(tokens):
            op_tok = tokens[i]
            rhs_tok = tokens[i + 1] if i + 1 < len(tokens) else None

            if not isinstance(op_tok, str):
                raise ValueError(f"Expected 'and/or' operator token, got: {op_tok}")

            rhs_expr = tok_to_expr(rhs_tok)
            if rhs_expr is None:
                return None

            bop = op_to_sparql(op_tok)
            expr = f"({expr} {bop} {rhs_expr})"
            i += 2

        return expr

    def apply_where(self, where_conds: List[Any]):
        if self._is_interleaved_where(where_conds):
            combined = self._build_interleaved_bool_expr(where_conds)
            if combined:
                self.filters.append(f"FILTER({combined}) .")
            return

        for cond in where_conds:
            if not self._is_atomic_cond(cond):
                continue
            if cond[1] == 8:
                in_filter = self._cond_to_in_filter(cond)
                if in_filter:
                    self.filters.append(in_filter)
                continue
            expr = self._cond_to_expr(cond)
            if expr:
                self.filters.append(f"FILTER({expr}) .")

    # ---- GROUP BY / HAVING / ORDER BY ----

    def apply_group_by(self, group_by: List[Any]):
        for col_unit in group_by:
            col_id, _ = parse_col_unit(col_unit)
            if col_id is None or col_id == 0:
                continue
            t_id, t_name, c_name = self.schema.col_ref(col_id)
            tvar = self.table_vars.get(t_id, self._default_tvar(1))
            v = self._ensure_col(tvar, t_name, c_name)
            self.group_by.append(v)

    def apply_having(self, having: List[Any]):
        for cond in having:
            if not self._is_atomic_cond(cond):
                continue
            not_flag, op_id, val_unit, value, _ = cond
            op = OP_MAP.get(op_id)
            agg, col_id, _ = parse_val_unit(val_unit)
            if col_id is None or col_id == 0 or agg is None or op is None:
                continue
            t_id, t_name, c_name = self.schema.col_ref(col_id)
            tvar = self.table_vars.get(t_id, self._default_tvar(1))
            v = self._ensure_col(tvar, t_name, c_name)

            rhs = self._rhs_literal(value)
            expr = f"({agg}({v}) {op} {rhs})"
            if not_flag:
                expr = f"!{expr}"
            self.having.append(expr)

    def apply_order_by(self, order_by: List[Any]):
        """
        PATCH: robust handling of ORDER BY COUNT(*)
        Spider may encode ORDER BY count(*) as:
          orderBy: ["desc", [ [0, [3, 0, false], null] ]]
        i.e., outer agg_id=0 (None) but inner col_unit[0]=3 (COUNT) and col_id=0 means "*".
        We must emit ORDER BY DESC(COUNT(*)) and NOT touch schema col_id=0 (which caused _:_).
        """
        if not order_by:
            return
        direction = order_by[0]  # "asc" or "desc"
        items = order_by[1]

        for it in items:
            if not isinstance(it, list) or len(it) < 2:
                continue
            outer_agg_id = it[0]
            col_unit = it[1]

            outer_agg = AGG_MAP.get(outer_agg_id)

            # Detect "COUNT(*)" special form in Spider ORDER BY
            # - either outer agg is COUNT and col_id==0
            # - or inner col_unit carries COUNT (col_unit[0]==3) and col_id==0
            inner_agg_id = None
            inner_col_id = None
            inner_is_distinct = False
            if isinstance(col_unit, list) and len(col_unit) >= 2:
                inner_agg_id = col_unit[0] if isinstance(col_unit[0], int) else None
                inner_col_id = col_unit[1] if isinstance(col_unit[1], int) else None
                inner_is_distinct = bool(col_unit[2]) if len(col_unit) > 2 else False

            # Case A: COUNT(*) is represented via inner agg_id
            if inner_agg_id == 3 and inner_col_id == 0:
                expr = "COUNT(*)"
                self.order_by.append(f"{direction.upper()}({expr})")
                continue

            # Case B: COUNT(*) represented via outer agg
            if outer_agg == "COUNT" and (inner_col_id == 0 or inner_col_id is None):
                expr = "COUNT(*)"
                self.order_by.append(f"{direction.upper()}({expr})")
                continue

            # General case: determine aggregation to apply
            agg = outer_agg
            if agg is None and inner_agg_id is not None and inner_agg_id in AGG_MAP:
                agg = AGG_MAP.get(inner_agg_id)

            # Now handle COUNT(col) / COUNT(DISTINCT col) / other aggs / plain var
            col_id, _ = parse_col_unit(col_unit) if isinstance(col_unit, list) else (None, False)
            if col_id is None:
                continue

            # If col_id==0 here, it's "*" but not COUNT(*) (handled above); skip to avoid _:_ pollution.
            if col_id == 0:
                continue

            t_id, t_name, c_name = self.schema.col_ref(col_id)
            tvar = self.table_vars.get(t_id, self._default_tvar(1))
            v = self._ensure_col(tvar, t_name, c_name)

            if agg == "COUNT":
                # If Spider marks DISTINCT at inner level, respect it
                if inner_is_distinct:
                    expr = f"COUNT(DISTINCT {v})"
                else:
                    expr = f"COUNT({v})"
            elif agg in {"AVG", "MIN", "MAX", "SUM"}:
                expr = f"{agg}({v})"
            else:
                expr = v

            self.order_by.append(f"{direction.upper()}({expr})")

    # ---- Render ----

    def _select_parts(self, override_select: Optional[List[str]] = None) -> List[str]:
        if override_select is not None:
            return override_select
        select_parts: List[str] = []
        select_parts.extend(self.select_vars)
        select_parts.extend(self.select_exprs)
        if not select_parts:
            return ["*"]
        return select_parts

    def _where_parts(self, extra_filters: Optional[List[str]] = None) -> List[str]:
        where_parts: List[str] = []
        where_parts.extend(self.patterns)
        where_parts.extend(self.optionals)
        where_parts.extend(self.filters)
        if extra_filters:
            where_parts.extend(extra_filters)
        return where_parts

    def _render_query(
        self,
        include_prefixes: bool = True,
        extra_filters: Optional[List[str]] = None,
        override_select: Optional[List[str]] = None,
    ) -> str:
        distinct = "DISTINCT " if self.distinct else ""
        select_parts = self._select_parts(override_select=override_select)
        where_parts = self._where_parts(extra_filters=extra_filters)

        q: List[str] = []
        if include_prefixes:
            prefix_block = self.ttl.prefixes_sparql(self.used_prefixes)
            if prefix_block:
                q.append(prefix_block)
        q.append(f"SELECT {distinct}{' '.join(select_parts)} WHERE {{")
        for p in where_parts:
            q.append(f"  {p}")
        q.append("}")

        if self.group_by:
            q.append("GROUP BY " + " ".join(self.group_by))
        if self.having:
            q.append("HAVING(" + " && ".join(self.having) + ")")
        if self.order_by:
            q.append("ORDER BY " + " ".join(self.order_by))
        if self.limit is not None:
            q.append(f"LIMIT {self.limit}")

        return "\n".join(q)

    def to_subquery(
        self,
        extra_filters: Optional[List[str]] = None,
        override_select: Optional[List[str]] = None,
    ) -> str:
        return self._render_query(
            include_prefixes=False,
            extra_filters=extra_filters,
            override_select=override_select,
        )

    def to_sparql(self) -> str:
        return self._render_query(include_prefixes=True)

# ----------------------------
# Conversion entry points
# ----------------------------

def load_db_schema(tables_json: List[Dict[str, Any]], db_id: str) -> DbSchema:
    row = next(x for x in tables_json if x["db_id"] == db_id)
    return DbSchema(
        db_id=db_id,
        table_names_original=row["table_names_original"],
        column_names_original=[(t, c) for (t, c) in row["column_names_original"]],
        primary_keys=row.get("primary_keys", []),
        foreign_keys=[tuple(x) for x in row.get("foreign_keys", [])],
    )

def load_ttl(ttl_path: str) -> TtlGrounding:
    g = Graph()
    g.parse(ttl_path, format="turtle")

    # rdflib returns prefixes without trailing ":" (e.g. "products"), and "" for default.
    # We store default under ":" for SPARQL default prefix, and keep others as-is.
    prefix_map: Dict[str, str] = {}
    for p, ns in g.namespace_manager.namespaces():
        if p == "":
            prefix_map[":"] = str(ns)
        else:
            prefix_map[p] = str(ns)

    return TtlGrounding(g=g, prefix_map=prefix_map)

def convert_one(example: Dict[str, Any], schema: DbSchema, ttl: TtlGrounding) -> str:
    sql = example["sql"]
    b = SparqlBuilder(schema, ttl)
    try:
        b.apply_from(sql.get("from", {}))
        b.apply_select(sql.get("select", [False, []]))
        b.apply_where(sql.get("where", []))
        b.apply_group_by(sql.get("groupBy", []))
        b.apply_having(sql.get("having", []))
        b.apply_order_by(sql.get("orderBy", []))
        b.limit = sql.get("limit", None)
    except Exception as e:
        return f"conversion error `{repr(e)}`"

    return b.to_sparql()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="Path to test.json (Spider format with sql AST)")
    ap.add_argument("--tables", required=True, help="Path to test_tables.json")
    ap.add_argument("--ttl_dir", required=True, help="Directory containing <db_id>.ttl files")
    ap.add_argument("--out", required=True, help="Output JSON with added sparql field")
    args = ap.parse_args()

    test = json.load(open(args.test, "r", encoding="utf-8"))
    tables = json.load(open(args.tables, "r", encoding="utf-8"))

    ttl_cache: Dict[str, TtlGrounding] = {}
    schema_cache: Dict[str, DbSchema] = {}

    out = []
    for ex in test:
        db_id = ex["db_id"]
        if db_id not in schema_cache:
            schema_cache[db_id] = load_db_schema(tables, db_id)
        if db_id not in ttl_cache:
            ttl_cache[db_id] = load_ttl(f"{args.ttl_dir}/{db_id}/{db_id}.ttl")

        sparql = convert_one(ex, schema_cache[db_id], ttl_cache[db_id])

        new_ex = dict(ex)
        new_ex["sparql"] = sparql
        out.append(new_ex)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
