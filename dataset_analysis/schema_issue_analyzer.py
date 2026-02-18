import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _is_resource_only_range(value) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        normalized = [str(v).strip() for v in value if str(v).strip()]
        return len(normalized) == 1 and normalized[0].lower() == "resource"
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        return len(tokens) == 1 and tokens[0].lower() == "resource"
    return False


def _find_neo4j_issues(schema: dict) -> list[dict]:
    issues = []
    for relationship in schema.get("Relationships", []):
        end_labels = relationship.get("endNodeLabels")
        if not end_labels:
            issues.append(
                {
                    "relationshipType": relationship.get("relationshipType"),
                    "startNodeLabels": relationship.get("startNodeLabels", []),
                    "endNodeLabels": end_labels,
                }
            )
    return issues


def _find_rdf_issues(schema: dict) -> list[str]:
    issues = []
    properties = schema.get("Properties")

    if isinstance(properties, dict) and properties:
        for property_name, property_info in properties.items():
            if not isinstance(property_info, dict):
                continue
            if _is_resource_only_range(property_info.get("range")):
                issues.append(property_name)
        return issues

    ranges = schema.get("Ranges", {})
    if isinstance(ranges, dict):
        for property_name, range_value in ranges.items():
            if _is_resource_only_range(range_value):
                issues.append(property_name)

    return issues


def analyze_database_root(database_root: Path) -> dict:
    database_root = database_root.expanduser().resolve()
    if not database_root.exists() or not database_root.is_dir():
        raise FileNotFoundError(f"Database root does not exist or is not a directory: {database_root}")

    db_dirs = sorted([path for path in database_root.iterdir() if path.is_dir()])

    problematic_dbs = []
    ok_dbs = []
    missing_schema_dbs = []
    invalid_schema_dbs = []
    missing_ttl_dbs = []
    details = {}

    for db_dir in db_dirs:
        db_name = db_dir.name
        ttl_path = db_dir / f"{db_name}.ttl"
        neo4j_schema_path = db_dir / f"{db_name}.neo4j-schema.json"
        rdf_schema_path = db_dir / f"{db_name}.rdf-schema.json"

        has_ttl = ttl_path.exists()
        has_neo4j_schema = neo4j_schema_path.exists()
        has_rdf_schema = rdf_schema_path.exists()

        db_detail = {
            "has_ttl": has_ttl,
            "has_neo4j_schema": has_neo4j_schema,
            "has_rdf_schema": has_rdf_schema,
            "status": None,
            "neo4j_issue_count": 0,
            "rdf_issue_count": 0,
            "neo4j_issues": [],
            "rdf_issues": [],
        }

        if not has_ttl:
            db_detail["status"] = "missing_ttl"
            missing_ttl_dbs.append(db_name)
            details[db_name] = db_detail
            continue

        if not (has_neo4j_schema and has_rdf_schema):
            db_detail["status"] = "missing_schema_files"
            missing_schema_dbs.append(db_name)
            details[db_name] = db_detail
            continue

        try:
            neo4j_schema = json.loads(neo4j_schema_path.read_text(encoding="utf-8"))
            rdf_schema = json.loads(rdf_schema_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            db_detail["status"] = "invalid_schema_files"
            invalid_schema_dbs.append(db_name)
            details[db_name] = db_detail
            continue

        neo4j_issues = _find_neo4j_issues(neo4j_schema)
        rdf_issues = _find_rdf_issues(rdf_schema)

        db_detail["neo4j_issue_count"] = len(neo4j_issues)
        db_detail["rdf_issue_count"] = len(rdf_issues)
        db_detail["neo4j_issues"] = neo4j_issues
        db_detail["rdf_issues"] = rdf_issues

        if neo4j_issues or rdf_issues:
            db_detail["status"] = "problem"
            problematic_dbs.append(db_name)
        else:
            db_detail["status"] = "ok"
            ok_dbs.append(db_name)

        details[db_name] = db_detail

    report = {
        "database_root": str(database_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_db_subfolders": len(db_dirs),
            "dbs_with_ttl": sum(1 for detail in details.values() if detail["has_ttl"]),
            "dbs_with_both_schema_files": sum(
                1
                for detail in details.values()
                if detail["has_neo4j_schema"] and detail["has_rdf_schema"]
            ),
            "problem_count": len(problematic_dbs),
            "ok_count": len(ok_dbs),
            "missing_schema_files_count": len(missing_schema_dbs),
            "invalid_schema_files_count": len(invalid_schema_dbs),
            "missing_ttl_count": len(missing_ttl_dbs),
        },
        "problem_databases": problematic_dbs,
        "ok_databases": ok_dbs,
        "missing_schema_files_databases": missing_schema_dbs,
        "invalid_schema_files_databases": invalid_schema_dbs,
        "missing_ttl_databases": missing_ttl_dbs,
        "details": details,
    }
    return report


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze database subfolders for missing schema files and schema issues "
            "(Neo4j missing endNodeLabels, RDF range == Resource)."
        )
    )
    parser.add_argument(
        "database_root",
        type=Path,
        help=(
            "Path to a folder containing database subfolders. "
            "Each subfolder is expected to contain <db>.ttl, <db>.neo4j-schema.json, <db>.rdf-schema.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output JSON path. Default: <database_root>/<database_root.name>_schema_issue_stats.json"
        ),
    )
    args = parser.parse_args()

    report = analyze_database_root(args.database_root)
    database_root = args.database_root.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else database_root / f"{database_root.name}_schema_issue_stats.json"
    )

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved report to: {output_path}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
