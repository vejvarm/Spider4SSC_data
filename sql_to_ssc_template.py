import argparse
import json
import pathlib


def main(args: argparse.Namespace) -> None:
    input_path = pathlib.Path(args.input_json)
    output_path = pathlib.Path(args.output_json)

    with input_path.open() as handle:
        entries = json.load(handle)

    output_entries: list[dict] = []
    for entry in entries:
        sql_query = entry.get("query")
        out_entry = {
            "db_id": entry["db_id"],
            "question": entry["question"],
            "sql": sql_query,
            "sparql": None,
            "cypher": None,
        }

        output_entries.append(out_entry)

    with output_path.open("w") as handle:
        json.dump(output_entries, handle, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a ssc-like JSON file with only SQL queries and the rest null for running SSC eval on the test set."
    )
    parser.add_argument(
        "input_json",
        type=pathlib.Path,
        help="Path to the input JSON file (list of dicts with a 'sparql' field).",
    )
    parser.add_argument(
        "output_json",
        type=pathlib.Path,
        help="Path to write the output JSON file.",
    )
    main(parser.parse_args())
