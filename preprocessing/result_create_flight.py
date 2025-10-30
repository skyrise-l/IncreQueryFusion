import os
import json
import glob
import argparse
from collections import defaultdict
import copy
import re

def normalize(s: str) -> str:
    """Normalize key value (e.g., ISBN, or other primary keys)."""
    return str(s).strip().strip('"').strip("'")

def parse_golden(golden_path: str):
    """Parse the golden file containing entity ids (primary keys)."""
    entity_ids = []
    with open(golden_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue  # skip blank separators
            if "\t" in raw:
                ent_id = raw.split("\t", 1)[0].strip()
                entity_ids.append(ent_id)
            else:
                entity_ids.append(raw.strip())
    return entity_ids

def parse_sid_from_filename(filename: str):
    """Extract sid from filenames like 'A1Books_1.txt' -> 1."""
    m = re.search(r'_(\d+)\.txt$', filename)
    if m:
        return int(m.group(1))
    return None

def load_raw_sources(raw_dir: str, key_column: str):
    """Load all *.txt files from raw_dir, keyed by sid parsed from filename."""
    sources = {}
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))

    for path in paths:
        fname = os.path.basename(path)
        sid = parse_sid_from_filename(fname)
        if sid is None:
            print(f"[WARN] Skip file without sid pattern: {fname}")
            continue
        if sid in sources:
            print(f"[WARN] Multiple files for sid={sid}. ")
            continue

        rows = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = True  # Flag to indicate the first line (column headers)
                for i, raw in enumerate(f, start=-1):
                    if first_line:
                        first_line = False  # Skip the first line
                        continue
                    
                    if not raw.strip():
           
                        continue
                    
                    # Split the row and clean up the parts
                    parts = raw.rstrip("\n").split("\t")
                    
                    # If the key column is empty, skip this row
                    key_value = normalize(parts[1])  # Assuming key column is the second column
                    if not key_value:

                        continue  # Skip rows where the key column is empty
                    
                    # Otherwise, we collect the data row
                    data = {key_column: key_value, "line": i, "data": parts}
                    rows.append(data)
        except Exception as e:
            print(f"[WARN] Failed to read {fname}: {e}")
        
        sources[sid] = rows
    return sources

def load_logs(log_file: str):
    """Load and group logs by timestamp."""
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        logs = json.load(f)
    
    # Group logs by timestamp
    grouped_logs = defaultdict(list)
    for log_entry in logs:
        timestamp = log_entry.get("timestamp", 0)
        if timestamp:
            grouped_logs[timestamp].append(log_entry)
    
    # Sort timestamps in ascending order
    sorted_timestamps = sorted(grouped_logs.keys())
    
    return grouped_logs, sorted_timestamps
    
def apply_insert_to_sources(sources, key_column, sid: int, data: dict, index):
    """Apply an insert to a given sid and update the key_column index."""
    key_value = normalize(data.get(key_column, ""))
    if not key_value:
        return None

    # Append the new record to the respective data source based on sid
    if sid not in sources:
        sources[sid] = []
    line = len(sources[sid])  # Line number is the next available line
    sources[sid].append({"data": data, "line": line, key_column: key_value})
    index[key_value][sid].append(line)

def build_index(sources, key_column):
    """Build a global index: key_column -> list of occurrences across all sources."""
    index = defaultdict(lambda: defaultdict(list)) 
    for sid, rows in sources.items():
        for row in rows:
            key_value = row[key_column]
            index[key_value][sid].append(row["line"])
    return index



def generate_answer_view(index, entity_ids):
    """Generate answer views based on the index and queries."""
    answer_views = []  # 创建一个字典来存储按 src（sid）为键的行号列表
    total_sum = 0
    for eid in entity_ids:
        eid_norm = normalize(eid)  # 标准化实体ID
        occurrences = index.get(eid_norm, [])  # 获取对应的出现记录

        total_sum += len(occurrences)

        # 按 sid 分组，每个 sid 对应一组行号
        answer_views.append(copy.deepcopy(occurrences))

        if len(occurrences)> 100:
            print(f"发现过大长度 {len(occurrences)},  {eid}")

    return answer_views 

def update_query_truth(query_truth, entity_ids, index, timestamp):
    """Update the query_truth with new answer views."""
    views = generate_answer_view(index, entity_ids)
    for i, view in enumerate(views):
        if "answer_view" not in query_truth[i]:
            query_truth[i]["answer_view"] = {}
        query_truth[i]["answer_view"][timestamp] = view

def write_query_truth(output_dir, query_truth):
    """Write the updated query_truth to the output directory."""
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(query_truth, f, ensure_ascii=False, indent=2)

def check_queries_have_answers(entity_ids, index):
    for id, eid in enumerate(entity_ids):
        eid_norm = normalize(eid)
        if eid_norm not in index or not any(index[eid_norm].values()):
            print(id, eid)
            return False
    return True

def main():
    # Set up argument parser
    ap = argparse.ArgumentParser(description="Process incremental updates and generate query views.")
    ap.add_argument("--raw-data-dir", required=True, help="Folder with initial *.txt data sources (tab-separated).")
    ap.add_argument("--log-file", required=True, help="Path to the log file containing incremental insertions.")
    ap.add_argument("--golden", help="Path to the book_golden.txt (entity ids per non-empty line; first column is key_column).")
    ap.add_argument("--queryjson", required=True, help="Path to query_truth.json (list; index i aligns to ith non-empty line in golden).")
    ap.add_argument("--key-column", required=True, help="The column used as the primary key (e.g., ISBN).")
    args = ap.parse_args()

    # Load entity IDs and query template
    with open(args.queryjson, "r", encoding="utf-8", errors="ignore") as f:
        query_truth = json.load(f)

    # Load raw data sources and initial index
    sources = load_raw_sources(args.raw_data_dir, args.key_column)
    index = build_index(sources, args.key_column)

    # Load and group logs by timestamp
    grouped_logs, sorted_timestamps = load_logs(args.log_file)

    entity_ids = []
    for i in query_truth:
        entity_ids.append(i["subquery"][0]["condition_value"])  

    if not check_queries_have_answers(entity_ids, index):
        print("[WARN] Initial data does not have answers for all queries. Proceeding anyway.")
        return
    
    # Process logs by timestamp
    for timestamp in sorted_timestamps:
        # Process all insertions for the current timestamp
        for log_entry in grouped_logs[timestamp]:
            sid = log_entry["sid"]
            data = log_entry["data"]
            # Apply the insert to the appropriate data source
            apply_insert_to_sources(sources, args.key_column, sid, data, index)

        # After processing all inserts for the current timestamp, update the query_truth
        update_query_truth(query_truth, entity_ids, index, timestamp)

        # Write the updated query_truth as a snapshot

    result_path = "query_truth_no_evaluate_flight.json"
    write_query_truth(result_path, query_truth)
    print(f"[INFO] Snapshot written to: {result_path}")

if __name__ == "__main__":
    main()


# python result_create_flight.py --raw-data-dir /home/lwh/QueryFusion/data/dataset/flight/raw_data  --log-file /home/lwh/QueryFusion/data/dataset/flight/log_data.json  --queryjson /home/lwh/QueryFusion/data/dataset/flight/query_truth.json --key-column flight