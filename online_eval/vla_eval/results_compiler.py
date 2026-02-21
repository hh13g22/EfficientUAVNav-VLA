import json
import sys
from collections import defaultdict

def compute_accuracy(entries):
    if not entries:
        return 0.0
    return sum(1 for e in entries if e["success"]) / len(entries) * 100

def analyse(file):
    # OPen 
    with open(file) as f:
        data = json.load(f)

    # open and load all json entries
    all_entries = list(data.values())

    # Separate by difficulty
    by_difficulty = defaultdict(list)
    for entry in all_entries:
        by_difficulty[entry["difficulty"]].append(entry)

    print(f"Trajs: {len(all_entries)}")
    print(f"Full SR: {compute_accuracy(all_entries):.2f}%\n")

    for difficulty in ["easy", "medium", "hard"]:
        # dict.get(key, default)
        entries = by_difficulty.get(difficulty, [])
        print(f"{difficulty}")
        print(f"Count: {len(entries)}")
        print(f"ccuracy: {compute_accuracy(entries):.2f}%")

if __name__ == "__main__":
    # Parse the second arg of python
    analyse(sys.argv[1])