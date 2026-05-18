import json
from pathlib import Path

base_path = Path("knowledge_base/clues.json")
additions_path = Path("knowledge_base/additions.json")
output_path = Path("knowledge_base/clues.json")

with open(base_path) as f:
    base = json.load(f)

with open(additions_path) as f:
    additions = json.load(f)

existing_ids = {c["id"] for c in base}
new_clues = [c for c in additions if c["id"] not in existing_ids]

merged = base + new_clues
print(f"Base: {len(base)} clues")
print(f"New: {len(new_clues)} clues")
print(f"Total: {len(merged)} clues")

with open(output_path, "w") as f:
    json.dump(merged, f, indent=2)

print("Saved to clues.json")