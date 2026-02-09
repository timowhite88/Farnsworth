#!/usr/bin/env python3
"""Scan staging folders to find useful code output."""
import json
import os
from pathlib import Path
from collections import Counter

STAGING = Path("/workspace/Farnsworth/farnsworth/staging")

approved = []
rejected = []
no_audit = []
tasks_seen = Counter()

for folder in sorted(STAGING.iterdir()):
    if not folder.is_dir() or not folder.name.startswith("auto_"):
        continue

    summary_file = folder / "SUMMARY.json"
    if not summary_file.exists():
        no_audit.append(folder.name)
        continue

    try:
        data = json.loads(summary_file.read_text())
        result = data.get("audit_result", "unknown")
        task = data.get("task", "")[:100]
        has_code = (folder / "CODE.md").exists() or (folder / "PLAN.md").exists()

        entry = {
            "folder": folder.name,
            "result": result,
            "task": task,
            "has_code": has_code,
        }

        if "APPROVE" in result.upper():
            approved.append(entry)
        else:
            rejected.append(entry)

        # Track unique tasks
        task_short = task[:60] if task else "unknown"
        tasks_seen[task_short] += 1
    except Exception as e:
        no_audit.append(f"{folder.name} (error: {e})")

print("=" * 60)
print(f"STAGING SCAN: {len(approved) + len(rejected) + len(no_audit)} total folders")
print(f"  Approved: {len(approved)}")
print(f"  Rejected/Other: {len(rejected)}")
print(f"  No summary: {len(no_audit)}")
print("=" * 60)

print("\n=== APPROVED CODE (worth reviewing) ===")
for e in approved[-30:]:  # last 30
    code_tag = " [HAS CODE]" if e["has_code"] else ""
    print(f"  {e['folder']}: {e['task']}{code_tag}")

print(f"\n=== TOP REPEATED TASKS ===")
for task, count in tasks_seen.most_common(15):
    print(f"  {count}x: {task}")

# Check memory folder
mem_dir = STAGING / "memory"
if mem_dir.exists():
    print(f"\n=== MEMORY TASKS ({len(list(mem_dir.iterdir()))} files) ===")
    for f in sorted(mem_dir.iterdir())[:5]:
        content = f.read_text()[:200]
        print(f"  {f.name}: {content[:100]}...")

# Check research folder
res_dir = STAGING / "research"
if res_dir.exists():
    print(f"\n=== RESEARCH TASKS ({len(list(res_dir.iterdir()))} files) ===")
    for f in sorted(res_dir.iterdir())[:5]:
        content = f.read_text()[:200]
        print(f"  {f.name}: {content[:100]}...")

# Check MCP folder
mcp_dir = STAGING / "mcp"
if mcp_dir.exists():
    print(f"\n=== MCP TASKS ({len(list(mcp_dir.iterdir()))} files) ===")
    for f in sorted(mcp_dir.iterdir())[:5]:
        content = f.read_text()[:200]
        print(f"  {f.name}: {content[:100]}...")

# Read a few approved PLAN.md and CODE.md samples
print("\n=== SAMPLE APPROVED PLANS ===")
for e in approved[-5:]:
    folder = STAGING / e["folder"]
    plan = folder / "PLAN.md"
    code = folder / "CODE.md"
    if plan.exists():
        print(f"\n--- {e['folder']} PLAN ---")
        print(plan.read_text()[:300])
    if code.exists():
        print(f"\n--- {e['folder']} CODE ---")
        print(code.read_text()[:300])
