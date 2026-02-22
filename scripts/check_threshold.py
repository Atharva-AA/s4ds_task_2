"""
check_threshold.py — CI gate script.

Reads evaluation_results.json and fails (exit 1) if the overall
accuracy score is below the minimum threshold (default: 0.70).

Usage:
    python scripts/check_threshold.py backend/evaluation_results.json
    python scripts/check_threshold.py backend/evaluation_results.json --threshold 0.80
"""

import json
import sys

MINIMUM_SCORE = 0.70


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_threshold.py <evaluation_results.json> [--threshold N]")
        sys.exit(2)

    results_path = sys.argv[1]

    # Optional: override threshold via --threshold flag
    threshold = MINIMUM_SCORE
    if "--threshold" in sys.argv:
        idx = sys.argv.index("--threshold")
        threshold = float(sys.argv[idx + 1])

    # Read evaluation results
    try:
        with open(results_path, "r") as f:
            report = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {results_path} not found.")
        sys.exit(2)

    score = report.get("score", 0.0)
    total = report.get("total", 0)
    passed = report.get("passed", 0)
    failed = report.get("failed", 0)

    print(f"Evaluation Results:")
    print(f"  Total:     {total}")
    print(f"  Passed:    {passed}")
    print(f"  Failed:    {failed}")
    print(f"  Score:     {score:.4f}")
    print(f"  Threshold: {threshold:.4f}")

    if total == 0:
        print("WARNING: No evaluation cases were run — passing by default.")
        sys.exit(0)

    if score >= threshold:
        print(f"PASS: Score {score:.4f} >= {threshold:.4f}")
        sys.exit(0)
    else:
        print(f"FAIL: Score {score:.4f} < {threshold:.4f} — blocking deployment.")
        sys.exit(1)


if __name__ == "__main__":
    main()
