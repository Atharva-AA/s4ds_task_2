"""
api_schema_hash.py — compute a SHA-256 hash of the backend's OpenAPI schema.

Writes the hash to api_schema_hash.txt.  CI compares this hash across runs
to detect API changes that require a frontend redeploy.

Usage:
    cd backend && python ../scripts/api_schema_hash.py
"""

import hashlib
import json
import sys
import os

# Add backend to path so we can import the FastAPI app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Set dummy env vars if not present (schema extraction doesn't need real keys)
os.environ.setdefault("GROQ_API_KEY", "dummy-for-schema-extraction")


def main():
    from main import app

    # FastAPI generates OpenAPI schema from routes + Pydantic models
    schema = app.openapi()

    # Remove non-deterministic fields
    schema.pop("info", None)

    # Serialise deterministically and hash
    schema_str = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    schema_hash = hashlib.sha256(schema_str.encode()).hexdigest()

    print(f"API Schema Hash: {schema_hash}")

    # Write hash file
    out_path = os.path.join("api_schema_hash.txt")
    with open(out_path, "w") as f:
        f.write(schema_hash)

    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
