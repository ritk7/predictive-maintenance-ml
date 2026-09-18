"""Send one real prediction request to a running API.

Builds the payload from actual test-set sensor readings (engine 24, whose
true RUL at its final cycle is 20) and prints the request summary, the full
response, and the ground truth — a self-contained demo of the API.

Usage:  make predict          (API must already be running)
"""
import json
import sys
import urllib.error
import urllib.request

import pandas as pd

from src.pipeline import config

API_URL = "http://127.0.0.1:8000/predict"
DEMO_UNIT = 24
HISTORY_CYCLES = 15


def build_payload(unit=DEMO_UNIT, n_cycles=HISTORY_CYCLES):
    df = pd.read_csv(config.TEST_FILE, sep=r"\s+", header=None, names=config.ALL_COLS)
    sub = df[df["unit"] == unit].tail(n_cycles)
    readings = []
    for _, r in sub.iterrows():
        reading = {"cycle": int(r["cycle"])}
        for i in (1, 2, 3):
            reading[f"setting_{i}"] = float(r[f"setting_{i}"])
        for i in range(1, 22):
            reading[f"sensor_{i}"] = float(r[f"sensor_{i}"])
        readings.append(reading)
    return {"unit_id": f"engine-{unit}", "readings": readings}


def true_rul(unit=DEMO_UNIT):
    rul = pd.read_csv(config.RUL_FILE, sep=r"\s+", header=None, names=["final_RUL"])
    return int(rul.iloc[unit - 1, 0])


def main():
    payload = build_payload()
    print(f"POST {API_URL}")
    print(f"  unit_id       : {payload['unit_id']}")
    print(f"  cycles sent   : {len(payload['readings'])} "
          f"(cycles {payload['readings'][0]['cycle']}..{payload['readings'][-1]['cycle']})")
    print(f"  minimum needed: {config.MIN_HISTORY_CYCLES}\n")

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.load(resp)
            status = resp.status
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}:\n{json.dumps(json.load(e), indent=2)}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Could not reach the API ({e.reason}). Start it with: make api")
        sys.exit(1)

    print(f"HTTP {status}")
    print(json.dumps(body, indent=2))
    print(f"\nGround truth: engine {DEMO_UNIT} true RUL at its final cycle = {true_rul()}")
    print(f"Model predicted {body['predicted_rul']} "
          f"(risk={body['risk_level']}, flagged={body['needs_maintenance']})")


if __name__ == "__main__":
    main()
