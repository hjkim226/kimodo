import json
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--scenario",
    type=str,
    required=True,
    help="Path to scenario json file"
)

args = parser.parse_args()

SCENARIO_FILE = args.scenario

RFGEN_OUTPUT = "/workspace/projects/RF-Genesis/output"

DEFAULT_ENV = "a kitchen"

KIMODO_MODEL = "Kimodo-SMPLX-RP-v1"
KIMODO_DURATION = "5.0"

ADAPTER_SCRIPT = "kimodo_smplx_to_rfgen_smpl.py"

if not os.path.exists(SCENARIO_FILE):
    print(f"Scenario file not found: {SCENARIO_FILE}")
    exit(1)

with open(SCENARIO_FILE, "r") as f:
    scenarios = json.load(f)

print(f"Loaded {len(scenarios)} scenarios")

for idx, s in enumerate(scenarios):

    name = s.get("name")
    desc = s.get("desc")
    env = s.get("env", DEFAULT_ENV)

    if not name or not desc:
        continue

    print("\n" + "=" * 50)
    print(f"[{idx+1}/{len(scenarios)}] {name}")

    output_dir = os.path.join(RFGEN_OUTPUT, name)

    obj_diff_path = os.path.join(output_dir, "obj_diff.npz")

    # already exists
    if os.path.exists(obj_diff_path):
        print(f"Already exists: {obj_diff_path}")
        continue

    os.makedirs(output_dir, exist_ok=True)

    # save prompt
    prompt_txt = os.path.join(output_dir, "prompt.txt")

    with open(prompt_txt, "w") as f:
        f.write(f"name: {name}\n")
        f.write(f"desc: {desc}\n")
        f.write(f"env: {env}\n")

    print("Step 1: Kimodo generation")

    cmd1 = [
        "kimodo_gen",
        desc,
        "--model", KIMODO_MODEL,
        "--duration", KIMODO_DURATION,
        "--output", name
    ]

    result1 = subprocess.run(cmd1)

    if result1.returncode != 0:
        print(f"Kimodo FAILED: {name}")
        continue

    amass_output = f"{name}_amass.npz"

    if not os.path.exists(amass_output):
        print(f"Missing output: {amass_output}")
        continue

    print("Step 2: Conversion into RFGen Format")

    cmd2 = [
        "python",
        ADAPTER_SCRIPT,
        amass_output,
        obj_diff_path,
        "--face-sensor"
    ]

    result2 = subprocess.run(cmd2)

    if result2.returncode != 0:
        print(f"Conversion FAILED: {name}")
        continue

    print(f"SUCCESS: {name}")

print("\nPhase 1 completed.")