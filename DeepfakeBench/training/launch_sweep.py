import wandb
import yaml
import sys
import pprint

# --- Configuration ---
# IMPORTANT: Set your W&B project name here.
# This is the project where the sweep will be created.
WANDB_PROJECT = "my-project"


def main():
    """
    Reads a sweep configuration from a YAML file, creates the W&B sweep,
    and starts an agent to run the sweep.
    """
    # --- 1. Get YAML config path from command line ---
    if len(sys.argv) > 1:
        sweep_config_path = sys.argv[1]
        print(f"Loading sweep configuration from: {sweep_config_path}")
    else:
        print("Error: Please provide the path to your sweep YAML file.")
        print(f"Usage: python {sys.argv[0]} <path_to_config.yaml>")
        sys.exit(1)

    # --- 2. Load and parse the YAML file ---
    try:
        with open(sweep_config_path, 'r') as f:
            sweep_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: The file '{sweep_config_path}' was not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    print("\n--- Successfully loaded configuration: ---")
    pprint.pprint(sweep_config)
    print("-----------------------------------------\n")

    # --- 3. Initialize the Sweep ---
    try:
        print(f"Creating sweep in project '{WANDB_PROJECT}'...")
        sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT)
        print(f"\nSweep created successfully! Sweep ID: {sweep_id}")
        print(f"View the sweep at: https://wandb.ai/{WANDB_PROJECT}/sweeps/{sweep_id}")

    except Exception as e:
        print(f"\nAn error occurred while creating the sweep: {e}")
        print("Please check that your W&B project is correctly set and you are logged in.")
        sys.exit(1)


if __name__ == '__main__':
    main()
