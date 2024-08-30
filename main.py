import os
import wandb
import argparse
from libraries import utils
from dotenv import load_dotenv
from libraries.MLExperiment import MLExperiment
from pathlib import Path


def load_config(path: str) -> dict:
    """
    Load a configuration file path in YAML format.

    Args:
        path (str): Path to the YAML configuration file.
    """
    return utils.load_yaml(path)


def parse_arguments():
    """
    Parse command-line arguments.
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run ML experiment with the specified configuration file."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    return parser.parse_args()


def run_ML_experiment(config, data_dir, saved_objects_dir):
    mlExperiment = MLExperiment(config, data_dir, saved_objects_dir)
    mlExperiment.run_full_experiment()


if __name__ == "__main__":
    """ Usage example
    python main.py --config experiment.yaml
    """
    # Load .env file
    load_dotenv()

    # Accessing variables
    project_name = os.getenv("PROJECT_NAME")
    wandb_dir = os.getenv("WANDB_DIR")
    data_dir = os.getenv("DATA_FOLDER")
    saved_objects_dir = os.getenv("SAVED_OBJECTS_FOLDER")

    # Parse command-line arguments
    args = parse_arguments()
    yaml_folder = "configs"
    yaml_config_file = Path(args.config)
    print("yaml_folder",yaml_folder)
    yaml_path = Path(yaml_folder) / yaml_config_file
    config = load_config(yaml_path)

    wandb.init(
        project=project_name,
        dir=wandb_dir,
        mode="disabled",
        config=config,
    )

    config = wandb.config
    run_ML_experiment(config, data_dir, saved_objects_dir)
    # print("The experiment would run here!")
