"""Default config generator for creating template YAML files."""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.configs import Config


def create_default_config(output_path: Path) -> None:
    """Create a default config YAML file."""
    config = Config()
    config.outputs.root_dir = Path("outputs")
    config.to_yaml(output_path)
    print(f"Default config saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate default config template")
    parser.add_argument(
        "--output",
        type=str,
        default="configs/config_default.yaml",
        help="Output path for config YAML",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_default_config(output_path)
