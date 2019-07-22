import os

from pathlib import Path

root_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# Create data dir paths
data_dir = root_dir / "data"
interim_data_dir = data_dir / "interim"
processed_data_dir = data_dir / "processed"
raw_data_dir = data_dir / "raw"
subs_dir = data_dir / "submissions"

if __name__ == "__main__":
    pass
