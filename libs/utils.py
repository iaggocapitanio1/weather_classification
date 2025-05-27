import random
import math
import shutil
from pathlib import Path
from typing import List


def get_files(directory: Path) -> List[Path]:
    """
    Return a list of all files (not directories) in `directory`.
    Raises NotADirectoryError if the path isn't a directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory.__str__()} is not a directory")
    return [p for p in directory.iterdir() if p.is_file()]

def get_dirs(directory: Path) -> List[Path]:
    """
    Return a list of all directories in `directory`.
    :param directory:
    :return:
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"{directory.__str__()} is not a directory")
    return [d for d in directory.iterdir() if d.is_dir()]





def move_percentage_to_test(data_dir: Path, keys: List[str], percentage: float = 0.2) -> None:
    """
    Moves a percentage of images from train/key to test/key based on filename keys.

    :param data_dir: Path to the base data folder containing 'train' and 'test'
    :param keys: List of keys like ['cloudy', 'rain', 'shine']
    :param percentage: Float between 0 and 1 indicating how many to move per class
    """
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'

    for key in keys:
        train_key_dir = train_dir / key
        test_key_dir = test_dir / key
        test_key_dir.mkdir(parents=True, exist_ok=True)

        if not train_key_dir.exists():
            print(f"Skipping {key}: folder not found in train/")
            continue

        images = [p for p in train_key_dir.iterdir() if p.is_file()]
        if not images:
            print(f"No images found in {train_key_dir}")
            continue

        n_to_move = math.ceil(len(images) * percentage)
        selected_images = random.sample(images, n_to_move)

        for image in selected_images:
            shutil.move(image, test_key_dir / image.name)
            print(f"Moved {image.name} -> {test_key_dir}")
