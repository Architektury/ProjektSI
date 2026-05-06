# A program for reading and displaying handwritten words downloaded from graphic 
# files based on descriptions from text files
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from pathlib import Path
import numpy as np

num_of_authors = 8

# Use script directory as base so relative paths work regardless of CWD
base_dir = Path(__file__).resolve().parent

for author_no in range(num_of_authors):
    desc_path = base_dir / f"author{author_no + 1}" / "word_places.txt"
    if not desc_path.exists():
        # skip missing descriptor files
        continue

    with desc_path.open("r", encoding="latin-1", errors="ignore") as file_desc_ptr:
        lines = file_desc_ptr.read().splitlines()

    image_file_name_prev = None
    for row in lines:
        if not row:
            continue
        parts = row.split()
        if not parts:
            continue
        # skip comment lines that start with '%'
        if parts[0].startswith('%'):
            continue

        # first token is image name in quotes, remove surrounding quotes if present
        img_token = parts[0]
        if img_token.startswith('"') and img_token.endswith('"'):
            img_name = img_token[1:-1]
        else:
            img_name = img_token

        image_path = base_dir / f"author{author_no + 1}" / img_name
        # avoid re-displaying the same image multiple times
        if image_path == image_file_name_prev:
            continue

        if not image_path.exists():
            # missing image file, skip
            continue

        try:
            image = mpimg.imread(str(image_path))
        except Exception:
            # if image can't be read, skip
            continue

        plt.title(f"Author {author_no+1}, image = {img_name}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.imshow(image)
        plt.show()
        image_file_name_prev = image_path