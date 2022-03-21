from pathlib import Path
from collections import defaultdict
from letter_recognition import ImageToText

if __name__ == "__main__":
    image_to_text = ImageToText()
    image_to_text.setup_knn("./data/out/train/")

    test_dir = Path("./data/out/")
    test_data = defaultdict(list)

    for img_path in sorted(test_dir.glob("*.png")):
        image_to_text.set_image(img_path)

        print(f"File: {img_path.name}, Result: {image_to_text.image_to_text()}")
