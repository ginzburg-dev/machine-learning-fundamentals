import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

@dataclass
class Point2d:
    x: int
    y: int

IMAGES = [
    "pytorch_experiments/03_block_match/examples/ball.0001.jpg",
    "pytorch_experiments/03_block_match/examples/ball.0002.jpg",
    "pytorch_experiments/03_block_match/examples/ball.0003.jpg",
]

def load_images() -> list[np.ndarray]:
    """Load images."""
    images = []
    for image in IMAGES:
        images.append(plt.imread(image))
    return images

def transform_ssd(radius: int, image: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    for y in range(h-1):


def search(x: int, y: int, radius: int, image: np.ndarray) -> Point2d:
    h, w, c = image.shape
    
    return Point2d(0, 0)


def test_padding_
    #roll and pad and max pool

def main() -> None:
    images = load_images()
    print(images[0].shape)
    plt.imshow(images[0])
    plt.show()

if __name__ == "__main__":
    main()
