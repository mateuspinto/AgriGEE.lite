import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely import Polygon

from agrigee_lite.get.image import download_multiple_images
from agrigee_lite.sat.abstract_satellite import OpticalSatellite


def visualize_multiple_images(
    geometry: Polygon,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: OpticalSatellite,
    invalid_images_threshold: float = 0.5,
    contrast: float = 1.3,
    num_threads_rush: int = 30,
    num_threads_retry: int = 10,
) -> None:
    images, image_names = download_multiple_images(
        geometry, start_date, end_date, satellite, invalid_images_threshold, num_threads_rush, num_threads_retry
    )

    images = np.clip(images * contrast, 0, 1)

    images_per_row = 10
    num_rows = (len(image_names) // images_per_row) + (len(image_names) % images_per_row > 0)

    fig, ax = plt.subplots(
        num_rows, images_per_row, figsize=(25, (len(image_names) // images_per_row) * 5), sharex=True, sharey=True
    )
    for i in range(num_rows * images_per_row):
        if i < len(image_names):
            name = image_names[i]
            image = images[i]

            ax[i // images_per_row, i % images_per_row].imshow(image)
            ax[i // images_per_row, i % images_per_row].set_title(name)
            ax[i // images_per_row, i % images_per_row].axis("off")
        else:
            fig.delaxes(ax[i // images_per_row, i % images_per_row])

    plt.tight_layout()
    plt.show()
