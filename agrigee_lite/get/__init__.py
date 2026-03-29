from agrigee_lite.get.image import download_multiple_images as images
from agrigee_lite.get.image import download_multiple_images_async as async_images
from agrigee_lite.get.image import download_single_image as image
from agrigee_lite.get.sits import download_multiple_sits as multiple_sits
from agrigee_lite.get.sits import download_multiple_sits_async as async_multiple_sits
from agrigee_lite.get.sits import download_multiple_sits_chunks_gcs as multiple_sits_gcs
from agrigee_lite.get.sits import download_multiple_sits_chunks_gdrive as multiple_sits_gdrive
from agrigee_lite.get.sits import download_single_sits as sits

__all__ = [
    # sync
    "image",
    "images",
    "multiple_sits",
    "multiple_sits_gcs",
    "multiple_sits_gdrive",
    "sits",
    # async (only functions that use aria2 have an async variant)
    "async_images",
    "async_multiple_sits",
]
