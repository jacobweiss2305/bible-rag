import logging

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_text_file(url, local_filename):
    try:
        logger.info(f"Starting download from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        logger.info("Download successful, writing to file")

        with open(local_filename, "w", encoding="utf-8") as file:
            file.write(response.text)

        logger.info(f"File successfully downloaded as {local_filename}")
    except requests.RequestException as e:
        logger.error(f"An error occurred while downloading the file from {url}: {e}")
