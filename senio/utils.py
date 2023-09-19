import logging
import os
import pathlib
import sys
import xml.etree.ElementTree as et
import zipfile

import numpy as np


def __version__() -> str:
    """Returns the current version."""
    return "1.0"


def setup_logger(log_file: str, log_to_file=False) -> logging.RootLogger:
    """Creates the logger and saves it in the Logs folder (optional)."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%d-%m-%Y - %H:%M:%S")

    if log_to_file:
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(scripts_dir, "..", "Logs")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file_path = os.path.join(log_dir, log_file)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger(log_file="logs.log", log_to_file=True)

def check_path_validity(product_path: pathlib.Path) -> pathlib.Path:
    """Checks if the product path is valid. If compressed, it unzip's it in the parent folder."""
    if product_path.exists() is False:
        raise SystemExit(f"File does not exist: {product_path}")

    if product_path.suffix not in (".zip", ".SAFE"):
        raise SystemExit(f"File does not have the correct suffix")

    if product_path.suffix == ".zip":
        logger.info(f"Zip-file detected: {product_path}")
        with zipfile.ZipFile(product_path, "r") as zfile:
            zfile.extractall(path=product_path.parent)
            logger.info(f"Extracted in {product_path.parent}")
            product_path = product_path.with_suffix("")
            product_path = product_path.with_suffix(".SAFE")
    else:
        logger.info(f"SAFE-file detected: {product_path}")

    return product_path


def calculate_statistics(stack_array: np.ndarray) -> list:
    """Calculates the (min, max, mean, std) for each band in the array."""
    statistics = [
        {
            "min": np.nanmin(band.flatten()),
            "max": np.nanmax(band.flatten()),
            "mean": np.nanmean(band.flatten()),
            "std": np.nanstd(band.flatten()),
        }
        for band in stack_array
    ]

    return statistics


def stack_sort(stack_array: np.ndarray, code_list: list, sorted_list: list) -> list:
    """Sorts the array given a prior, and a reference list."""
    band, row, column = stack_array.shape
    stack_sorted = np.zeros((band, row, column), dtype=np.uint16)

    len_list_bands = len(code_list)
    c_arr = np.zeros((len_list_bands), dtype=np.uint8)

    count = 0
    count_sort = 0

    while count_sort != len_list_bands:
        if code_list[count] == sorted_list[count_sort]:
            c_arr[count_sort] = count
            count_sort = count_sort + 1
            count = 0
        else:
            count = count + 1

    logger.info(f"Sorted input list: {sorted_list}")

    for sorted_band in range(0, len_list_bands):
        stack_sorted[sorted_band, :, :] = stack_array[c_arr[sorted_band], :, :]

    return stack_sorted


def create_pam_xml(stack_array: np.ndarray, out_name: pathlib.Path) -> None:
    """Computes the image statistics, and saves them in an XML file (GDAL PAMDataset)."""
    bands = [f"{band}" for band in range(1, stack_array.shape[0] + 1)]
    stats = calculate_statistics(stack_array)

    pam_dataset = et.Element("PAMDataset")
    for band in bands:
        pam_array_band = et.SubElement(pam_dataset, "PAMRasterBand", band=band)
        metadata = et.SubElement(pam_array_band, "Metadata")
        et.SubElement(metadata, "MDI", key="STATISTICS_MINIMUM").text = str(stats[int(band) - 1]["min"])
        et.SubElement(metadata, "MDI", key="STATISTICS_MAXIMUM").text = str(stats[int(band) - 1]["max"])
        et.SubElement(metadata, "MDI", key="STATISTICS_MEAN").text = str(stats[int(band) - 1]["mean"])
        et.SubElement(metadata, "MDI", key="STATISTICS_STDDEV").text = str(stats[int(band) - 1]["std"])

    tree = et.ElementTree(pam_dataset)

    if sys.version_info[1] >= 9:
        et.indent(tree, space="\t", level=0)
        
    tree.write(out_name.with_suffix(".tif.aux.xml").as_posix())
