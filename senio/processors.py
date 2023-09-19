import datetime
import pathlib
import shutil

import numpy as np
import rasterio
import rasterio.enums

from senio.utils import check_path_validity, create_pam_xml, setup_logger, stack_sort

logger = setup_logger(log_file="logs.log")

def processor_all(
    product_path: str,
    output_path: str = None,
    resample: str = "nearest",
    xml: bool = True,
    pyramids: bool = True,
) -> None:
    """
    Creates a stacked numpy array of all the available Sentinel-2 L2A bands, sorts in ascending order, resamples to 10m, and saves them in GTiff format. The user can generate the .aux.xml, and .ovr files for better handling within a GIS environment. All output files are compressed with the deflate method (lossless). Each output file will have the code name of the .SAFE path, plus a signature tag "_STACK_ALL" at the end.

    :param str product_path: The path of the Sentinel-2 product (.SAFE or .zip).
    :param str, Optional output_path: The location to output the formatted product. By *default* it creates a folder in the .SAFE directory.
    :param str, Optional resample: The sampling method to use for the 20m, and 60m bands. The available options are "nearest", "bilinear", and "cubic". By *default* it's "nearest".
    :param bool, Optional xml: Create the .aux.xml image statistics. These statistics are not extracted by an external auxiliary file, but calculated on the spot. By *default* it's True.
    :param bool, Optional pyramids: Create the .ovr pyramids file with zoom factors (2, 4, 8, 16), and "nearest" resampling method. By *default* it's True.

    :raise SystemExit: If the product path does not contain the "L2A" sequence of characters.

    :returns: None.
    """
    dt = datetime.datetime
    dt_0 = dt.now()

    path = pathlib.Path(product_path)
    safe_path = check_path_validity(path)

    if output_path is not None:
        output_folder_path = output_path
    else:
        output_folder_path = safe_path / "GTIFF_PRODUCT"
        if output_folder_path.exists():
            logger.info("Deleting files in .\GTIFF_PRODUCT folder")
            shutil.rmtree(output_folder_path)
        output_folder_path.mkdir(parents=False, exist_ok=True)
        logger.info(f"Created output folder: {output_folder_path}")

    temp_path = safe_path / "TEMP"
    temp_path.mkdir(parents=False, exist_ok=True)
    logger.info(f"Created temporary folder: {temp_path}")

    granule = safe_path / "GRANULE"
    granule = list(granule.rglob("*L2A*"))[0]

    if "L2A" in granule.name:
        safe_path_10m = granule / "IMG_DATA" / "R10m"
        safe_path_20m = granule / "IMG_DATA" / "R20m"
        safe_path_60m = granule / "IMG_DATA" / "R60m"

        band_names_10m = [file.name for file in safe_path_10m.iterdir() if file.suffix == ".jp2"]
        band_names_20m = [file.name for file in safe_path_20m.iterdir() if file.suffix == ".jp2"]
        band_names_60m = [file.name for file in safe_path_60m.iterdir() if file.suffix == ".jp2"]
    else:
        raise SystemExit(f"L2A product not found:\n{granule}")

    stack_10m_list = []
    code_10m_list = []
    stack_20m_list = []
    code_20m_list = []
    stack_60m_list = []
    code_60m_list = []

    for band in range(0, len(band_names_10m)):
        if (
            ("B02_10m" in band_names_10m[band])
            or ("B03_10m" in band_names_10m[band])
            or ("B04_10m" in band_names_10m[band])
            or ("B08_10m" in band_names_10m[band])
        ):
            logger.info(f"Processing: {band_names_10m[band]}")

            code_10m_list.append(band_names_10m[band][24:26])
            band_path_10m = safe_path_10m / band_names_10m[band]

            with rasterio.open(band_path_10m) as src_10m:
                src_10m_kwargs = src_10m.profile.copy()
                stack_10m_list.append(src_10m.read(1))

    for band in range(0, len(band_names_20m)):
        if (
            ("B05_20m" in band_names_20m[band])
            or ("B06_20m" in band_names_20m[band])
            or ("B07_20m" in band_names_20m[band])
            or ("B8A_20m" in band_names_20m[band])
            or ("B11_20m" in band_names_20m[band])
            or ("B12_20m" in band_names_20m[band])
        ):
            logger.info(f"Processing: {band_names_20m[band]}")

            code_20m_list.append(band_names_20m[band][24:26])
            band_path_20m = safe_path_20m / band_names_20m[band]

            with rasterio.open(band_path_20m) as src_20m:
                src_20m_kwargs = src_20m.profile.copy()
                stack_20m_list.append(src_20m.read(1))

    for band in range(0, len(band_names_60m)):
        if (
            ("B01_60m" in band_names_60m[band])
            or ("B09_60m" in band_names_60m[band])
            or ("B10_60m" in band_names_60m[band])
        ):
            logger.info(f"Processing: {band_names_60m[band]}")

            code_60m_list.append(band_names_60m[band][24:26])
            band_path_60m = safe_path_60m / band_names_60m[band]

            with rasterio.open(band_path_60m) as src_60m:
                src_60m_kwargs = src_60m.profile.copy()
                stack_60m_list.append(src_60m.read(1))

    logger.info(f"Sorting 10m stack")

    sorted_list_10m = ["02", "03", "04", "08"]
    stack_10m = np.asarray(stack_10m_list)
    stack_10m_sorted = stack_sort(stack_10m, code_10m_list, sorted_list_10m)
    del stack_10m, stack_10m_list

    logger.info(f"Sorting 20m stack")

    sorted_list_20m = ["05", "06", "07", "11", "12", "8A"]
    stack_20m = np.asarray(stack_20m_list)
    stack_20m_sorted = stack_sort(stack_20m, code_20m_list, sorted_list_20m)
    del stack_20m, stack_20m_list

    logger.info(f"Sorting 60m stack")

    sorted_list_60m = ["01", "09"]
    stack_60m = np.asarray(stack_60m_list)
    stack_60m_sorted = stack_sort(stack_60m, code_60m_list, sorted_list_60m)
    del stack_60m, stack_60m_list

    logger.info(f"Saving the 10m, 20m, and 60m stacks in Temp folder")

    src_10m_kwargs.update(
        {
            "driver": "GTiff",
            "compress": "deflate",
            "interleave": "band",
            "count": 4,
            "dtype": rasterio.uint16,
        }
    )

    src_20m_kwargs.update(
        {
            "driver": "GTiff",
            "compress": "deflate",
            "interleave": "band",
            "count": 6,
            "dtype": rasterio.uint16,
        }
    )

    src_60m_kwargs.update(
        {
            "driver": "GTiff",
            "compress": "deflate",
            "interleave": "band",
            "count": 2,
            "dtype": rasterio.uint16,
        }
    )

    with rasterio.Env(
        NUM_THREADS="ALL_CPUS",
        TILED=True,
        BLOCKXSIZE=1024,
        BLOCKYSIZE=1024,
        COMPRESS="DEFLATE"
    ):
        stack_temp_10m = temp_path / "Temp_10m.tif"
        with rasterio.open(stack_temp_10m, "w", **src_10m_kwargs) as dst:
            dst.write(stack_10m_sorted)

        stack_temp_20m = temp_path / "Temp_20m.tif"
        with rasterio.open(stack_temp_20m, "w", **src_20m_kwargs) as dst:
            dst.write(stack_20m_sorted)

        stack_temp_60m = temp_path / "Temp_60m.tif"
        with rasterio.open(stack_temp_60m, "w", **src_60m_kwargs) as dst:
            dst.write(stack_60m_sorted)

    logger.info(f"Resampling the 20m and 60m arrays")

    scale_factor_20m = 2
    scale_factor_60m = 6

    if resample == "nearest":
        res = rasterio.enums.Resampling.nearest
    elif resample == "bilinear":
        res = rasterio.enums.Resampling.bilinear
    elif resample == "cubic":
        res = rasterio.enums.Resampling.cubic
    else:
        raise SystemExit(f"{resample} not a valid option")

    with rasterio.open(stack_temp_20m) as res_src_20m:
        data_20m = res_src_20m.read(
            out_shape=(
                res_src_20m.count,
                int(res_src_20m.height * scale_factor_20m),
                int(res_src_20m.width * scale_factor_20m),
            ),
            resampling=res,
        )

        transform_20m = res_src_20m.transform * res_src_20m.transform.scale(
            res_src_20m.width / data_20m.shape[2],
            res_src_20m.height / data_20m.shape[1],
        )

    with rasterio.open(stack_temp_60m) as res_src_60m:
        data_60m = res_src_60m.read(
            out_shape=(
                res_src_60m.count,
                int(res_src_60m.height * scale_factor_60m),
                int(res_src_60m.width * scale_factor_60m),
            ),
            resampling=res,
        )

        transform_60m = res_src_60m.transform * res_src_60m.transform.scale(
            res_src_60m.width / data_60m.shape[2],
            res_src_60m.height / data_60m.shape[1],
        )

    if transform_20m == transform_60m:
        logger.info(f"Stacking bands into a single array")

        stack = np.concatenate(
            (
                data_60m[0:1],
                stack_10m_sorted[0:1],
                stack_10m_sorted[1:2],
                stack_10m_sorted[2:3],
                data_20m[0:1],
                data_20m[1:2],
                data_20m[2:3],
                stack_10m_sorted[3:4],
                data_20m[5:6],
                data_60m[1:2],
                data_20m[3:4],
                data_20m[4:5],
            ),
            axis=0,
        )

    del stack_10m_sorted, stack_20m_sorted, stack_60m_sorted, data_20m, data_60m

    new_kwargs = src_10m_kwargs.copy()
    new_kwargs.update(
        {
            "driver": "GTiff",
            "count": 12,
            "dtype": rasterio.uint16,
            "compress": "deflate",
            "interleave": "band",
        }
    )

    name_output = output_folder_path / pathlib.Path(safe_path.name[:-5] + "_" + resample.upper() +"_STACK_ALL.tif")

    if xml:
        create_pam_xml(stack, name_output)

    logger.info(f"Exporting: {name_output}")

    with rasterio.Env(
        TIFF_USE_OVR=True,
        GDAL_TIFF_OVR_BLOCKSIZE=1024,
        COMPRESS_OVERVIEW="DEFLATE",
        NUM_THREADS="ALL_CPUS",
        TILED=True,
        BLOCKXSIZE=1024,
        BLOCKYSIZE=1024,
        COMPRESS="DEFLATE",
    ):
        with rasterio.open(name_output, "w", **new_kwargs) as dst:
            dst.write(stack)
            if pyramids:
                logger.info(f"Building and compressing pyramids")
                factors = [2, 4, 8, 16]
                dst.build_overviews(factors, rasterio.enums.Resampling.nearest)

    shutil.rmtree(temp_path)
    dt_1 = dt.now()

    logger.info(f"Completed in {dt_1 - dt_0}\n")

def processor_rgbn(
    product_path: str,
    output_path: str = None,
    xml: bool = True,
    pyramids: bool = True
) -> None:
    """
    Creates a stacked numpy array of the four 10m Sentinel-2 L2A bands (Blue, Green, Red, NIR), sorts in ascending order, and saves them in GTiff
    format. The user can generate the .aux.xml, and .ovr files for better handling within a GIS environment. All output files are
    compressed with the deflate method (lossless). Each output file will have the code name of the .SAFE path, plus a signature tag "_STACK_RGBN"
    at the end.

    :param str product_path: The path of the Sentinel-2 product.
    :param str, Optional output_path: The location to output the formatted product. By *default* it creates a folder in the .SAFE directory.
    :param bool, Optional xml: Create the .aux.xml image statistics. These statistics are not extracted by an external auxiliary file, but calculated on the spot. By *default* it's True.
    :param bool, Optional pyramids: Create the .ovr pyramids file with zoom factors (2, 4, 8, 16), and "nearest" resampling method. By *default* it's True.
    :param bool, Optional verbose: Log each step of the function's execution. By *default* it's True.

    :raise SystemExit: If the product path does not contain the "L2A" sequence of characters.

    :returns: None.
    """
    dt = datetime.datetime
    dt_0 = dt.now()

    path = pathlib.Path(product_path)
    safe_path = check_path_validity(path)

    if output_path is not None:
        output_folder_path = output_path
    else:
        output_folder_path = safe_path / "GTIFF_PRODUCT"
        if output_folder_path.exists():
            logger.info("Deleting files in .\GTIFF_PRODUCT folder")
            shutil.rmtree(output_folder_path)
        output_folder_path.mkdir(parents=False, exist_ok=True)
        logger.info(f"Created output folder: {output_folder_path}")

    granule = safe_path / "GRANULE"
    granule = list(granule.rglob("*L2A*"))[0]

    if "L2A" in granule.name:
        safe_path_10m = granule / "IMG_DATA" / "R10m"
        band_names_10m = [file.name for file in safe_path_10m.iterdir() if file.suffix == ".jp2"]
    else:
        raise SystemExit(f"L2A product not found:\n{granule}")

    stack_10m_list = []
    code_10m_list = []

    for band in range(0, len(band_names_10m)):
        if (
            ("B02_10m" in band_names_10m[band])
            or ("B03_10m" in band_names_10m[band])
            or ("B04_10m" in band_names_10m[band])
            or ("B08_10m" in band_names_10m[band])
        ):
            logger.info(f"Processing: {band_names_10m[band]}")

            code_10m_list.append(band_names_10m[band][24:26])
            band_path_10m = safe_path_10m / band_names_10m[band]

            with rasterio.open(band_path_10m) as src_10m:
                src_10m_kwargs = src_10m.profile.copy()
                stack_10m_list.append(src_10m.read(1))

    logger.info(f"Sorting 10m stack")

    sorted_list_10m = ["02", "03", "04", "08"]
    stack_10m = np.asarray(stack_10m_list)
    stack_10m_sorted = stack_sort(stack_10m, code_10m_list, sorted_list_10m)
    del stack_10m, stack_10m_list

    new_kwargs = src_10m_kwargs.copy()
    new_kwargs.update(
        {
            "driver": "GTiff",
            "count": 4,
            "dtype": rasterio.uint16,
            "compress": "deflate",
            "interleave": "band",
        }
    )

    name_output = output_folder_path / pathlib.Path(safe_path.name[:-5] + "_STACK_RGBN.tif")

    if xml:
        create_pam_xml(stack_10m_sorted, name_output)

    logger.info(f"Exporting: {name_output}")

    with rasterio.Env(
        TIFF_USE_OVR=True,
        GDAL_TIFF_OVR_BLOCKSIZE=1024,
        COMPRESS_OVERVIEW="DEFLATE",
        NUM_THREADS="ALL_CPUS",
        TILED=True,
        BLOCKXSIZE=1024,
        BLOCKYSIZE=1024,
        COMPRESS="DEFLATE",
    ):
        with rasterio.open(name_output, "w", **new_kwargs) as dst:
            dst.write(stack_10m_sorted)
            if pyramids:
                logger.info(f"Building and compressing pyramids")
                factors = [2, 4, 8, 16]
                dst.build_overviews(factors, rasterio.enums.Resampling.nearest)

    dt_1 = dt.now()

    logger.info(f"Completed in {dt_1 - dt_0}\n")
