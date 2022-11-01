import datetime
import os
import re
import zipfile
from pathlib import Path

import lxml.etree as ET
import numpy as np
import pandas as pd
import shapely.ops
import shapely.wkt
import shapely.wkb
from shapely.geometry import LinearRing, Polygon, MultiPolygon
from .s1_reader import load_bursts

# For IW mode, one burst has a duration of ~2.75 seconds and a burst
# overlap of approximately ~0.4 seconds.
# https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar/product-types-processing-levels/level-1
# Additional precision calculated from averaging the differences between
# burst sensing starts in prototyping test data
BURST_INTERVAL = 2.758277
ESA_TRACK_BURST_ID_FILE = (
    f"{os.path.dirname(os.path.realpath(__file__))}/data/sentinel1_track_burst_id.txt"
)


def get_info(annotation_path, open_method):
    track_burst_num = get_track_burst_num()
    _, tail = os.path.split(annotation_path)
    platform_id, swath_name, _, pol = [x.upper() for x in tail.split("-")[:4]]
    with open_method(annotation_path, "r") as f:
        tree = ET.parse(f)

    image_info_element = tree.find("imageAnnotation/imageInformation")
    ascending_node_time = as_datetime(image_info_element.find("ascendingNodeTime").text)

    orbit_number = int(tree.find("adsHeader/absoluteOrbitNumber").text)
    orbit_number_offset = 73 if platform_id == "S1A" else 202
    track_number = (orbit_number - orbit_number_offset) % 175 + 1

    boundary_pts = get_boundaries(tree)
    burst_list_elements = tree.find("swathTiming/burstList")

    burst_ids = []
    for i, burst_list_element in enumerate(burst_list_elements):
        # get burst timing
        sensing_time = as_datetime(burst_list_element.find("sensingTime").text)
        dt = sensing_time - ascending_node_time
        # local burst_num within one track, starting from 0
        burst_num = int((dt.seconds + dt.microseconds / 1e6) // BURST_INTERVAL)

        # convert the local burst_num to the global burst_num, starting from 1
        burst_num += track_burst_num[track_number][0]

        burst_id = f"t{track_number:03d}_{burst_num:06d}_{swath_name.lower()}"
        burst_ids.append(burst_id)
    return burst_ids, boundary_pts


def run(zip_list, pol="vv", out_name="sentinel1_boundaries.csv"):
    processed_safes = set()
    pat = re.compile(f"/s1[ab]-iw[123]-slc-{pol}-.*\.xml")
    out_dict = {"burst_id": [], "boundary_wkt": [], "ann_path": []}
    for zip_path in zip_list:
        try:
            z = zipfile.ZipFile(zip_path)
            z.close()
        except zipfile.BadZipFile:
            continue

        zip_path = os.path.abspath(zip_path)
        if not os.access(zip_path, os.R_OK):
            print(f"Cannot read {zip_path}")
            continue
        product_name = os.path.basename(zip_path).split(".")[0]
        if product_name in processed_safes:
            print(f"Skipping {product_name} as it has already been processed")
            continue
        if "IW" not in product_name:
            print(f"Skipping {product_name} as it is not IW mode")
            continue

        processed_safes.add(product_name)

        with zipfile.ZipFile(zip_path, "r") as z_file:
            annotation_files = list(
                sorted(f for f in z_file.namelist() if pol in f and re.search(pat, f))
            )
            assert len(annotation_files) == 3
            for ann_path in annotation_files:
                bids, bounds = get_info(ann_path, z_file.open)
                out_dict["burst_id"].extend(bids)
                out_dict["boundary_wkt"].extend([shapely.wkt.dumps(b) for b in bounds])

                p = Path(zip_path) / "/".join(Path(ann_path).parts[1:])
                out_dict["ann_path"].extend([str(p)] * len(bids))

    df = pd.DataFrame(out_dict)
    df.to_csv(out_name, index=False)
    df["boundary"] = df["boundary_wkt"].apply(shapely.wkt.loads)
    return df, processed_safes


def run2(zip_list, pol="vv", out_name="sentinel1_boundaries_s1reader.csv"):
    processed_safes = set()
    out_dict = {"burst_id": [], "boundary_wkt": [], "zip_path": []}
    for zip_path in zip_list:
        try:
            z = zipfile.ZipFile(zip_path)
            z.close()
        except zipfile.BadZipFile:
            continue

        zip_path = os.path.abspath(zip_path)
        if not os.access(zip_path, os.R_OK):
            print(f"Cannot read {zip_path}")
            continue
        product_name = os.path.basename(zip_path).split(".")[0]
        if product_name in processed_safes:
            print(f"Skipping {product_name} as it has already been processed")
            continue
        if "IW" not in product_name:
            print(f"Skipping {product_name} as it is not IW mode")
            continue
        processed_safes.add(product_name)
        bursts = load_bursts(zip_path, pol=pol, flag_apply_eap=False)
        for iw_bs in bursts:
            for b in iw_bs:
                out_dict["burst_id"].append(b.burst_id)
                out_dict["boundary_wkt"].append(
                    shapely.wkt.dumps(MultiPolygon(b.border))
                )
                out_dict["zip_path"].append(zip_path)
    df = pd.DataFrame(out_dict)
    df.to_csv(out_name, index=False)
    df["boundary"] = df["boundary_wkt"].apply(shapely.wkt.loads)
    return df, processed_safes


def run_asfsmd(safe_list, pol="vv", out_name="sentinel1_boundaries_asfsmd.csv"):
    processed_safes = set()
    out_dict = {"burst_id": [], "boundary_wkt": [], "safe_path": []}
    for safe_path in safe_list:
        safe_path = os.path.abspath(safe_path)
        product_name = os.path.basename(safe_path).split(".")[0]
        if product_name in processed_safes:
            print(f"Skipping {product_name} as it has already been processed")
            continue
        try:
            bursts = [
                load_bursts(
                    safe_path,
                    orbit_path=None,
                    swath_num=i,
                    pol=pol,
                    flag_apply_eap=False,
                )
                for i in [1, 2, 3]
            ]
        except Exception as e:
            print(f"Failed to load {safe_path}: {e}")
            continue

        processed_safes.add(product_name)
        for iw_bs in bursts:
            for b in iw_bs:
                out_dict["burst_id"].append(b.burst_id)
                out_dict["boundary_wkt"].append(
                    shapely.wkt.dumps(MultiPolygon(b.border))
                )
                out_dict["safe_path"].append(safe_path)
    df = pd.DataFrame(out_dict)
    df = df.sort_values("burst_id").reset_index(drop=True)
    if out_name:
        df.to_csv(out_name, index=False)
    df["boundary"] = df["boundary_wkt"].apply(shapely.wkt.loads)
    return df


def run_asfsmd_parallel(safe_list, pol="vv", out_name="sentinel1_boundaries_asfsmd.csv"):
    # out_dict = {"burst_id": [], "boundary_wkt": [], "safe_path": []}
    from concurrent.futures import ProcessPoolExecutor, as_completed
    all_results = []
    with ProcessPoolExecutor(max_workers=10) as exc:
        futures = [
            exc.submit(_run_safe, safe_name, pol) for safe_name in safe_list
        ]
        for fut in as_completed(futures):
            all_results.extend(fut.result())
    
    df = pd.DataFrame(all_results, columns=["burst_id", "boundary_wkt", "safe_path"])
    df = df.sort_values("burst_id").reset_index(drop=True)
    if out_name:
        df.to_csv(out_name, index=False)
    df["boundary"] = df["boundary_wkt"].apply(shapely.wkt.loads)
    return df


def _run_safe(safe_path, pol="vv"):
    safe_path = os.path.abspath(safe_path)
    try:
        bursts = [
            load_bursts(
                safe_path,
                orbit_path=None,
                swath_num=i,
                pol=pol,
                flag_apply_eap=False,
            )
            for i in [1, 2, 3]
        ]
    except Exception as e:
        print(f"Failed to load {safe_path}: {e}")
        return []

    out = []
    for iw_bs in bursts:
        for b in iw_bs:
            out.append((
                b.burst_id, 
                shapely.wkt.dumps(MultiPolygon(b.border)),
                safe_path
            ))
    return out

def as_datetime(t_str):
    """Parse given time string to datetime.datetime object.

    Parameters:
    ----------
    t_str : string
        Time string to be parsed. (e.g., "2021-12-10T12:00:0.0")
    fmt : string
        Format of string provided. Defaults to az time format found in annotation XML.
        (e.g., "%Y-%m-%dT%H:%M:%S.%f").

    Returns:
    ------
    _ : datetime.datetime
        datetime.datetime object parsed from given time string.
    """
    return datetime.datetime.fromisoformat(t_str)


def get_boundaries(tree):
    """Parse grid points list and calculate burst center lat and lon

    Parameters:
    -----------
    tree : Element
        Element containing geolocation grid points.

    Returns:
    --------
    center_pts : list
        List of burst centroids ass shapely Points
    boundary_pts : list
        List of burst boundaries as shapely Polygons
    """
    # find element tree
    grid_pt_list = tree.find("geolocationGrid/geolocationGridPointList")

    # read in all points
    n_grid_pts = int(grid_pt_list.attrib["count"])
    lines = np.empty(n_grid_pts)
    pixels = np.empty(n_grid_pts)
    lats = np.empty(n_grid_pts)
    lons = np.empty(n_grid_pts)
    for i, grid_pt in enumerate(grid_pt_list):
        lines[i] = int(grid_pt[2].text)
        pixels[i] = int(grid_pt[3].text)
        lats[i] = float(grid_pt[4].text)
        lons[i] = float(grid_pt[5].text)

    unique_line_indices = np.unique(lines)
    boundary_pts = []

    # zip lines numbers of bursts together and iterate
    for i, (ln0, ln1) in enumerate(
        zip(unique_line_indices[:-1], unique_line_indices[1:])
    ):
        # create masks for lines in current burst
        mask0 = lines == ln0
        mask1 = lines == ln1

        # reverse order of 2nd set of points so plots of boundaries
        # are not connected by a diagonal line
        burst_lons = np.concatenate((lons[mask0], lons[mask1][::-1]))
        burst_lats = np.concatenate((lats[mask0], lats[mask1][::-1]))

        poly = Polygon(zip(burst_lons, burst_lats))
        boundary_pts.append(MultiPolygon(check_dateline(poly)))

    return boundary_pts


def check_dateline(poly):
    """Split `poly` if it crosses the dateline.
    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Input polygon.
    Returns
    -------
    polys : list of shapely.geometry.Polygon
         A list containing: the input polygon if it didn't cross
        the dateline, or two polygons otherwise (one on either
        side of the dateline).
    """

    xmin, _, xmax, _ = poly.bounds
    # Check dateline crossing
    if (xmax - xmin) > 180.0:
        dateline = shapely.wkt.loads("LINESTRING( 180.0 -90.0, 180.0 90.0)")

        # build new polygon with all longitudes between 0 and 360
        x, y = poly.exterior.coords.xy
        new_x = (k + (k <= 0.0) * 360 for k in x)
        new_ring = LinearRing(zip(new_x, y))

        # Split input polygon
        # (https://gis.stackexchange.com/questions/232771/splitting-polygon-by-linestring-in-geodjango_)
        merged_lines = shapely.ops.linemerge([dateline, new_ring])
        border_lines = shapely.ops.unary_union(merged_lines)
        decomp = shapely.ops.polygonize(border_lines)

        polys = list(decomp)
        assert len(polys) == 2
    else:
        # If dateline is not crossed, treat input poly as list
        polys = [poly]

    return polys


def get_track_burst_num(track_burst_num_file: str = ESA_TRACK_BURST_ID_FILE):
    """Read the start / stop burst number info of each track from ESA.

    Parameters:
    -----------
    track_burst_num_file : str
        Path to the track burst number files.

    Returns:
    --------
    track_burst_num : dict
        Dictionary where each key is the track number, and each value is a list
        of two integers for the start and stop burst number
    """

    # read the text file to list
    track_burst_info = np.loadtxt(track_burst_num_file, dtype=int)

    # convert lists into dict
    track_burst_num = dict()
    for track_num, burst_num0, burst_num1 in track_burst_info:
        track_burst_num[track_num] = [burst_num0, burst_num1]

    return track_burst_num


TRANSFORMERS = {}
from pyproj import Transformer


def transform(geom, src_epsg, dst_epsg):
    if (src_epsg, dst_epsg) in TRANSFORMERS:
        t = TRANSFORMERS[(src_epsg, dst_epsg)]
    else:
        t = Transformer.from_crs(
            src_epsg,
            dst_epsg,
            always_xy=True,
        )
        TRANSFORMERS[(src_epsg, dst_epsg)] = t
    return shapely.ops.transform(t.transform, geom)


import sqlite3


def compare_boundaries(df, db_path=None):
    results = []
    failed_idxs = []
    if db_path is None:
        db_path = "/home/staniewi/dev/burst_map_margin4000.sqlite3"
    query = "SELECT epsg, asbinary(geometry) FROM burst_id_map WHERE burst_id_jpl = ?"
    with sqlite3.connect(db_path) as con:
        con.enable_load_extension(True)
        con.load_extension("mod_spatialite")
        for idx, bid in enumerate(df.burst_id):
            cur = con.execute(query, (bid,))
            r = cur.fetchone()
            if r is None:
                failed_idxs.append(idx)
            else:
                results.append(r)

    bad_df = df.index.isin(failed_idxs)
    df_failed = df.loc[bad_df].copy()
    print(f"Failed to find {len(df_failed)} bursts")
    df = df.loc[~bad_df]

    epsgs, db_geoms_wkb = zip(*results)
    df["db_boundary"] = [shapely.wkb.loads(g) for g in db_geoms_wkb]
    df["epsg"] = epsgs

    df["boundary_utm"] = df[["boundary", "epsg"]].apply(
        lambda row: transform(row.boundary, 4326, row.epsg), axis=1
    )
    df["db_boundary_utm"] = df[["db_boundary", "epsg"]].apply(
        lambda row: transform(row.db_boundary, 4326, row.epsg), axis=1
    )

    # Get the differences in bounds
    # a positive means the db boundary is larger than the actual data boundary
    # negative is an underestimation, bad
    bounds_actual = np.array(df.boundary_utm.apply(lambda g: g.bounds).to_list())
    bounds_db = np.array(df.db_boundary_utm.apply(lambda g: g.bounds).tolist())
    # for xmin, ymin, we want the db to be smaller than the actual
    df["xmins"] = bounds_actual[:, 0] - bounds_db[:, 0]
    df["ymins"] = bounds_actual[:, 1] - bounds_db[:, 1]
    # for xmax, ymax, we want the db to be larger than the actual
    df["xmaxs"] = bounds_db[:, 2] - bounds_actual[:, 2]
    df["ymaxs"] = bounds_db[:, 3] - bounds_actual[:, 3]
    df = df.drop(["boundary_wkt"], axis=1, errors="ignore")
    df["iou"] = df.apply(lambda row: iou(row.boundary_utm, row.db_boundary_utm), axis=1)
    mismatches = df["iou"] < 0.60
    print(f"Found {mismatches.sum()} mismatches")
    return df, df_failed.reset_index()


def iou(geom1, geom2):
    """Calculate the intersection over union of two polygons."""
    return geom1.intersection(geom2).area / geom1.union(geom2).area
