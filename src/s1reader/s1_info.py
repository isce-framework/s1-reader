"""Extract the burst ID information from a Sentinel-1 SLC product."""
import argparse
import shutil
import sys
import tempfile
import warnings
import zipfile
from itertools import chain
from pathlib import Path
from typing import List, Optional, Union

import lxml.etree as ET
import shapely.geometry
import shapely.ops

import s1reader


def get_bursts(
    filename: Union[Path, str], pol: str = "vv", iw: Optional[int] = None
) -> List[s1reader.Sentinel1BurstSlc]:
    if iw is not None:
        iws = [iw]
    else:
        iws = [1, 2, 3]
    burst_nested_list = [
        s1reader.load_bursts(filename, None, iw, pol, flag_apply_eap=False)
        for iw in iws
    ]
    return list(chain.from_iterable(burst_nested_list))


def _is_safe_dir(path: Union[Path, str]) -> bool:
    # Rather than matching the name, we just check for the existence of the
    # manifest.safe file and annotation files
    if not (path / "manifest.safe").is_file():
        return False
    annotation_dir = path / "annotation"
    if not annotation_dir.is_dir():
        return False
    if len(list(annotation_dir.glob("*.xml"))) == 0:
        return False
    return True


def _plot_bursts(safe_path: Union[Path, str], output_dir="burst_maps") -> None:
    from s1reader.utils import plot_bursts

    orbit_dir = None
    xs, ys = 5, 10
    epsg = 4326
    d = Path(output_dir).resolve()
    d.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {d}")
    output_filename = d / safe_path.stem
    plot_bursts.burst_map(safe_path, orbit_dir, xs, ys, epsg, output_filename)


def get_frame_bounds(safe_path: Union[Path, str]) -> List[float]:
    """Get the bounding box of the frame from the union of all burst bounds.

    bounding box format is [lonmin, latmin, lonmax, latmax]

    Notes
    -----
    Will use the preview/map-overlay.kml file if it exists, otherwise will
    use the union of all burst bounds (which is slower).

    Parameters
    ----------
    safe_path : Union[Path, str]
        Path to the SAFE directory or zip file.

    Returns
    -------
    List[float]
        [lonmin, latmin, lonmax, latmax]
    """
    try:
        return _bounds_from_preview(safe_path)
    except Exception as e:
        warnings.warn(f"Could not get bounds for {safe_path}: {e}")
    return _bounds_from_bursts(safe_path)


def _bounds_from_preview(safe_path: Union[Path, str]) -> List[float]:
    """Get the bounding box of the frame from the preview/map-overlay.kml."""
    # looking for:
    # S1A_IW_SLC__1SDV_20221005T125539_20221005T125606_045307_056AA5_CB45.SAFE/preview/map-overlay.kml
    if _is_safe_dir(safe_path):
        overlay_path = Path(safe_path) / "preview" / "map-overlay.kml"
        temp_dir = None
    else:
        # The name of the unzipped .SAFE directory (with .zip stripped)
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(safe_path, "r") as zip_ref:
            zname = [
                zi
                for zi in zip_ref.infolist()
                if "preview/map-overlay.kml" in zi.filename
            ]
            overlay_path = Path(zip_ref.extract(zname[0], path=temp_dir))

    # Check that they have all the necessary kmls
    if not overlay_path.exists():
        raise ValueError(f"{overlay_path} does not exist.")
    root = ET.parse(overlay_path).getroot()

    # point_str looks like:
    # <coordinates>-102.552971,31.482372 -105.191353,31.887299...
    point_str = list(elem.text for elem in root.iter("coordinates"))[0]
    coords = [p.split(",") for p in point_str.split()]
    lons, lats = zip(*[(float(lon), float(lat)) for lon, lat in coords])
    if temp_dir is not None:
        shutil.rmtree(temp_dir)
    return [min(lons), min(lats), max(lons), max(lats)]


def _bounds_from_bursts(safe_path: Union[Path, str]) -> List[float]:
    """Get the bounding box of the frame from the union of all burst bounds."""
    # Get all the bursts from subswath 1, 2, 3
    bursts = get_bursts(safe_path)
    # Convert the border (list of polygons) into a MultiPolygon
    all_borders = [shapely.geometry.MultiPolygon(b.border) for b in bursts]
    # Perform a union to get one shape for the entire frame
    border_geom = shapely.ops.unary_union(all_borders)
    # grab the bounds and pad as needed
    return list(border_geom.bounds)


EXAMPLE = """
Example usage:

    # Print all bursts in a Sentinel-1 SLC product
    s1_info.py S1A_IW_SLC__1SDV_20180601T000000_20180601T000025_021873_025F3D_9E9E.zip

    # Print only the burst IDs
    s1_info.py S1A_IW_SLC__1SDV_20180601T000000_20180601T000025_021873_025F3D_9E9E.SAFE --burst-id

    # Print burst ids for all files matching the pattern
    s1_info.py -b S1A_IW_SLC__1SDV_2018*

    # Print only from subswath IW1, and "vv" polarization
    s1_info.py -b S1A_IW_SLC__1SDV_2018* --iw 1 --pol vv

    # Get info for all products in the 'data/' directory
    s1_info.py data/
 
    # Plot the burst map, saving files into the 'burst_maps/' directory
    s1_info.py S1A_IW_SLC__1SDV_20180601T000000_20180601T000025_021873_025F3D_9E9E.SAFE/ --plot
    s1_info.py S1A_IW_SLC__1SDV_20180601T000000_20180601T000025_021873_025F3D_9E9E.zip -p -o my_burst_maps
"""


def get_cli_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Extract the burst ID information from a Sentinel-1 SLC product.",
        epilog=EXAMPLE,
    )
    parser.add_argument(
        "paths",
        help="Path to the Sentinel-1 SLC product(s), or directory containing products.",
        nargs="+",
    )
    parser.add_argument(
        "--pol",
        default="vv",
        choices=["vv", "vh", "hh", "hv"],
        help="Polarization to use.",
    )
    parser.add_argument(
        "-i",
        "--iw",
        type=int,
        choices=[1, 2, 3],
        help="Print only the burst IDs for the given IW.",
    )
    parser.add_argument(
        "-b",
        "--burst-id",
        action="store_true",
        help="Print only the burst IDs for all bursts.",
    )
    parser.add_argument(
        "--bbox",
        action="store_true",
        help="Print the bounding box (lonmin, latmin, lonmax, latmax) of the S1 product.",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="Plot the burst map for all bursts.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="burst_maps",
        help=(
            "Name of the output directory for the burst maps (if plotting),"
            " with files named for each S1 product (default= %(default)s)."
        ),
    )
    return parser.parse_args()


def main():
    args = get_cli_args()
    paths = [Path(p) for p in args.paths]
    all_files = []
    for path in paths:
        if path.is_file() or _is_safe_dir(path):
            all_files.append(path)
        elif path.is_dir():
            # Get all matching files within the directory
            files = path.glob("S1[AB]_IW*")
            all_files.extend(list(sorted(files)))
        else:
            warnings.warn(f"{path} is not a file or directory. Skipping.")

    print(f"Found {len(all_files)} Sentinel-1 SLC products.", file=sys.stderr)
    for path in all_files:
        if args.plot:
            _plot_bursts(path)
            continue
        elif args.bbox:
            msg = f"{path}: {get_frame_bounds(path)}"
            print(msg)
            continue

        print(f"Bursts in {path}:")
        print("-" * 80)
        # Do we want to pretty-print this with rich?
        for burst in get_bursts(path, args.pol, args.iw):
            if args.burst_id:
                print(burst.burst_id)
            else:
                print(burst)


if __name__ == "__main__":
    main()
