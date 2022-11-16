"""Extract the burst ID information from a Sentinel-1 SLC product."""
import argparse
from itertools import chain
from pathlib import Path
from typing import Dict, Optional, Union
import warnings

from s1reader import load_bursts


def get_bursts(
    filename: Union[Path, str], pol: str = "vv", iw: Optional[int] = None
) -> Dict[int, list]:

    if iw is not None:
        iws = [iw]
    else:
        iws = [1, 2, 3]
    burst_nested_list = [
        load_bursts(filename, None, iw, pol, flag_apply_eap=False) for iw in iws
    ]
    return list(chain.from_iterable(burst_nested_list))


def get_cli_args():
    parser = argparse.ArgumentParser(
        description="Extract the burst ID information from a Sentinel-1 SLC product."
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
        "-p",
        "--plot",
        action="store_true",
        help="Plot the burst map for all bursts.",
    )
    parser.add_argument(
        "-o",
        "--output-filename",
        help=(
            "Name of the output file for the burst map."
            " Defaults to 'burst_map_<SAFE_ID>.gpkg'."
        ),
    )
    return parser.parse_args()


def _plot_bursts(safe_path, output_filename=None):
    from s1reader.utils import plot_bursts

    orbit_dir = None
    xs, ys = 5, 10
    epsg = 4326
    if not output_filename:
        output_filename = "burst_map_{}.gpkg".format(safe_path.stem)
        print(f"Output filename: {output_filename}")

    plot_bursts.burst_map(safe_path, orbit_dir, xs, ys, epsg, output_filename)


def main():
    args = get_cli_args()
    paths = [Path(p) for p in args.paths]
    all_files = []
    for path in paths:
        if path.is_dir():
            # Get all matching files within the directory
            files = path.glob("S1[AB]_IW*")
            all_files.extend(list(sorted(files)))
        elif path.is_file():
            all_files.append(path)
        else:
            warnings.warn(f"{path} is not a file or directory. Skipping.")

    for path in all_files:
        print(f"Bursts in {path}:")
        print("-" * 80)
        # Do we want to pretty-print this with rich?
        if args.plot:
            _plot_bursts(path)
            continue
        for burst in get_bursts(path, args.pol, args.iw):
            if args.burst_id:
                print(burst.burst_id)
            else:
                print(burst)


if __name__ == "__main__":
    main()
