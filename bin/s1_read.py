import os
import sys

from sentinel1_reader import sentinel1_reader, sentinel1_orbit_reader

if __name__ == "__main__":
    # TODO replace with argparse
    zip_path = sys.argv[1]
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"{zip_path} does not exist")

    i_subswath = int(sys.argv[2])
    if i_subswath < 1  or i_subswath > 3:
        raise ValueError("i_subswath not <1 or >3")

    pol = sys.argv[3]
    pols = ['vv', 'vh', 'hh', 'hv']
    if pol not in pols:
        raise ValueError("polarization not in {pols}")

    orbit_dir = sys.argv[4]
    if not os.path.isdir(orbit_dir):
        raise NotADirectoryError(f"{orbit_dir} not found")
    orbit_path = sentinel1_orbit_reader.get_swath_orbit_file(zip_path, orbit_dir)

    bursts = sentinel1_reader.zip2bursts(zip_path, orbit_path, i_subswath, pol)

    for i, burst in enumerate(bursts):
        print(burst.burst_id, burst.center)
