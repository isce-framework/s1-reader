import sys

from sentinel1_reader import sentinel1_reader

if __name__ == "__main__":
    # TODO replace with argparse
    zip_path = sys.argv[1]
    i_subswath = int(sys.argv[2])
    if i_subswath < 1  or i_subswath > 3:
        raise ValueError("i_subswath not <1 or >3")
    pol = sys.argv[3]
    if pol not in ['vv', 'vh']:
        raise ValueError("polarization not 'vv' or 'vh'")
    bursts = sentinel1_reader.zip2bursts(zip_path, i_subswath, pol)
    for i, burst in enumerate(bursts):
        print(burst.id_str, burst.center)
