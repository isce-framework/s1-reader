import pandas as pd
from pathlib import Path
import glob
import s1reader
from shapely import wkb, wkt
import sqlite3


DEFAULT_BURST_DB_FILE = "/home/staniewi/dev/burst_map_margin4000.sqlite3"


def get_full_db():
    with sqlite3.connect(DEFAULT_BURST_DB_FILE) as con:
        con.enable_load_extension(True)
        con.load_extension("mod_spatialite")
        return pd.read_sql(
            """SELECT relative_orbit_number, burst_id_jpl,
                      AsBinary(geometry) as geometry
            FROM burst_id_map""",
            con,
        )


def make_sample_db(out_name="data/esa_burst_db_sample.csv"):
    df_db = get_full_db()
    all_bursts = []
    for f in glob.glob("data/*zip"):
        orbf = s1reader.get_orbit_file_from_dir(f, Path(f).parent / "orbits")
        for i in [1, 2, 3]:
            try:
                all_bursts.extend(s1reader.load_bursts(f, orbf, i, pol="vv", flag_apply_eap=False))
            except ValueError as e:
                print(f"Error loading {f} subswath {i}: {e}")
                continue

    all_bids = [b.burst_id for b in all_bursts]
    matching_rows = df_db.burst_id_jpl.str.contains(f"{'|'.join(all_bids)}")
    assert matching_rows.sum() == len(all_bids)

    cols = ["burst_id_jpl", "relative_orbit_number", "geometry"]
    out_df = df_db.loc[matching_rows, cols].copy()
    out_df["geometry"] = out_df.geometry.apply(lambda x: wkt.dumps(wkb.loads(x)))
    out_df.to_csv(out_name, index=False)


if __name__ == "__main__":
    make_sample_db()
