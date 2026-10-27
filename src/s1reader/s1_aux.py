"""Search and download Sentinel-1 auxiliary products (AUX_INS, AUX_ITC, ...).

ESA's SAR Mission Performance Cluster serves every auxiliary data file
generation through a public JSON API at https://sar-mpc.eu/api/v1/, with an MD5
for each file and no login. That is the source here: ASF's aux mirror stops in
2016 and carries no S1C/S1D, and CDSE needs credentials.

This is a staging tool. Nothing in s1-reader's burst loading calls it, and a
production job should never fetch at run time: download once, review, and stage
the files with the rest of the static inputs.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import zipfile
from dataclasses import dataclass

import lxml.etree as ET
import requests

from s1reader.s1_reader import get_aux_ins_name

SAR_MPC_API = "https://sar-mpc.eu/api/v1/"
AUX_TYPES = ("AUX_CAL", "AUX_INS", "AUX_ITC", "AUX_PP1", "AUX_PP2", "AUX_SCF")
_TIMEOUT = 60
_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class AuxRecord:
    """One auxiliary product generation as SAR-MPC lists it."""

    product_name: str
    product_type: str
    validity_start: datetime.datetime
    creation_date: datetime.datetime
    url: str
    md5: str | None
    size: int | None

    @property
    def platform(self) -> str:
        """Mission identifier, e.g. ``"S1D"``."""
        return self.product_name[:3]

    @property
    def file_name(self) -> str:
        """Name the product is shipped under, ``<product_name>.SAFE.zip``."""
        return f"{self.product_name}.SAFE.zip"

    @classmethod
    def from_api(cls, raw: dict) -> "AuxRecord":
        """Build from one ``results`` entry of the SAR-MPC API."""
        hash_ = raw.get("hash") or ""
        return cls(
            product_name=raw["product_name"],
            product_type=raw["product_type"],
            validity_start=_parse_api_time(raw["validity_start"]),
            creation_date=_parse_api_time(raw["creation_date"]),
            url=raw["remote_url"],
            md5=hash_[4:] if hash_.startswith("md5:") else None,
            size=raw.get("size"),
        )


def _parse_api_time(text: str) -> datetime.datetime:
    # The API writes e.g. 2026-03-19T00:00:00Z or with microseconds
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).replace(
        tzinfo=None
    )


def search_aux(
    product_type: str | None = None,
    platform: str | None = None,
    product_name: str | None = None,
    session: requests.Session | None = None,
) -> list[AuxRecord]:
    """List auxiliary product generations on SAR-MPC.

    Parameters
    ----------
    product_type : str, optional
        One of :data:`AUX_TYPES`, e.g. ``"AUX_INS"``.
    platform : str, optional
        Mission, e.g. ``"S1D"``.
    product_name : str, optional
        Exact product name (no ``.SAFE``); returns at most one record.
    session : requests.Session, optional
        Session to use; a new one by default.

    Returns
    -------
    list of AuxRecord
        Every page of results, in API order.
    """
    if product_type is not None and product_type not in AUX_TYPES:
        raise ValueError(f"product_type must be one of {AUX_TYPES}, got {product_type}")
    if product_type is None and product_name is None:
        raise ValueError("Give a product_type or a product_name")

    params = {}
    if product_type:
        params["product_type"] = product_type
    if platform:
        params["sentinel1__mission"] = platform
    if product_name:
        params["product_name"] = product_name

    session = session or requests.Session()
    url = SAR_MPC_API
    records = []
    while url:
        response = session.get(url, params=params, timeout=_TIMEOUT)
        response.raise_for_status()
        page = response.json()
        records.extend(AuxRecord.from_api(r) for r in page["results"])
        # "next" already carries the query string
        url, params = page.get("next"), None
    return records


def latest_aux(
    records: list[AuxRecord], at: datetime.datetime | None = None
) -> dict[str, AuxRecord]:
    """Pick the generation in force at ``at`` for each platform.

    The newest validity start not after ``at``, and of those the newest
    generation: ESA revises aux products in place under the same validity start.

    Parameters
    ----------
    records : list of AuxRecord
        Records of one product type, from :func:`search_aux`.
    at : datetime.datetime, optional
        UTC time; now by default.

    Returns
    -------
    dict
        ``{platform: AuxRecord}``; a platform with nothing valid yet is absent.
    """
    at = at or datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    chosen = {}
    for rec in records:
        if rec.validity_start > at:
            continue
        key = (rec.validity_start, rec.creation_date)
        best = chosen.get(rec.platform)
        if best is None or key > (best.validity_start, best.creation_date):
            chosen[rec.platform] = rec
    return chosen


def _md5(path: str) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_aux(
    record: AuxRecord, out_dir: str, session: requests.Session | None = None
) -> str:
    """Download one auxiliary product as ``<product_name>.SAFE.zip``.

    An existing file with the right MD5 is kept; a download that does not match
    the listed size or MD5 is discarded and raises, so a truncated file never
    sits where a reader would pick it up.

    Parameters
    ----------
    record : AuxRecord
        Product to fetch, from :func:`search_aux`.
    out_dir : str
        Directory to write into; created if needed.
    session : requests.Session, optional
        Session to use; a new one by default.

    Returns
    -------
    str
        Path of the ``.SAFE.zip``.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, record.file_name)
    if os.path.exists(path) and record.md5 and _md5(path) == record.md5:
        return path

    session = session or requests.Session()
    part = f"{path}.part"
    with session.get(record.url, stream=True, timeout=_TIMEOUT) as response:
        response.raise_for_status()
        with open(part, "wb") as f:
            for chunk in response.iter_content(_CHUNK):
                f.write(chunk)

    problem = None
    if record.size is not None and os.path.getsize(part) != record.size:
        problem = f"size {os.path.getsize(part)} != {record.size}"
    elif record.md5 and _md5(part) != record.md5:
        problem = "MD5 mismatch"
    if problem:
        os.remove(part)
        raise IOError(f"Download of {record.product_name} failed: {problem}")
    os.replace(part, path)
    return path


def aux_ins_name_from_safe(safe_path: str) -> str | None:
    """AUX_INS product name recorded in a SAFE's ``manifest.safe``.

    Parameters
    ----------
    safe_path : str
        SLC ``.zip`` or ``.SAFE`` directory.

    Returns
    -------
    str or None
        e.g. ``"S1D_AUX_INS_V20260319T000000_G20260603T113743"``, or None if
        the manifest names no AUX_INS.
    """
    if os.path.isdir(safe_path):
        with open(os.path.join(safe_path, "manifest.safe"), "rb") as f:
            tree = ET.parse(f)
    else:
        with zipfile.ZipFile(safe_path) as z:
            name = next(n for n in z.namelist() if n.endswith("manifest.safe"))
            with z.open(name) as f:
                tree = ET.parse(f)
    return get_aux_ins_name(tree)


def get_parser() -> argparse.ArgumentParser:
    """Command line of :func:`main`."""
    parser = argparse.ArgumentParser(
        description=(
            "Download Sentinel-1 auxiliary products from ESA SAR-MPC "
            "(https://sar-mpc.eu). Pick them by type and platform (the generation "
            "in force now, or every generation with --all), by exact name, or as "
            "the AUX_INS a SAFE was focused with."
        ),
    )
    parser.add_argument(
        "--type",
        nargs="+",
        choices=AUX_TYPES,
        default=[],
        dest="types",
        help="Product types, e.g. AUX_INS AUX_ITC",
    )
    parser.add_argument(
        "--platform",
        nargs="+",
        default=[],
        help="Platforms, e.g. S1C S1D (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Every generation, not only the one in force now",
    )
    parser.add_argument(
        "--name", nargs="+", default=[], help="Exact product names (no .SAFE)"
    )
    parser.add_argument(
        "--from-safe",
        nargs="+",
        default=[],
        metavar="SAFE",
        help="SLC .zip/.SAFE whose manifest names the AUX_INS to fetch",
    )
    parser.add_argument("-o", "--out-dir", default=".", help="Output directory")
    return parser


def main(argv=None, session: requests.Session | None = None) -> list[str]:
    """Download what the command line asks for; print and return the paths."""
    args = get_parser().parse_args(argv)
    if not (args.types or args.name or args.from_safe):
        get_parser().error("give --type, --name or --from-safe")

    session = session or requests.Session()
    names = list(args.name)
    for safe in args.from_safe:
        name = aux_ins_name_from_safe(safe)
        if name is None:
            raise ValueError(f"{safe} names no AUX_INS in its manifest")
        names.append(name)

    records = []
    for name in dict.fromkeys(names):
        found = search_aux(product_name=name, session=session)
        if not found:
            raise ValueError(f"{name} is not on SAR-MPC")
        records.extend(found)
    for product_type in args.types:
        for platform in args.platform or [None]:
            found = search_aux(product_type, platform, session=session)
            records.extend(found if args.all else latest_aux(found).values())

    paths = []
    for record in {r.product_name: r for r in records}.values():
        path = download_aux(record, args.out_dir, session=session)
        print(path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
