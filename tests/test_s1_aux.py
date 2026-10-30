"""Tests for the SAR-MPC auxiliary product search and download.

A fake session stands in for requests: it serves canned API pages and file
bodies and records every call, so the tests run offline and can assert what
was asked of the server.
"""

import datetime
import hashlib
import json
import pathlib
import sys

import pytest

from s1reader import s1_aux
from s1reader.s1_aux import AuxRecord

BODY = b"not really a zip, but bytes with a known md5"
MD5 = hashlib.md5(BODY).hexdigest()


def api_entry(name, validity, created, uuid="u1", body=BODY):
    return {
        "product_type": name.split("_")[1] + "_" + name.split("_")[2],
        "product_name": name,
        "validity_start": validity,
        "creation_date": created,
        "remote_url": f"https://sar-mpc.eu/download/{uuid}/",
        "hash": "md5:" + hashlib.md5(body).hexdigest(),
        "size": len(body),
    }


class FakeResponse:
    def __init__(self, payload=None, body=b"", status=200):
        self.payload, self.body, self.status = payload, body, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i : i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Serves API pages keyed by (url, frozen params) and downloads by url."""

    def __init__(self, pages=None, files=None):
        self.pages = pages or {}
        self.files = files or {}
        self.calls = []

    def get(self, url, params=None, timeout=None, stream=False):
        self.calls.append((url, params))
        if url in self.files:
            return FakeResponse(body=self.files[url])
        key = (url, tuple(sorted((params or {}).items())))
        return FakeResponse(payload=self.pages[key])


def page(results, next_url=None):
    return {"count": len(results), "next": next_url, "results": results}


S1D_INS = [
    api_entry(
        "S1D_AUX_INS_V20260319T000000_G20260318T105812",
        "2026-03-19T00:00:00Z",
        "2026-03-18T10:58:12Z",
        "a",
    ),
    api_entry(
        "S1D_AUX_INS_V20260319T000000_G20260603T113743",
        "2026-03-19T00:00:00Z",
        "2026-06-03T11:37:43Z",
        "b",
    ),
    api_entry(
        "S1D_AUX_INS_V20251104T000000_G20251029T104247",
        "2025-11-04T00:00:00Z",
        "2025-10-29T10:42:47Z",
        "c",
    ),
]


def api_key(**params):
    return (s1_aux.SAR_MPC_API, tuple(sorted(params.items())))


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def test_search_sends_api_filters_and_parses_records():
    session = FakeSession(
        pages={api_key(product_type="AUX_INS", sentinel1__mission="S1D"): page(S1D_INS)}
    )
    records = s1_aux.search_aux("AUX_INS", "S1D", session=session)
    assert session.calls == [
        (s1_aux.SAR_MPC_API, {"product_type": "AUX_INS", "sentinel1__mission": "S1D"})
    ]
    rec = records[1]
    assert rec.product_name == "S1D_AUX_INS_V20260319T000000_G20260603T113743"
    assert rec.platform == "S1D"
    assert rec.validity_start == datetime.datetime(2026, 3, 19)
    assert rec.creation_date == datetime.datetime(2026, 6, 3, 11, 37, 43)
    assert rec.md5 == MD5
    assert rec.file_name.endswith(".SAFE.zip")


def test_search_follows_every_page():
    nxt = "https://sar-mpc.eu/api/v1/?page=2&product_type=AUX_INS"
    session = FakeSession(
        pages={
            api_key(product_type="AUX_INS"): page(S1D_INS[:1], nxt),
            (nxt, ()): page(S1D_INS[1:]),
        }
    )
    assert len(s1_aux.search_aux("AUX_INS", session=session)) == 3
    # the next link carries its own query string; params are not resent
    assert session.calls[1] == (nxt, None)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "product_type or a product_name"),
        ({"product_type": "AUX_FOO"}, "product_type must be one of"),
    ],
)
def test_search_rejects_bad_queries(kwargs, match):
    with pytest.raises(ValueError, match=match):
        s1_aux.search_aux(session=FakeSession(), **kwargs)


# --------------------------------------------------------------------------- #
# latest
# --------------------------------------------------------------------------- #
def records():
    return [AuxRecord.from_api(e) for e in S1D_INS]


def test_latest_picks_newest_validity_then_newest_generation():
    chosen = s1_aux.latest_aux(records(), at=datetime.datetime(2026, 10, 23))
    assert chosen["S1D"].product_name == "S1D_AUX_INS_V20260319T000000_G20260603T113743"


def test_latest_ignores_products_not_yet_valid():
    chosen = s1_aux.latest_aux(records(), at=datetime.datetime(2026, 1, 1))
    assert chosen["S1D"].product_name == "S1D_AUX_INS_V20251104T000000_G20251029T104247"
    assert s1_aux.latest_aux(records(), at=datetime.datetime(2025, 1, 1)) == {}


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def test_download_writes_safe_zip(tmp_path):
    rec = records()[1]
    session = FakeSession(files={rec.url: BODY})
    path = s1_aux.download_aux(rec, str(tmp_path / "aux"), session=session)
    assert pathlib.Path(path).name == rec.file_name
    assert pathlib.Path(path).read_bytes() == BODY
    assert not list((tmp_path / "aux").glob("*.part"))


def test_download_keeps_a_verified_file(tmp_path):
    rec = records()[1]
    (tmp_path / rec.file_name).write_bytes(BODY)
    session = FakeSession()
    s1_aux.download_aux(rec, str(tmp_path), session=session)
    assert session.calls == []


def test_download_replaces_a_stale_file(tmp_path):
    rec = records()[1]
    (tmp_path / rec.file_name).write_bytes(b"old generation")
    session = FakeSession(files={rec.url: BODY})
    s1_aux.download_aux(rec, str(tmp_path), session=session)
    assert (tmp_path / rec.file_name).read_bytes() == BODY


@pytest.mark.parametrize(
    "body,match", [(BODY[:-1], "size"), (BODY[::-1], "MD5 mismatch")]
)
def test_download_discards_a_bad_file(tmp_path, body, match):
    rec = records()[1]
    session = FakeSession(files={rec.url: body})
    with pytest.raises(IOError, match=match):
        s1_aux.download_aux(rec, str(tmp_path), session=session)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# SAFE manifest + CLI
# --------------------------------------------------------------------------- #
def test_aux_ins_name_from_test_safe(test_paths):
    assert (
        s1_aux.aux_ins_name_from_safe(test_paths.safe)
        == "S1A_AUX_INS_V20171017T080000_G20180313T104658"
    )


def test_cli_from_safe_fetches_exactly_that_generation(test_paths, tmp_path, capsys):
    name = "S1A_AUX_INS_V20171017T080000_G20180313T104658"
    entry = api_entry(name, "2017-10-17T08:00:00Z", "2018-03-13T10:46:58Z")
    session = FakeSession(
        pages={api_key(product_name=name): page([entry])},
        files={entry["remote_url"]: BODY},
    )
    s1_aux.main(["--from-safe", test_paths.safe, "-o", str(tmp_path)], session=session)
    paths = capsys.readouterr().out.split()
    assert [pathlib.Path(p).name for p in paths] == [f"{name}.SAFE.zip"]


def test_cli_type_and_platform_takes_the_current_generation(tmp_path, capsys):
    session = FakeSession(
        pages={
            api_key(product_type="AUX_INS", sentinel1__mission="S1D"): page(S1D_INS)
        },
        files={e["remote_url"]: BODY for e in S1D_INS},
    )
    s1_aux.main(
        ["--type", "AUX_INS", "--platform", "S1D", "-o", str(tmp_path)], session=session
    )
    paths = capsys.readouterr().out.split()
    assert [pathlib.Path(p).name for p in paths] == [
        "S1D_AUX_INS_V20260319T000000_G20260603T113743.SAFE.zip"
    ]


def test_cli_all_takes_every_generation(tmp_path, capsys):
    session = FakeSession(
        pages={
            api_key(product_type="AUX_INS", sentinel1__mission="S1D"): page(S1D_INS)
        },
        files={e["remote_url"]: BODY for e in S1D_INS},
    )
    s1_aux.main(
        ["--type", "AUX_INS", "--platform", "S1D", "--all", "-o", str(tmp_path)],
        session=session,
    )
    assert len(capsys.readouterr().out.split()) == 3


def test_cli_success_exits_zero(tmp_path):
    """The console script runs sys.exit(main()); a return value there is a failure."""
    session = FakeSession(
        pages={
            api_key(product_type="AUX_INS", sentinel1__mission="S1D"): page(S1D_INS)
        },
        files={e["remote_url"]: BODY for e in S1D_INS},
    )
    argv = ["--type", "AUX_INS", "--platform", "S1D", "-o", str(tmp_path)]
    with pytest.raises(SystemExit) as err:
        sys.exit(s1_aux.main(argv, session=session))
    assert err.value.code is None


def test_cli_unknown_name_fails():
    name = "S1D_AUX_INS_V20990101T000000_G20990101T000000"
    session = FakeSession(pages={api_key(product_name=name): page([])})
    with pytest.raises(ValueError, match="not on SAR-MPC"):
        s1_aux.main(["--name", name], session=session)


def test_cli_needs_something_to_fetch():
    with pytest.raises(SystemExit) as err:
        s1_aux.main([], session=FakeSession())
    assert err.value.code == 2


def test_cli_help():
    with pytest.raises(SystemExit) as err:
        s1_aux.main(["--help"])
    assert err.value.code == 0
