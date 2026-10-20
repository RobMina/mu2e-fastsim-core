"""Fetch single files from a CVMFS repository over HTTP from a Stratum-1, without a CVMFS mount (e.g. Polaris compute
nodes, which reach the Stratum-1s through the site proxy but cannot mount CVMFS).

CVMFS layout: <server>/cvmfs/<repo>/.cvmfspublished gives the root catalog hash (line 'C...'); a catalog is a
zlib-compressed SQLite file at data/<h[:2]>/<h[2:]>C; its `catalog` table maps md5(path) to the file's content hash
(`hash`, 20-byte SHA-1), its `nested_catalogs` table lists sub-catalogs by path prefix. A file's content is at
data/<h[:2]>/<h[2:]> (zlib-compressed); large files are chunked (table `chunks`). Uses only the standard library.

    python3 -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/photonMap.root OUT
"""
import hashlib
import os
import sqlite3
import struct
import sys
import tempfile
import urllib.request
import zlib

SERVERS = ("http://cvmfs-s1fnal.opensciencegrid.org:8000", "http://cvmfs-s1bnl.opensciencegrid.org:8000",
           "http://oasis-replica.opensciencegrid.org:8000")


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mu2efastsim-cvmfsget"}),
                                timeout=60) as r:
        return r.read()


class Repo:
    def __init__(self, repo, servers=SERVERS):
        self.repo, self.servers, self._cat = repo, servers, {}
        err = None
        for s in servers:
            try:
                pub = _get(f"{s}/cvmfs/{repo}/.cvmfspublished").decode("latin-1")
                self.server = s
                break
            except Exception as e:  # noqa: BLE001
                err = e
        else:
            raise RuntimeError(f"no Stratum-1 answered for {repo}: {err}")
        self.root = next(l[1:] for l in pub.splitlines() if l.startswith("C"))

    def _object(self, h, suffix=""):
        return zlib.decompress(_get(f"{self.server}/cvmfs/{self.repo}/data/{h[:2]}/{h[2:]}{suffix}"))

    def _catalog(self, h):
        if h not in self._cat:
            fd, path = tempfile.mkstemp(suffix=".sqlite")
            with os.fdopen(fd, "wb") as f:
                f.write(self._object(h, "C"))
            self._cat[h] = sqlite3.connect(path)
        return self._cat[h]

    def _lookup(self, path):
        """(catalog connection, row) for an absolute repository path, descending into nested catalogs."""
        h = self.root
        while True:
            db = self._catalog(h)
            nested = db.execute("SELECT path, sha1 FROM nested_catalogs").fetchall()
            sub = [(p, s) for p, s in nested if path == p or path.startswith(p + "/")]
            if sub:
                h = max(sub, key=lambda x: len(x[0]))[1]
                continue
            m = hashlib.md5(path.encode()).digest()
            p1, p2 = struct.unpack("<qq", m)
            row = db.execute("SELECT hash, size, flags FROM catalog WHERE md5path_1=? AND md5path_2=?", (p1, p2)).fetchone()
            if row is None:
                raise FileNotFoundError(f"{self.repo}{path}")
            return db, (p1, p2), row

    def read(self, path):
        db, key, (h, size, flags) = self._lookup(path)
        chunks = db.execute("SELECT offset, hash FROM chunks WHERE md5path_1=? AND md5path_2=? ORDER BY offset",
                            key).fetchall()
        if chunks:
            data = b"".join(self._object(ch.hex(), "P") for _, ch in chunks)
        else:
            data = self._object(h.hex())
        if len(data) != size:
            raise IOError(f"{path}: got {len(data)} bytes, catalog says {size}")
        return data

    def fetch(self, path, out):
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "wb") as f:
            f.write(self.read(path))
        return out


if __name__ == "__main__":
    repo, path, out = sys.argv[1:4]
    print(Repo(repo).fetch(path, out))
