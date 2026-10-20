#!/bin/bash
# ESnet East pod probe: resources, storage, identity, egress. Synthetic only; runs in python:3.12-slim.
echo "== start $(date -u +%FT%TZ) host $(hostname)"
echo "== id"; id; echo "HOME=$HOME PWD=$PWD"
echo "== cpu"; nproc; grep -m1 'model name' /proc/cpuinfo
cat /sys/fs/cgroup/cpu.max 2>/dev/null
echo "== memory"; grep -E 'MemTotal|MemAvailable' /proc/meminfo; cat /sys/fs/cgroup/memory.max 2>/dev/null
echo "== disk"; df -h / /tmp /dev/shm /data/home/* /data/projects/* 2>/dev/null; mount | grep -E ' / | /tmp | /data' | head
echo "== gpu"; ls /dev/nvidia* 2>/dev/null || echo "no /dev/nvidia*"
echo "== fuse/userns"; ls -l /dev/fuse 2>/dev/null || echo "no /dev/fuse"; cat /proc/sys/user/max_user_namespaces 2>/dev/null
echo "== env (proxy, k8s, IRI)"; env | grep -i -E 'proxy|^KUBERNETES_SERVICE_HOST|_VISIBLE_DEVICES|^IRI|^JOB' | sort
echo "== tmp write speed"; dd if=/dev/zero of=/tmp/ddtest bs=1M count=512 conv=fdatasync 2>&1 | tail -1; rm -f /tmp/ddtest
echo "== efs write speed"; dd if=/dev/zero of="$PWD/ddtest" bs=1M count=64 conv=fdatasync 2>&1 | tail -1; rm -f "$PWD/ddtest"
echo "== egress"
python3 - <<'EOF'
import socket, time, urllib.request
for host, port in [("pypi.org", 443), ("files.pythonhosted.org", 443), ("github.com", 443), ("ghcr.io", 443),
                   ("cvmfs-s1fnal.opensciencegrid.org", 8000), ("cvmfs-s1bnl.opensciencegrid.org", 8000),
                   ("dbdata0vm.fnal.gov", 9443)]:
    t = time.time()
    try:
        ip = socket.gethostbyname(host)
    except Exception as e:
        print(f"DNS  FAIL {host}: {e}"); continue
    try:
        socket.create_connection((ip, port), timeout=8).close()
        print(f"TCP  ok   {host}:{port} ({ip}) {1e3*(time.time()-t):.0f} ms")
    except Exception as e:
        print(f"TCP  FAIL {host}:{port} ({ip}): {e}")
for url in ["https://pypi.org/simple/jax/", "https://github.com/RobMina/mu2e-fastsim",
            "http://cvmfs-s1fnal.opensciencegrid.org:8000/cvmfs/mu2e.opensciencegrid.org/.cvmfspublished"]:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            print(f"HTTP {r.status} {url} ({len(r.read())} B)")
    except Exception as e:
        print(f"HTTP FAIL {url}: {e}")
EOF
echo "== done $(date -u +%FT%TZ)"
