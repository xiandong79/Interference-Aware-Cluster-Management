#!/bin/sh
echo "kill all benchmarks"
kill -9 `pgrep gcc`
kill -9 `pgrep live`
kill -9 `pgrep base`

echo "kill all ibench"
kill -9 `pgrep cpu`
kill -9 `pgrep l1d`
kill -9 `pgrep l1i`
kill -9 `pgrep l2d`
kill -9 `pgrep l3d`
kill -9 `pgrep membw`
kill -9 `pgrep memcap`

echo "kill all monitors"
kill -9 `pgrep task`

sleep 3
