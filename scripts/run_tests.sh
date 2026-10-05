#!/usr/bin/env bash
# Single entry point for every project test (ROS + Python learning/experiments).
#
#   bash scripts/run_tests.sh
#
# It does not launch Gazebo or MATLAB; dedicated integration tests exercise
# those runtimes separately.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"

echo "== colcon build =="
colcon build --symlink-install

echo "== colcon test =="
rm -f build/*/pytest.xml
colcon test --event-handlers console_direct+
colcon test-result --verbose

echo "== standalone Python tests (controllers + learning + experiments) =="
python -m pytest \
    controllers/python/tests \
    learning/tests \
    experiments \
    experiments_v2/tests \
    -q

echo "All project tests passed."
