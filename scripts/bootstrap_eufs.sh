#!/usr/bin/env bash
# Reproducible EUFS Sim 2 bootstrap.
#
#   bash scripts/bootstrap_eufs.sh            # full bootstrap
#   bash scripts/bootstrap_eufs.sh --check-only
#
# It never calls sudo.  If system packages are missing it prints the single apt
# command to run and exits non-zero.  It never runs `git pull`; all checkouts
# are pinned by third_party/eufs.repos.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EUFS_WS="$REPO_ROOT/external/eufs_ws"
EUFS_SRC="$EUFS_WS/src"
PATCH_DIR="$REPO_ROOT/third_party/patches"
BOOTSTRAP_DIR="$REPO_ROOT/runs/eufs_v1/bootstrap"

REQUIRED_PACKAGES=(
    ros-humble-geodesy
    python3-vcstool
    libyaml-cpp-dev
    libjsoncpp-dev
    libspdlog-dev
)

CHECK_ONLY=0
if [ "${1:-}" = "--check-only" ]; then
    CHECK_ONLY=1
fi

LOCAL_PREFIX="$REPO_ROOT/external/deps/local_prefix"

missing=()
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -s "$package" >/dev/null 2>&1; then
        missing+=("$package")
    fi
done

if [ "${#missing[@]}" -gt 0 ]; then
    echo "System packages missing: ${missing[*]}"
    echo "Installing them locally into $LOCAL_PREFIX (no sudo required)."
    echo "Preferred permanent fix (single command, needs sudo password):"
    echo "  sudo apt update && sudo apt install ${missing[*]}"
    download_dir="$(mktemp -d)"
    if ! (cd "$download_dir" && apt-get download "${missing[@]}"); then
        echo "error: could not download ${missing[*]}; install them with apt." >&2
        rm -rf "$download_dir"
        exit 1
    fi
    mkdir -p "$LOCAL_PREFIX"
    for package_file in "$download_dir"/*.deb; do
        dpkg-deb -x "$package_file" "$LOCAL_PREFIX"
    done
    rm -rf "$download_dir"
    echo "Local dependency prefix ready: $LOCAL_PREFIX"
fi

# Expected commits, checked after import.
declare -A EXPECTED=(
    [eufs_sim2]=9f5df79a03725ea7d10542fc2ce8224d90836560
    [eufs_msgs]=9e918686c9e9292c613f321e6fd85e3a5d87cd87
    [eufs-gmock-matchers]=7ef83d030746c6a31bcf4f888d4121fcf4b7e8a9
    [eufs-logger]=375ea1d8f8885af66809129e444624ba13353fa7
    [state_lib]=ec83a141f188e8a4c39a381f4666485d8cc83e20
    [map_lib]=1919b36062850c9ba4553d1833a9b517c61c2e86
    [vehicle_models]=3508bec2c3d77e0ff16f08794675d4f7b52479b7
)

if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "bootstrap_eufs.sh --check-only: system packages OK."
    exit 0
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"

mkdir -p "$EUFS_SRC" "$BOOTSTRAP_DIR"
touch "$REPO_ROOT/external/COLCON_IGNORE"

if [ -z "$(ls -A "$EUFS_SRC" 2>/dev/null)" ]; then
    echo "Importing pinned EUFS repositories"
    vcs import "$EUFS_SRC" < "$REPO_ROOT/third_party/eufs.repos"
else
    echo "EUFS source tree already present; skipping import"
fi

for name in "${!EXPECTED[@]}"; do
    actual="$(git -C "$EUFS_SRC/$name" rev-parse HEAD)"
    if [ "$actual" != "${EXPECTED[$name]}" ]; then
        echo "error: $name is at $actual, expected ${EXPECTED[$name]}" >&2
        exit 1
    fi
done
echo "All pinned commits verified."

apply_patch() {
    local patch_path="$1"
    local repo="$2"
    if git -C "$repo" apply --reverse --check "$patch_path" >/dev/null 2>&1; then
        echo "patch already applied: $(basename "$patch_path")"
    elif git -C "$repo" apply --check "$patch_path" >/dev/null 2>&1; then
        git -C "$repo" apply "$patch_path"
        echo "applied: $(basename "$patch_path")"
    else
        echo "error: $(basename "$patch_path") does not apply cleanly to $repo" >&2
        exit 1
    fi
}

if [ -f "$PATCH_DIR/map_lib_core_only.patch" ]; then
    apply_patch "$PATCH_DIR/map_lib_core_only.patch" "$EUFS_SRC/map_lib"
fi
if [ -f "$PATCH_DIR/eufs_sim2_imu_orientation.patch" ]; then
    apply_patch "$PATCH_DIR/eufs_sim2_imu_orientation.patch" "$EUFS_SRC/eufs_sim2"
fi
if [ -f "$PATCH_DIR/vehicle_models_applied_steering.patch" ]; then
    apply_patch "$PATCH_DIR/vehicle_models_applied_steering.patch" "$EUFS_SRC/vehicle_models"
fi
if [ -f "$PATCH_DIR/vehicle_models_grip_scale.patch" ]; then
    apply_patch "$PATCH_DIR/vehicle_models_grip_scale.patch" "$EUFS_SRC/vehicle_models"
fi
if [ -f "$PATCH_DIR/eufs_sim2_runtime_grip.patch" ]; then
    apply_patch "$PATCH_DIR/eufs_sim2_runtime_grip.patch" "$EUFS_SRC/eufs_sim2"
fi

vcs export --exact "$EUFS_SRC" > "$BOOTSTRAP_DIR/eufs_exact.repos"
echo "Exact repository state written to $BOOTSTRAP_DIR/eufs_exact.repos"
