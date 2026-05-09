#!/usr/bin/env bash
# Idempotent first-run installer for the teletop server on a Raspberry Pi.
#
# Run with sudo. Re-running is safe — apt-get / usermod / mkdir / uv sync /
# udev install are all idempotent operations.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "This script must be run with sudo." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$SERVER_DIR/.." && pwd)"

TARGET_USER="${SUDO_USER:-${USER:-talat}}"
TARGET_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
if [[ -z "$TARGET_HOME" ]]; then
    echo "Could not resolve home directory for user '$TARGET_USER'." >&2
    exit 1
fi

DATA_DIR="$TARGET_HOME/teletop"
REGISTRY_FILE="$DATA_DIR/devices.json"
RULES_FILE="/etc/udev/rules.d/99-teletop.rules"

echo "==> teletop setup"
echo "    repo:        $REPO_DIR"
echo "    target user: $TARGET_USER"
echo "    data dir:    $DATA_DIR"

# 1. APT dependencies. DEBIAN_FRONTEND=noninteractive avoids the dpkg
#    config prompts that otherwise stall an unattended install on RPi OS.
echo "==> installing apt packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv build-essential \
    network-manager udev curl ca-certificates

# 2. Add target user to the dialout group so they can open /dev/ttyUSB*
#    without sudo. usermod is a no-op if the user is already a member.
if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -qx dialout; then
    echo "==> $TARGET_USER already in 'dialout'"
else
    echo "==> adding $TARGET_USER to 'dialout'"
    usermod -aG dialout "$TARGET_USER"
    DIALOUT_NEWLY_ADDED=1
fi

# 3. Data dir owned by the target user.
install -d -o "$TARGET_USER" -g "$TARGET_USER" "$DATA_DIR"

# 4. uv presence check. Don't install uv as root — it lives in the user's
#    ~/.local. We just point the user at the install command if missing.
UV_BIN="$(sudo -iu "$TARGET_USER" bash -lc 'command -v uv' || true)"
if [[ -z "$UV_BIN" ]]; then
    cat <<EOF >&2
==> uv not found in $TARGET_USER's PATH

Install uv (as $TARGET_USER, NOT root):

  sudo -iu $TARGET_USER bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'

Then re-run this script.
EOF
    exit 1
fi
echo "==> uv at $UV_BIN"

# 5. Sync server deps as the target user (so the venv is owned by them).
echo "==> uv sync server"
sudo -iu "$TARGET_USER" bash -lc "cd '$SERVER_DIR' && uv sync --frozen"

# 6. udev rules. If the registry already exists with at least one device,
#    regenerate and install the rules. First-time installs (no registry yet)
#    just print the next-step hint.
if [[ -s "$REGISTRY_FILE" ]]; then
    echo "==> generating udev rules from $REGISTRY_FILE"
    RULES_CONTENT="$(sudo -iu "$TARGET_USER" bash -lc \
        "cd '$SERVER_DIR' && uv run teletop-server udev-rules")"
    TMP_RULES="$(mktemp)"
    printf '%s\n' "$RULES_CONTENT" > "$TMP_RULES"
    install -m 0644 -o root -g root "$TMP_RULES" "$RULES_FILE"
    rm -f "$TMP_RULES"
    udevadm control --reload
    udevadm trigger
    echo "==> udev rules installed at $RULES_FILE"
else
    echo "==> no devices registered yet — skipping udev rules"
    echo "    after registering, run:"
    echo "      sudo \$(which uv) run teletop-server udev-install"
fi

echo
echo "==> done."
if [[ "${DIALOUT_NEWLY_ADDED:-0}" == "1" ]]; then
    echo "    NOTE: $TARGET_USER was just added to the 'dialout' group;"
    echo "    log out and back in (or reboot) for the membership to take effect."
fi
