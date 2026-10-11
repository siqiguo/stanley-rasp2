#!/bin/sh
set -eu

env_file="${HOME}/.config/front-gate-package-monitor.env"
set -a
# The EnvironmentFile is locally managed and may use quoted or unquoted values.
. "$env_file"
set +a
if [ -z "${CALIBRATION_TOKEN:-}" ]; then
  echo "Calibration token is missing" >&2
  exit 1
fi

curl --fail --silent --show-error \
  -X POST \
  -H "X-Calibration-Token: ${CALIBRATION_TOKEN}" \
  http://127.0.0.1:8791/api/calibrate
echo
