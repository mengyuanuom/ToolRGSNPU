#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="${REPO_ROOT}/tools/lib/torchrun_process_guard.sh"
TMP_DIR="$(mktemp -d)"
QUEUE_PID=""

cleanup() {
  if [[ -n "${QUEUE_PID}" ]] && kill -0 "${QUEUE_PID}" 2>/dev/null; then
    kill -KILL "${QUEUE_PID}" 2>/dev/null || true
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

cat >"${TMP_DIR}/fake_queue.sh" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
source "${GUARD}"
TORCHRUN_GUARD_TERM_TIMEOUT_SECONDS=3
TORCHRUN_GUARD_GROUP_TIMEOUT_SECONDS=2
torchrun_guard_install
torchrun_guard_start "${TMP_DIR}/launcher.log" bash -c \
  'trap "exit 0" TERM; sleep 300 & wait'
echo "\${TORCHRUN_GUARD_ACTIVE_PID} \${TORCHRUN_GUARD_ACTIVE_PGID}" \
  >"${TMP_DIR}/active"
torchrun_guard_wait
EOF
chmod +x "${TMP_DIR}/fake_queue.sh"

bash "${TMP_DIR}/fake_queue.sh" >"${TMP_DIR}/queue.log" 2>&1 &
QUEUE_PID=$!

for _ in {1..50}; do
  [[ -s "${TMP_DIR}/active" ]] && break
  sleep 0.1
done
[[ -s "${TMP_DIR}/active" ]] || {
  echo "guard did not record its active process" >&2
  exit 1
}

read -r agent_pid agent_pgid <"${TMP_DIR}/active"
kill -TERM "${QUEUE_PID}"

for _ in {1..100}; do
  kill -0 "${QUEUE_PID}" 2>/dev/null || break
  sleep 0.1
done
wait "${QUEUE_PID}" || rc=$?
[[ "${rc:-0}" -eq 143 ]] || {
  echo "unexpected queue exit code: ${rc:-0}" >&2
  exit 1
}
kill -0 "${agent_pid}" 2>/dev/null && {
  echo "agent PID still exists: ${agent_pid}" >&2
  exit 1
}
kill -0 -- "-${agent_pgid}" 2>/dev/null && {
  echo "process group still exists: ${agent_pgid}" >&2
  exit 1
}

QUEUE_PID=""
echo "torchrun process guard test passed"
