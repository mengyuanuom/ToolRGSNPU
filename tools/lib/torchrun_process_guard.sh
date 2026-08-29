#!/usr/bin/env bash

# Lifecycle guard for a torchrun agent launched by a queue script.
# The agent receives SIGTERM first. A private process group is only used as a
# bounded fallback for ranks or DataLoader workers that outlive the agent.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "torchrun_process_guard.sh must be sourced" >&2
  exit 2
fi

: "${TORCHRUN_GUARD_TERM_TIMEOUT_SECONDS:=30}"
: "${TORCHRUN_GUARD_GROUP_TIMEOUT_SECONDS:=10}"

TORCHRUN_GUARD_ACTIVE_PID=""
TORCHRUN_GUARD_ACTIVE_PGID=""
TORCHRUN_GUARD_ACTIVE_LOG=""

_torchrun_guard_group_is_alive() {
  local pgid="${1:-}"
  [[ -n "${pgid}" ]] || return 1
  kill -0 -- "-${pgid}" 2>/dev/null
}

_torchrun_guard_wait_for_group() {
  local pgid="$1"
  local timeout_seconds="$2"
  local deadline=$((SECONDS + timeout_seconds))
  while _torchrun_guard_group_is_alive "${pgid}"; do
    (( SECONDS < deadline )) || return 1
    sleep 1
  done
}

_torchrun_guard_cleanup_group() {
  local pgid="${1:-}"
  [[ -n "${pgid}" ]] || return 0
  _torchrun_guard_group_is_alive "${pgid}" || return 0

  echo "TORCHRUN_GROUP_REMAINS pgid=${pgid}; sending SIGTERM" >&2
  kill -TERM -- "-${pgid}" 2>/dev/null || true
  if _torchrun_guard_wait_for_group \
      "${pgid}" "${TORCHRUN_GUARD_GROUP_TIMEOUT_SECONDS}"; then
    return 0
  fi
  echo "TORCHRUN_GROUP_TIMEOUT pgid=${pgid}; sending SIGKILL" >&2
  kill -KILL -- "-${pgid}" 2>/dev/null || true
  _torchrun_guard_wait_for_group "${pgid}" 2 || true
}

torchrun_guard_start() {
  local launcher_log="$1"
  shift
  if [[ -n "${TORCHRUN_GUARD_ACTIVE_PID}" ]]; then
    echo "torchrun guard already owns PID ${TORCHRUN_GUARD_ACTIVE_PID}" >&2
    return 70
  fi
  command -v setsid >/dev/null 2>&1 || {
    echo "setsid is required for isolated torchrun cleanup" >&2
    return 71
  }

  setsid "$@" >"${launcher_log}" 2>&1 &
  TORCHRUN_GUARD_ACTIVE_PID=$!
  TORCHRUN_GUARD_ACTIVE_LOG="${launcher_log}"

  local pgid=""
  local attempt
  for attempt in {1..20}; do
    pgid="$(ps -o pgid= -p "${TORCHRUN_GUARD_ACTIVE_PID}" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "${pgid}" ]] && break
    kill -0 "${TORCHRUN_GUARD_ACTIVE_PID}" 2>/dev/null || break
    sleep 0.05
  done
  if [[ -z "${pgid}" ]]; then
    wait "${TORCHRUN_GUARD_ACTIVE_PID}" 2>/dev/null || true
    TORCHRUN_GUARD_ACTIVE_PID=""
    TORCHRUN_GUARD_ACTIVE_LOG=""
    echo "torchrun exited before its process group was recorded" >&2
    return 72
  fi

  local queue_pgid
  queue_pgid="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]')"
  if [[ "${pgid}" == "${queue_pgid}" ]]; then
    kill -TERM "${TORCHRUN_GUARD_ACTIVE_PID}" 2>/dev/null || true
    wait "${TORCHRUN_GUARD_ACTIVE_PID}" 2>/dev/null || true
    TORCHRUN_GUARD_ACTIVE_PID=""
    TORCHRUN_GUARD_ACTIVE_LOG=""
    echo "refusing process group shared with the queue" >&2
    return 73
  fi
  TORCHRUN_GUARD_ACTIVE_PGID="${pgid}"
  echo "TORCHRUN_START pid=${TORCHRUN_GUARD_ACTIVE_PID} pgid=${pgid} log=${launcher_log}"
}

torchrun_guard_wait() {
  local pid="${TORCHRUN_GUARD_ACTIVE_PID:-}"
  local pgid="${TORCHRUN_GUARD_ACTIVE_PGID:-}"
  [[ -n "${pid}" ]] || return 0
  local rc=0
  wait "${pid}" || rc=$?
  TORCHRUN_GUARD_ACTIVE_PID=""
  TORCHRUN_GUARD_ACTIVE_PGID=""
  TORCHRUN_GUARD_ACTIVE_LOG=""
  _torchrun_guard_cleanup_group "${pgid}"
  return "${rc}"
}

torchrun_guard_stop() {
  local pid="${TORCHRUN_GUARD_ACTIVE_PID:-}"
  local pgid="${TORCHRUN_GUARD_ACTIVE_PGID:-}"
  [[ -n "${pid}" ]] || return 0
  echo "TORCHRUN_STOP pid=${pid} pgid=${pgid} log=${TORCHRUN_GUARD_ACTIVE_LOG}" >&2
  kill -TERM "${pid}" 2>/dev/null || true

  (
    sleep "${TORCHRUN_GUARD_TERM_TIMEOUT_SECONDS}"
    if _torchrun_guard_group_is_alive "${pgid}"; then
      echo "TORCHRUN_TERM_TIMEOUT pid=${pid} pgid=${pgid}" >&2
      kill -TERM -- "-${pgid}" 2>/dev/null || true
      sleep "${TORCHRUN_GUARD_GROUP_TIMEOUT_SECONDS}"
    fi
    if _torchrun_guard_group_is_alive "${pgid}"; then
      echo "TORCHRUN_KILL_TIMEOUT pid=${pid} pgid=${pgid}" >&2
      kill -KILL -- "-${pgid}" 2>/dev/null || true
    fi
  ) &
  local watchdog_pid=$!

  wait "${pid}" 2>/dev/null || true
  kill -TERM "${watchdog_pid}" 2>/dev/null || true
  wait "${watchdog_pid}" 2>/dev/null || true
  TORCHRUN_GUARD_ACTIVE_PID=""
  TORCHRUN_GUARD_ACTIVE_PGID=""
  TORCHRUN_GUARD_ACTIVE_LOG=""
  _torchrun_guard_cleanup_group "${pgid}"
}

_torchrun_guard_handle_signal() {
  local signal_name="$1"
  local exit_code="$2"
  trap - INT TERM
  echo "QUEUE_SIGNAL signal=${signal_name}" >&2
  torchrun_guard_stop
  exit "${exit_code}"
}

_torchrun_guard_handle_exit() {
  local exit_code=$?
  trap - EXIT INT TERM
  torchrun_guard_stop
  exit "${exit_code}"
}

torchrun_guard_install() {
  trap '_torchrun_guard_handle_signal SIGINT 130' INT
  trap '_torchrun_guard_handle_signal SIGTERM 143' TERM
  trap '_torchrun_guard_handle_exit' EXIT
}
