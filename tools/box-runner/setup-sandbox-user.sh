#!/usr/bin/env bash
# setup-sandbox-user.sh -- the sandbox user participant code runs as on a ranked
# box (docs/sandbox-design.md section 5; every name below is section 6a's).
#
#   sudo tools/box-runner/setup-sandbox-user.sh           one-time setup, as root
#   tools/box-runner/setup-sandbox-user.sh --check        verify only, as runner
#
# Setup is idempotent and ends by running --check. It creates:
#   - group nemoshare (members runner and nemosub; runner's other groups untouched)
#   - system user nemosub: shell /usr/sbin/nologin, home /var/lib/nemosub 0700,
#     not in docker, bench or sudo
#   - /var/lib/nemotron35/sandbox, runner:nemoshare 2750
#   - /etc/sudoers.d/nemotron35-sandbox (validated with visudo, installed by rename)
#   - a Rust toolchain in /opt/rust (root-owned, world-readable), the same
#     toolchain runner uses. Callers set RUSTUP_HOME=/opt/rust/rustup and put
#     /opt/rust/cargo/bin on PATH.
#   - iptables + ip6tables: nemosub may send NO TCP/UDP at all, loopback
#     included (it only ever needs the resident's Unix socket); the older
#     loopback-exempt rule is removed if a previous run installed it.
#     Re-applied at every boot by nemotron35-sandbox-egress.service (netfilter
#     rules are not persistent unless iptables-persistent is installed).
#   - an ACL denying nemosub the host resolver sockets (systemd-resolved's
#     varlink socket, nscd's), re-applied at every boot by
#     nemotron35-sandbox-acl.service, because /run is a tmpfs
#   - nemosub in /etc/cron.deny and /etc/at.deny, no crontab, no at job, no
#     linger: nothing it schedules can outlive `kill -1`
#   - traverse-only (--x) ACLs for nemosub on runner's home, the runner
#     install and its _work dir, and a deny (---) on every other entry directly
#     under them (runner's dotfiles and caches, actions-runner/.env, _diag,
#     .credentials*, _work's runner-internal _* dirs): --x on a directory would
#     otherwise open every world-readable entry in it to a guessed name. The
#     workflow grants read on each job's checkout itself and repeats the deny
#     per job (setfacl, package acl, is required)
#   - the wipe list /etc/nemotron35-sandbox/wipe-dirs (root, 0644):
#     every directory on the box nemosub can write (world-writable: /tmp,
#     /var/tmp, /run/lock, /var/crash, ... plus /dev/shm and /dev/mqueue), which
#     tools/sandbox-wipe.sh clears of nemosub's files at every job start and end.
#     Re-run setup after adding a mount or a world-writable directory.
#   - the resident log dir (SERVE_UP_LOG_DIR from runner's actions-runner/.env,
#     else /var/lib/nemotron35/serve-log) runner:runner 0750, other access
#     removed from everything already in it: a candidate resident's log must not
#     be readable by a later job's nemosub
#
# --check changes nothing and prints one PASS/FAIL line per item (WARN for an
# item only root can read). It runs as runner or root.
#
# Names (env; the default is the name used above). A second runner on the same
# box sets its own value for each, so its setup adds its own user, rules, units
# and files and never changes the first runner's (docs/sandbox-design.md 6b):
#   RUNNER_USER   runner                        the runner account
#   SANDBOX_USER  nemosub                       the sandbox user (home /var/lib/<name>)
#   SHARE_GROUP   nemoshare                     the group of the two
#   SANDBOX_ROOT  /var/lib/nemotron35/sandbox   the per-job sandbox root
#   SANDBOX_ETC   /etc/nemotron35-sandbox       holds wipe-dirs and egress.sh
#   SUDOERS_NAME  nemotron35-sandbox            the file in /etc/sudoers.d (no dot)
#   EGRESS_UNIT   nemotron35-sandbox-egress.service
#   ACL_UNIT      nemotron35-sandbox-acl.service
#   RUST_DIR      /opt/rust                     the sandbox user's Rust toolchain
#   RUNNER_DIR    <runner home>/actions-runner  the runner install, directly under
#                                               the runner home (a second runner of
#                                               the same account has its own)
# --check reads the same names, so the runner's jobs must have them too
# (actions-runner/.env).
#
# Exit: 0 all checks passed; 1 a check failed; 78 usage or setup error.
set -euo pipefail

SANDBOX_USER="${SANDBOX_USER:-nemosub}"
RUNNER_USER="${RUNNER_USER:-runner}"
SHARE_GROUP="${SHARE_GROUP:-nemoshare}"
SANDBOX_HOME="/var/lib/${SANDBOX_USER}"
SANDBOX_ROOT="${SANDBOX_ROOT:-/var/lib/nemotron35/sandbox}"
SANDBOX_ETC="${SANDBOX_ETC:-/etc/nemotron35-sandbox}"
SUDOERS_NAME="${SUDOERS_NAME:-nemotron35-sandbox}"
SUDOERS_FILE="/etc/sudoers.d/${SUDOERS_NAME}"
SUDOERS_LINE="${RUNNER_USER} ALL=(${SANDBOX_USER}) NOPASSWD:SETENV: ALL"
RUST_DIR="${RUST_DIR:-/opt/rust}"
FORBIDDEN_GROUPS=(docker bench sudo)
EGRESS_RULE=(OUTPUT -m owner --uid-owner "${SANDBOX_USER}" -j REJECT)
# The rule an earlier version of this script installed (loopback exempt).
OLD_EGRESS_RULE=(OUTPUT -m owner --uid-owner "${SANDBOX_USER}" ! -o lo -j REJECT)
RESOLVER_SOCKETS=(/run/systemd/resolve/io.systemd.Resolve /run/nscd/socket)
ACL_UNIT="/etc/systemd/system/${ACL_UNIT:-nemotron35-sandbox-acl.service}"
EGRESS_UNIT="/etc/systemd/system/${EGRESS_UNIT:-nemotron35-sandbox-egress.service}"
DENY_FILES=(/etc/cron.deny /etc/at.deny)
WIPE_LIST="${SANDBOX_ETC}/wipe-dirs"
# Local filesystems scanned for world-writable directories (the kernel and
# container-image trees are not: /proc and /sys are pseudo filesystems, and
# docker's and containerd's layers sit under root-only directories).
WIPE_FS_TYPES=ext2,ext3,ext4,xfs,btrfs,f2fs,vfat,tmpfs
WIPE_ALWAYS=(/dev/shm /dev/mqueue)
DEFAULT_SERVE_LOG=/var/lib/nemotron35/serve-log
PUBLIC_V4=1.1.1.1
PUBLIC_V6=2606:4700:4700::1111

fail() { echo "setup-sandbox-user.sh: $*" >&2; exit 78; }
info() { echo "setup-sandbox-user.sh: $*"; }

IS_ROOT=0
[[ "$(id -u)" -eq 0 ]] && IS_ROOT=1

# as_runner CMD... -- run as runner (we are runner, or root dropping to it).
as_runner() {
  if [[ "${IS_ROOT}" -eq 1 ]]; then runuser -u "${RUNNER_USER}" -- "$@"; else "$@"; fi
}

# as_sandbox CMD... -- the contract's drop: runner's sudo to nemosub.
as_sandbox() {
  as_runner sudo -n -u "${SANDBOX_USER}" -- "$@"
}

# runner_sh SCRIPT -- runner's own login environment, the way supervisor.sh
# starts a job (su -), with rustup's default bin dir first.
runner_sh() {
  local script="cd / && PATH=\"\${HOME}/.cargo/bin:\${PATH}\" && ${1}"
  if [[ "${IS_ROOT}" -eq 1 ]]; then su - "${RUNNER_USER}" -c "${script}"; else bash -c "${script}"; fi
}

sandbox_cargo_version() {
  as_sandbox env -C / RUSTUP_HOME="${RUST_DIR}/rustup" "${RUST_DIR}/cargo/bin/cargo" --version
}

in_group() { id -nG "$1" 2>/dev/null | tr ' ' '\n' | grep -qx "$2"; }
stat_is() { [[ "$(stat -c '%a %U %G' "$1" 2>/dev/null)" == "$2" ]]; }
in_no_forbidden_group() {
  local g
  id -u "${SANDBOX_USER}" || return 1
  for g in "${FORBIDDEN_GROUPS[@]}"; do in_group "${SANDBOX_USER}" "${g}" && return 1; done
  return 0
}
rust_tree_locked() {
  # Every file root-owned, world-readable, writable by nobody else.
  [[ -d "${RUST_DIR}/cargo" && -d "${RUST_DIR}/rustup" ]] || return 1
  [[ -z "$(find "${RUST_DIR}" ! -type l \( ! -user root -o -perm /022 -o ! -perm -o+r \) -print -quit)" ]]
}
# sandbox_denied CMD... -- the drop to nemosub works, and CMD fails under it
# (so a broken sudo cannot read as a PASS).
sandbox_denied() { as_sandbox true && ! as_sandbox "$@"; }
# shellcheck disable=SC2016  # $0 expands in the inner bash, not here
TCP_CONNECT=(timeout 5 bash -c 'exec 3<>"/dev/tcp/$0/443"')
no_sandbox_process() { id -u "${SANDBOX_USER}" && ! pgrep -u "${SANDBOX_USER}"; }
sudoers_ok() { stat_is "${SUDOERS_FILE}" "440 root root" && grep -qxF "${SUDOERS_LINE}" "${SUDOERS_FILE}"; }
# shellcheck disable=SC2016  # $0 expands in the inner bash, not here
UDP_SEND=(timeout 5 bash -c 'exec 3>"/dev/udp/$0/53" && printf x >&3')
runner_home() { getent passwd "${RUNNER_USER}" | cut -d: -f6; }
# runner_dir -- the runner install: RUNNER_DIR, else <runner home>/actions-runner.
# It must sit directly under the runner home (setup_work_traverse denies every
# other entry there).
runner_dir() {
  local d="${RUNNER_DIR:-$(runner_home)/actions-runner}"
  [[ "$(basename -- "${d}")" != "." && "$(basename -- "${d}")" != ".." \
     && "$(dirname -- "${d}")" == "$(runner_home)" ]] \
    || fail "RUNNER_DIR ${d} is not directly under ${RUNNER_USER}'s home $(runner_home)"
  echo "${d}"
}
runner_work() { echo "$(runner_dir)/_work"; }
# serve_log_dir -- the resident log dir runner's jobs use: SERVE_UP_LOG_DIR from
# the runner's .env (the environment every job inherits), else the default.
serve_log_dir() {
  local v
  v="$(sed -n 's/^SERVE_UP_LOG_DIR=//p' "$(runner_dir)/.env" 2>/dev/null | tail -1)"
  v="${v%\"}"; v="${v#\"}"
  echo "${v:-${DEFAULT_SERVE_LOG}}"
}
# sandbox_cannot_read PATH -- nemosub can neither list/read PATH nor pass through it.
sandbox_cannot_read() { sandbox_denied test -r "$1" -o -x "$1"; }
# no_sandbox_files -- nothing nemosub owns in any wipe-list directory or its home.
no_sandbox_files() {
  local d uid
  [[ -f "${WIPE_LIST}" ]] || return 1
  uid="$(id -u "${SANDBOX_USER}")" || return 1
  while IFS= read -r d; do
    [[ -d "${d}" ]] || continue
    [[ -z "$(as_sandbox find "${d}" -xdev -mindepth 1 -uid "${uid}" -print -quit 2>/dev/null)" ]] || return 1
  done < <(grep '^/' "${WIPE_LIST}"; echo "${SANDBOX_HOME}")
}
# loopback_tcp_denied -- a listener we own on 127.0.0.1 that we CAN reach and
# the sandbox user cannot, so "nothing is listening" never reads as a PASS.
loopback_tcp_denied() {
  local port rc=1
  coproc LISTEN { exec python3 -c 'import socket,time
l = socket.socket(); l.bind(("127.0.0.1", 0)); l.listen(8)
print(l.getsockname()[1], flush=True); time.sleep(30)'; }
  read -r -t 5 port <&"${LISTEN[0]}" || port=""
  # shellcheck disable=SC2016  # $0/$1 expand in the inner bash
  if [[ -n "${port}" ]] && timeout 5 bash -c 'exec 3<>"/dev/tcp/$0/$1"' 127.0.0.1 "${port}"; then
    sandbox_denied timeout 5 bash -c 'exec 3<>"/dev/tcp/$0/$1"' 127.0.0.1 "${port}" && rc=0
  fi
  kill "${LISTEN_PID}" 2>/dev/null; wait "${LISTEN_PID}" 2>/dev/null
  return "${rc}"
}
# crontab_denied -- cron refuses the user outright (cron.deny), not merely "no crontab".
crontab_denied() {
  command -v crontab >/dev/null 2>&1 || return 0
  local out
  out="$(as_sandbox crontab -l 2>&1)" && return 1
  grep -qiE 'not allowed|denied|permission' <<<"${out}"
}
no_at_jobs() { ! command -v atq >/dev/null 2>&1 || [[ -z "$(atq 2>/dev/null | awk -v u="${SANDBOX_USER}" '$NF == u')" ]]; }
no_linger() {
  [[ ! -e "/var/lib/systemd/linger/${SANDBOX_USER}" ]] \
    && [[ "$(loginctl show-user "${SANDBOX_USER}" -p Linger --value 2>/dev/null)" != yes ]]
}
in_deny_file() { grep -qxF "${SANDBOX_USER}" "$1" 2>/dev/null; }
# runner_home_default_deny -- the runner home carries default:user:SANDBOX_USER:---
# so new entries (for example ~/.cache) inherit a deny without re-running setup.
runner_home_default_deny() {
  # Plain getfacl prints default entries with the default: prefix (getfacl -d omits it).
  getfacl -p "$(runner_home)" 2>/dev/null | grep -qxF "default:user:${SANDBOX_USER}:---"
}

# ----------------------------------------------------------------------------
# --check
# ----------------------------------------------------------------------------
fails=0
check() { # check DESCRIPTION CMD...
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "PASS  ${desc}"; else echo "FAIL  ${desc}"; fails=$((fails + 1)); fi
}
warn() { echo "WARN  $*"; }

run_check() {
  if [[ "${IS_ROOT}" -eq 0 && "$(id -un)" != "${RUNNER_USER}" ]]; then
    fail "--check runs as ${RUNNER_USER} or root (it tests ${RUNNER_USER}'s sudo to ${SANDBOX_USER}), not as $(id -un)"
  fi
  local uid
  uid="$(id -u "${SANDBOX_USER}" 2>/dev/null || true)"

  check "user ${SANDBOX_USER} exists" test -n "${uid}"
  check "${SANDBOX_USER} is a system user (uid ${uid:-none} < 1000)" test "${uid:-1000}" -lt 1000
  check "${SANDBOX_USER} shell is /usr/sbin/nologin" \
    test "$(getent passwd "${SANDBOX_USER}" | cut -d: -f7)" = /usr/sbin/nologin
  check "${SANDBOX_USER} home is ${SANDBOX_HOME}" \
    test "$(getent passwd "${SANDBOX_USER}" | cut -d: -f6)" = "${SANDBOX_HOME}"
  check "${SANDBOX_HOME} is ${SANDBOX_USER}-owned, mode 0700" stat_is "${SANDBOX_HOME}" "700 ${SANDBOX_USER} ${SANDBOX_USER}"
  check "${SANDBOX_USER} is in none of: ${FORBIDDEN_GROUPS[*]}" in_no_forbidden_group
  check "group ${SHARE_GROUP} exists" getent group "${SHARE_GROUP}"
  check "${RUNNER_USER} is in ${SHARE_GROUP}" in_group "${RUNNER_USER}" "${SHARE_GROUP}"
  check "${SANDBOX_USER} is in ${SHARE_GROUP}" in_group "${SANDBOX_USER}" "${SHARE_GROUP}"

  check "${SANDBOX_ROOT} is ${RUNNER_USER}:${SHARE_GROUP}, mode 2750" stat_is "${SANDBOX_ROOT}" "2750 ${RUNNER_USER} ${SHARE_GROUP}"
  check "${SANDBOX_USER} can traverse and read ${SANDBOX_ROOT}" as_sandbox test -r "${SANDBOX_ROOT}" -a -x "${SANDBOX_ROOT}"
  check "${SANDBOX_USER} cannot write ${SANDBOX_ROOT}" sandbox_denied test -w "${SANDBOX_ROOT}"

  if [[ "${IS_ROOT}" -eq 1 ]]; then
    check "${SUDOERS_FILE} is root:root 0440 and holds '${SUDOERS_LINE}'" sudoers_ok
    check "${SUDOERS_FILE} passes visudo -c" visudo -cf "${SUDOERS_FILE}"
  else
    warn "${SUDOERS_FILE} is readable by root only; its effect is checked below"
  fi
  check "${RUNNER_USER} can run 'sudo -n -u ${SANDBOX_USER}' without a password" as_sandbox true
  check "${RUNNER_USER} can pass an environment variable through (SETENV)" \
    test "$(as_runner env SANDBOX_CHECK_VAR=ok sudo -n -u "${SANDBOX_USER}" --preserve-env=SANDBOX_CHECK_VAR -- printenv SANDBOX_CHECK_VAR 2>/dev/null)" = ok
  # run-engine-sandboxed.sh hands the adapter its per-job HOME and TMPDIR this way.
  check "${RUNNER_USER} can set HOME and TMPDIR for ${SANDBOX_USER} (via env inside sudo)" \
    test "$(as_runner sudo -n -u "${SANDBOX_USER}" -- env HOME=/sandbox-check TMPDIR=/sandbox-check printenv HOME TMPDIR 2>/dev/null | paste -sd' ' -)" = "/sandbox-check /sandbox-check"
  check "no ${SANDBOX_USER} process is running" no_sandbox_process

  local want got
  want="$(runner_sh 'cargo --version' 2>/dev/null || true)"
  got="$(sandbox_cargo_version 2>/dev/null || true)"
  check "${RUST_DIR} is root-owned and world-readable, writable by root only" rust_tree_locked
  check "cargo as ${SANDBOX_USER} ('${got}') matches ${RUNNER_USER}'s ('${want}')" test -n "${got}" -a "${got}" = "${want}"

  if [[ "${IS_ROOT}" -eq 1 ]]; then
    check "iptables: ${EGRESS_RULE[*]}" iptables -C "${EGRESS_RULE[@]}"
    check "ip6tables: ${EGRESS_RULE[*]}" ip6tables -C "${EGRESS_RULE[@]}"
    check "${EGRESS_UNIT##*/} is enabled (re-applies the egress block at boot)" \
      systemctl is-enabled --quiet "${EGRESS_UNIT##*/}"
  else
    warn "iptables/ip6tables rules are readable by root only; egress is checked below"
    warn "${EGRESS_UNIT##*/} is checked as root only; egress probes are checked above"
  fi
  check "${SANDBOX_USER} cannot open a TCP connection to ${PUBLIC_V4}:443" sandbox_denied "${TCP_CONNECT[@]}" "${PUBLIC_V4}"
  check "${SANDBOX_USER} cannot open a TCP connection to [${PUBLIC_V6}]:443" sandbox_denied "${TCP_CONNECT[@]}" "${PUBLIC_V6}"
  check "${SANDBOX_USER} cannot open a TCP connection to a live listener on 127.0.0.1" loopback_tcp_denied
  check "${SANDBOX_USER} cannot send UDP to 127.0.0.53:53 (the stub resolver)" sandbox_denied "${UDP_SEND[@]}" 127.0.0.53
  check "${SANDBOX_USER} cannot resolve a name (getent hosts example.com)" sandbox_denied timeout 10 getent hosts example.com
  if [[ "${IS_ROOT}" -eq 1 ]]; then
    check "${ACL_UNIT##*/} is enabled (re-applies the resolver ACL at boot)" systemctl is-enabled --quiet "${ACL_UNIT##*/}"
  else
    warn "${ACL_UNIT##*/} is checked as root only; its effect (no name resolution) is checked above"
  fi

  local f
  for f in "${DENY_FILES[@]}"; do check "${SANDBOX_USER} is listed in ${f}" in_deny_file "${f}"; done
  check "cron refuses ${SANDBOX_USER} ('crontab -l' is not allowed, not merely empty)" crontab_denied
  if [[ "${IS_ROOT}" -eq 1 ]]; then
    check "no crontab file for ${SANDBOX_USER}" test ! -e "/var/spool/cron/crontabs/${SANDBOX_USER}"
    check "no at job queued for ${SANDBOX_USER}" no_at_jobs
  else
    warn "the cron spool and the at queue are readable by root only; cron.deny/at.deny are checked above"
  fi
  check "${SANDBOX_USER} does not linger (no user manager outlives kill -1)" no_linger

  local work home p
  work="$(runner_work)"
  home="$(runner_home)"
  if [[ -d "${work}" ]]; then
    check "${SANDBOX_USER} can traverse ${work} (the workflow grants read on the checkout)" as_sandbox test -x "${work}"
  else
    warn "${work} does not exist yet (no job has run); the workflow grants ${SANDBOX_USER} its ACLs per job"
  fi
  check "runner home ${home} has default ACL default:user:${SANDBOX_USER}:--- (new entries inherit deny)" runner_home_default_deny
  for p in "$(runner_dir)/.env" "$(runner_dir)/_diag" "${home}/.cache"; do
    [[ -e "${p}" ]] && check "${SANDBOX_USER} cannot read ${p}" sandbox_cannot_read "${p}"
  done

  check "wipe list ${WIPE_LIST} exists (tools/sandbox-wipe.sh refuses to run without it)" test -f "${WIPE_LIST}"
  check "wipe list ${WIPE_LIST} is root:root 0644" stat_is "${WIPE_LIST}" "644 root root"
  check "nothing ${SANDBOX_USER}-owned in any wipe-list directory or ${SANDBOX_HOME}" no_sandbox_files

  p="$(serve_log_dir)"
  check "resident log dir ${p} is ${RUNNER_USER}:${RUNNER_USER} 0750" stat_is "${p}" "750 ${RUNNER_USER} ${RUNNER_USER}"
  check "${SANDBOX_USER} cannot read the resident log dir ${p}" sandbox_cannot_read "${p}"

  if [[ "${fails}" -gt 0 ]]; then
    echo "setup-sandbox-user.sh --check: ${fails} check(s) FAILED" >&2
    return 1
  fi
  echo "setup-sandbox-user.sh --check: all checks passed"
}

# ----------------------------------------------------------------------------
# setup (root)
# ----------------------------------------------------------------------------
setup_users() {
  getent group "${SHARE_GROUP}" >/dev/null || { groupadd --system "${SHARE_GROUP}"; info "created group ${SHARE_GROUP}"; }
  if ! id "${SANDBOX_USER}" >/dev/null 2>&1; then
    useradd --system --user-group --groups "${SHARE_GROUP}" --shell /usr/sbin/nologin \
      --home-dir "${SANDBOX_HOME}" --no-create-home "${SANDBOX_USER}"
    info "created user ${SANDBOX_USER}"
  fi
  [[ "$(getent passwd "${SANDBOX_USER}" | cut -d: -f6,7)" == "${SANDBOX_HOME}:/usr/sbin/nologin" ]] \
    || usermod --shell /usr/sbin/nologin --home "${SANDBOX_HOME}" "${SANDBOX_USER}"
  in_group "${SANDBOX_USER}" "${SHARE_GROUP}" || usermod --append --groups "${SHARE_GROUP}" "${SANDBOX_USER}"
  local g
  for g in "${FORBIDDEN_GROUPS[@]}"; do
    if in_group "${SANDBOX_USER}" "${g}"; then gpasswd -d "${SANDBOX_USER}" "${g}"; info "removed ${SANDBOX_USER} from ${g}"; fi
  done
  # --append: runner keeps every group it has.
  in_group "${RUNNER_USER}" "${SHARE_GROUP}" || { usermod --append --groups "${SHARE_GROUP}" "${RUNNER_USER}"; info "added ${RUNNER_USER} to ${SHARE_GROUP}"; }
}

setup_dirs() {
  mkdir -p "${SANDBOX_HOME}" "${SANDBOX_ROOT}"
  chown "${SANDBOX_USER}:${SANDBOX_USER}" "${SANDBOX_HOME}"
  chmod 0700 "${SANDBOX_HOME}"
  chown "${RUNNER_USER}:${SHARE_GROUP}" "${SANDBOX_ROOT}"
  chmod 2750 "${SANDBOX_ROOT}"
}

setup_sudoers() {
  sudoers_ok && return 0
  # sudo skips names containing a dot, so the temp file is inert until the rename.
  local tmp
  tmp="$(mktemp "$(dirname "${SUDOERS_FILE}")/.${SUDOERS_NAME}.XXXXXX")"
  printf '%s\n' "${SUDOERS_LINE}" > "${tmp}"
  chmod 0440 "${tmp}"
  visudo -cf "${tmp}" >/dev/null || { rm -f "${tmp}"; fail "visudo rejected the sudoers drop-in; nothing installed"; }
  mv -f "${tmp}" "${SUDOERS_FILE}"
  info "installed ${SUDOERS_FILE}"
}

setup_rust() {
  local active toolchain
  active="$(runner_sh 'rustup show active-toolchain' 2>/dev/null | awk 'NR==1 {print $1}' || true)"
  case "${active}" in
    "") toolchain=stable ;;
    # A channel name drifts; pin the version runner actually has.
    stable*) toolchain="$(runner_sh 'rustc --version' 2>/dev/null | awk '{print $2}' || true)" ;;
    *) toolchain="${active}" ;;
  esac
  [[ -n "${toolchain}" ]] || fail "${RUNNER_USER} has rustup toolchain '${active}' but no working rustc"
  info "Rust toolchain for ${SANDBOX_USER}: ${toolchain} (${RUNNER_USER}'s active: ${active:-none})"

  local -a env=(env RUSTUP_HOME="${RUST_DIR}/rustup" CARGO_HOME="${RUST_DIR}/cargo")
  umask 022
  if [[ -x "${RUST_DIR}/cargo/bin/rustup" ]]; then
    "${env[@]}" "${RUST_DIR}/cargo/bin/rustup" toolchain install "${toolchain}" --profile minimal
    "${env[@]}" "${RUST_DIR}/cargo/bin/rustup" default "${toolchain}"
  else
    local triple tmp
    triple="$(uname -m)-unknown-linux-gnu"
    tmp="$(mktemp -d)"
    if ! curl -fsSL --proto '=https' --tlsv1.2 -o "${tmp}/rustup-init" "https://static.rust-lang.org/rustup/dist/${triple}/rustup-init" \
       || ! curl -fsSL --proto '=https' --tlsv1.2 -o "${tmp}/rustup-init.sha256" "https://static.rust-lang.org/rustup/dist/${triple}/rustup-init.sha256"; then
      rm -rf "${tmp}"; fail "could not download rustup-init for ${triple}"
    fi
    [[ "$(sha256sum "${tmp}/rustup-init" | awk '{print $1}')" == "$(awk '{print $1}' "${tmp}/rustup-init.sha256")" ]] \
      || { rm -rf "${tmp}"; fail "rustup-init does not match its published sha256"; }
    chmod 0755 "${tmp}/rustup-init"
    "${env[@]}" "${tmp}/rustup-init" -y --no-modify-path --profile minimal --default-toolchain "${toolchain}"
    rm -rf "${tmp}"
  fi
  chown -R root:root "${RUST_DIR}"
  chmod -R a+rX,go-w "${RUST_DIR}"
}

# ensure_egress_rules -- idempotent iptables/ip6tables block for nemosub.
ensure_egress_rules() {
  local tool
  for tool in iptables ip6tables; do
    # Migration: the older rule let loopback through; remove every copy.
    while "${tool}" -C "${OLD_EGRESS_RULE[@]}" 2>/dev/null; do
      "${tool}" -D "${OLD_EGRESS_RULE[@]}"
      info "removed ${tool} ${OLD_EGRESS_RULE[*]}"
    done
    "${tool}" -C "${EGRESS_RULE[@]}" 2>/dev/null && continue
    # Inserted first: an appended rule sits behind any earlier ACCEPT (ufw's
    # conntrack NEW accept, for one) and would never be reached.
    "${tool}" -I "${EGRESS_RULE[0]}" 1 "${EGRESS_RULE[@]:1}"
    info "added ${tool} ${EGRESS_RULE[*]}"
  done
}

setup_egress() {
  local ipt_bin ip6_bin
  ipt_bin="$(command -v iptables)"
  ip6_bin="$(command -v ip6tables)"
  ensure_egress_rules
  setup_egress_unit "${ipt_bin}" "${ip6_bin}"
  if command -v netfilter-persistent >/dev/null 2>&1; then
    netfilter-persistent save
  fi
}

# The egress rules live in the kernel, not on disk: without iptables-persistent
# they vanish at reboot. The oneshot unit re-applies them at boot; --check
# (run by every ranked preflight) proves egress is blocked, so a lost rule
# fails closed.
setup_egress_unit() {
  local ipt_bin="$1" ip6_bin="$2" tmp script
  # The rules go into a small root-owned script the unit runs by path: systemd
  # does not parse quoting the way bash does, so no rule text is embedded in
  # the unit file itself.
  script="${SANDBOX_ETC}/egress.sh"
  install -d -m 0755 -o root -g root "$(dirname -- "${script}")"
  tmp="$(mktemp)"
  {
    echo "#!/bin/sh"
    echo "# Installed by tools/box-runner/setup-sandbox-user.sh: reject all ${SANDBOX_USER} egress."
    echo "set -e"
    for bin in "${ipt_bin}" "${ip6_bin}"; do
      echo "while ${bin} -C ${OLD_EGRESS_RULE[*]} 2>/dev/null; do ${bin} -D ${OLD_EGRESS_RULE[*]}; done"
      # Always first in OUTPUT: delete every copy, then insert at position 1, so no
      # ACCEPT rule (ufw's or anyone's) can sit in front of it.
      echo "while ${bin} -C ${EGRESS_RULE[*]} 2>/dev/null; do ${bin} -D ${EGRESS_RULE[*]}; done"
      echo "${bin} -I ${EGRESS_RULE[0]} 1 ${EGRESS_RULE[*]:1}"
    done
  } > "${tmp}"
  install -m 0755 -o root -g root "${tmp}" "${script}"
  {
    echo "# Installed by tools/box-runner/setup-sandbox-user.sh (docs/sandbox-design.md)."
    echo "[Unit]"
    echo "Description=Reject all ${SANDBOX_USER} egress (re-applied every boot)"
    echo "After=network-pre.target ufw.service"
    echo "Before=nemotron35-box-runner.service"
    echo "[Service]"
    echo "Type=oneshot"
    echo "RemainAfterExit=yes"
    echo "ExecStart=${script}"
    echo "[Install]"
    echo "WantedBy=multi-user.target"
  } > "${tmp}"
  if ! cmp -s "${tmp}" "${EGRESS_UNIT}"; then
    install -m 0644 -o root -g root "${tmp}" "${EGRESS_UNIT}"
    systemctl daemon-reload
    info "installed ${EGRESS_UNIT}"
  fi
  rm -f "${tmp}"
  systemctl enable "${EGRESS_UNIT##*/}" >/dev/null 2>&1 || fail "could not enable ${EGRESS_UNIT##*/}"
  systemctl restart "${EGRESS_UNIT##*/}" || fail "${EGRESS_UNIT##*/} failed; see journalctl -u ${EGRESS_UNIT##*/}"
}

# The resolver sockets live on /run, a tmpfs: an ACL set there is gone at the
# next boot (and when the service recreates its socket). The oneshot unit
# re-applies it at boot; --check (run by every ranked preflight) proves name
# resolution fails, so a lost ACL fails closed.
setup_resolver_acl() {
  local sock setfacl_bin tmp
  setfacl_bin="$(command -v setfacl)"
  tmp="$(mktemp)"
  {
    echo "# Installed by tools/box-runner/setup-sandbox-user.sh (docs/sandbox-design.md)."
    echo "[Unit]"
    echo "Description=Deny ${SANDBOX_USER} the host resolver sockets (/run is tmpfs; re-applied every boot)"
    echo "After=systemd-resolved.service nscd.service"
    echo "[Service]"
    echo "Type=oneshot"
    echo "RemainAfterExit=yes"
    # '-': a socket this box does not have is not an error.
    for sock in "${RESOLVER_SOCKETS[@]}"; do echo "ExecStart=-${setfacl_bin} -m u:${SANDBOX_USER}:--- ${sock}"; done
    echo "[Install]"
    echo "WantedBy=multi-user.target"
  } > "${tmp}"
  if ! cmp -s "${tmp}" "${ACL_UNIT}"; then
    install -m 0644 -o root -g root "${tmp}" "${ACL_UNIT}"
    systemctl daemon-reload
    info "installed ${ACL_UNIT}"
  fi
  rm -f "${tmp}"
  systemctl enable "${ACL_UNIT##*/}" >/dev/null 2>&1 || fail "could not enable ${ACL_UNIT##*/}"
  systemctl restart "${ACL_UNIT##*/}" || fail "${ACL_UNIT##*/} failed; see journalctl -u ${ACL_UNIT##*/}"
}

setup_no_persistence() {
  local f
  for f in "${DENY_FILES[@]}"; do
    [[ -e "${f}" ]] || install -m 0644 -o root -g root /dev/null "${f}"
    in_deny_file "${f}" || { echo "${SANDBOX_USER}" >> "${f}"; info "added ${SANDBOX_USER} to ${f}"; }
  done
  if command -v crontab >/dev/null 2>&1; then crontab -r -u "${SANDBOX_USER}" 2>/dev/null || true; fi
  rm -f "/var/spool/cron/crontabs/${SANDBOX_USER}"
  if command -v atq >/dev/null 2>&1; then
    atq | awk -v u="${SANDBOX_USER}" '$NF == u {print $1}' | xargs -r atrm
  fi
  loginctl disable-linger "${SANDBOX_USER}" 2>/dev/null || true
}

# Traverse only (--x): nemosub can pass through runner's home toward a job
# checkout, and list or read nothing on the way. --x alone would still open any
# world-readable entry whose name nemosub guesses, so every entry beside the
# path to the checkout gets an explicit deny. Under _work the job's checkout
# dir is not known here; the runner's own dirs there are all _-prefixed. The
# workflow grants rX on the checkout itself and repeats the deny per job.
#
# A default deny on the runner home covers entries created later (for example
# ~/.cache after a job): new subdirectories inherit default:user:${SANDBOX_USER}:---
# (side effect a). The job checkout is unaffected: benchmark.yml grants rX on the
# workspace recursively and sets default:user:${SANDBOX_USER}:rX on it per job
# (.github/workflows/benchmark.yml setfacl -R -P -m / -R -P -d -m). Side effect
# (b): inside a directory with a default ACL the process umask is not applied to
# new entries; the requested mode is limited by the default ACL's base entries,
# which come from the home's own mode. So the runner's own group can read new
# entries if the home grants the group read. This script does not set the home's
# mode.
setup_work_traverse() {
  local home dir d
  home="$(runner_home)"
  dir="$(runner_dir)"
  for d in "${home}" "${dir}" "${dir}/_work"; do
    [[ -d "${d}" ]] && setfacl -m "u:${SANDBOX_USER}:--x" "${d}"
  done
  # -n: keep any existing default mask (setfacl -m recalculates it, which can widen
  # access for other named entries in an existing default ACL).
  setfacl -n -d -m "u:${SANDBOX_USER}:---" "${home}"
  deny_entries "${home}" ! -name "$(basename -- "${dir}")"
  deny_entries "${dir}" ! -name _work
  deny_entries "${dir}/_work" -name '_*'
}

# deny_entries DIR FIND-TESTS... -- u:nemosub:--- on each entry directly under
# DIR that matches (symlinks skipped: setfacl would change their target).
deny_entries() {
  local dir="$1"; shift
  [[ -d "${dir}" ]] || return 0
  find "${dir}" -mindepth 1 -maxdepth 1 ! -type l "$@" -exec setfacl -m "u:${SANDBOX_USER}:---" {} \;
}

# The wipe list: every directory nemosub can write, found once here (a full
# scan is too slow for every job). World-writable directories on each local
# filesystem, plus /dev/shm and /dev/mqueue, kept only if nemosub can in fact
# write and enter them.
setup_wipe_list() {
  local m d tmp
  local -a found=()
  while IFS= read -r m; do
    while IFS= read -r -d '' d; do
      [[ "${d}" == *$'\n'* ]] || found+=("${d}")
    done < <(find "${m}" -xdev \( -path /proc -o -path /sys -o -path /var/lib/docker -o -path /var/lib/containerd \) -prune \
               -o -type d -perm -0002 -print0 2>/dev/null)
  done < <(findmnt -rn -o TARGET -t "${WIPE_FS_TYPES}")
  found+=("${WIPE_ALWAYS[@]}")
  tmp="$(mktemp)"
  # shellcheck disable=SC2016  # $d expands in the inner shell
  as_sandbox bash -c 'for d; do [[ -d "$d" && -w "$d" && -x "$d" ]] && printf "%s\n" "$d"; done; exit 0' _ "${found[@]}" \
    | sort -u > "${tmp}"
  [[ -s "${tmp}" ]] || { rm -f "${tmp}"; fail "found no directory ${SANDBOX_USER} can write (not even /tmp); refusing an empty wipe list"; }
  # Its own root-owned 0755 directory, never the App-key directory
  # (/etc/nemotron35-box-runner stays root-only).
  install -d -m 0755 -o root -g root "$(dirname "${WIPE_LIST}")"
  install -m 0644 -o root -g root "${tmp}" "${WIPE_LIST}"
  rm -f "${tmp}"
  info "recorded $(wc -l < "${WIPE_LIST}") directories ${SANDBOX_USER} can write in ${WIPE_LIST}"
}

# The resident log dir holds every leg's resident log; a candidate's may carry
# the hidden prompts. runner-only, and nothing already there readable by others.
setup_serve_log() {
  local d
  d="$(serve_log_dir)"
  [[ "${d}" == /* ]] || fail "SERVE_UP_LOG_DIR '${d}' in $(runner_dir)/.env is not an absolute path"
  mkdir -p "${d}"
  chown "${RUNNER_USER}:${RUNNER_USER}" "${d}"
  chmod 0750 "${d}"
  chmod -R o-rwx "${d}"
  info "resident log dir ${d}: ${RUNNER_USER}:${RUNNER_USER} 0750, no access for others"
}

case "${1:-}" in
  --check) runner_dir >/dev/null; run_check ;;
  "")
    [[ "${IS_ROOT}" -eq 1 ]] || fail "setup runs as root (use --check to verify as ${RUNNER_USER})"
    id "${RUNNER_USER}" >/dev/null 2>&1 || fail "user ${RUNNER_USER} does not exist; set up the box runner first"
    runner_dir >/dev/null
    for tool in useradd usermod groupadd gpasswd visudo runuser iptables ip6tables curl sha256sum systemctl loginctl python3 install findmnt; do
      command -v "${tool}" >/dev/null 2>&1 || fail "${tool} is required"
    done
    command -v setfacl >/dev/null 2>&1 || fail "setfacl is required (install the acl package: apt-get install acl)"
    setup_users
    setup_dirs
    setup_sudoers
    setup_rust
    setup_egress
    setup_resolver_acl
    setup_no_persistence
    setup_work_traverse
    setup_wipe_list
    setup_serve_log
    run_check
    ;;
  *) fail "usage: $0 [--check]" ;;
esac
