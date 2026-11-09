#!/usr/bin/env python3
"""Open a private UHeM SFT session using the caller's local SSH credentials."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser


MODEL_TAG = "tr_d32_bpe32k_capability_calibration_eff020_v1"
CONTRACT_SHA256 = "8e1ccd717f4a666eed73fd30e3e3e1cec5f273842d99fce895479ecd4cf8befd"
BUNDLE_DIR = Path(__file__).resolve().parent
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LaunchError(RuntimeError):
    pass


def checked(command, **kwargs):
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    if result.returncode:
        raise LaunchError(result.stderr.strip() or result.stdout.strip() or "Command failed")
    return result.stdout


def local_free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def validate_health(health):
    return (
        health.get("ready") is True
        and health.get("source") == "sft"
        and health.get("model_tag") == MODEL_TAG
        and health.get("step") == 32
        and (health.get("chat_contract") or {}).get("contract_sha256") == CONTRACT_SHA256
        and health.get("num_gpus") == 1
    )


def read_health(url):
    with HTTP.open(url + "/health", timeout=5) as response:
        return json.load(response)


def load_bundle():
    manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text())
    for name in ("remote_control.py", "chat_web.py", "launch_chat.sbatch"):
        expected = manifest["files"][name]["sha256"]
        actual = hashlib.sha256((BUNDLE_DIR / name).read_bytes()).hexdigest()
        if actual != expected:
            raise LaunchError(f"Bundle checksum mismatch: {name}. Download the bundle again.")
    names = ("chat_web.py", "launch_chat.sbatch")
    return {
        "bundle": {name: base64.b64encode((BUNDLE_DIR / name).read_bytes()).decode() for name in names},
        "expected": {name: manifest["files"][name]["sha256"] for name in names},
    }


class SSHConnection:
    def __init__(self, host, control_path=None):
        self.host = host
        self.owns_master = control_path is None
        self.temp_dir = tempfile.TemporaryDirectory(prefix="dsai-", dir="/tmp")
        self.control_path = control_path or str(Path(self.temp_dir.name) / "ssh")

    def command(self, *args):
        return [
            "ssh", "-S", self.control_path,
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            *args, self.host,
        ]

    def connect(self):
        if self.owns_master:
            print(f"Connecting to {self.host}. Use your normal UHeM SSH login.", flush=True)
            command = [
                "ssh", "-M", "-S", self.control_path, "-fNT",
                "-o", "ControlPersist=no", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                self.host,
            ]
            # Leave stdin/stderr attached to Terminal for password and host-key prompts.
            if subprocess.run(command).returncode:
                raise LaunchError("SSH login failed. Check the UHeM VPN and your SSH credentials.")
        checked(self.command("-O", "check"), timeout=20)

    def run(self, arguments, *, input=None, timeout=60):
        return checked(self.command() + [shlex.join(arguments)], input=input, timeout=timeout)

    def remote(self, action, **payload):
        source = (BUNDLE_DIR / "remote_control.py").read_text()
        result = subprocess.run(
            self.command() + [shlex.join(["python3", "-c", source])],
            input=json.dumps({"action": action, **payload}),
            text=True, capture_output=True, timeout=90,
        )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise LaunchError(result.stderr.strip() or result.stdout.strip() or "UHeM did not respond") from exc
        if result.returncode or data.get("error"):
            raise LaunchError(data.get("error") or result.stderr.strip() or "UHeM command failed")
        return data

    def close(self):
        if self.owns_master:
            subprocess.run(self.command("-O", "exit"), capture_output=True, timeout=20)
        self.temp_dir.cleanup()


class ChatSession:
    def __init__(self, ssh):
        self.ssh = ssh
        self.job = None
        self.forward = None
        self.relay = None
        self.relay_log = None

    def attach_tunnel(self, node, port=0):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", node):
            raise LaunchError("Unexpected compute-node name")
        choose_port = "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()"
        relay_port = int(self.ssh.run(["python3", "-c", choose_port]).strip())
        if not 1024 <= relay_port <= 65535:
            raise LaunchError("Unexpected relay port")
        local_port = port or local_free_port()
        self.relay_log = tempfile.TemporaryFile(mode="w+t")
        relay_command = [
            "timeout", "3700", "ssh", "-N", "-o", "BatchMode=yes",
            "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-L", f"127.0.0.1:{relay_port}:127.0.0.1:8000", node,
        ]
        self.relay = subprocess.Popen(
            self.ssh.command() + ["exec " + shlex.join(relay_command)],
            stdin=subprocess.DEVNULL, stdout=self.relay_log, stderr=self.relay_log, text=True,
        )
        forward = f"127.0.0.1:{local_port}:127.0.0.1:{relay_port}"
        checked(self.ssh.command("-O", "forward", "-o", "ExitOnForwardFailure=yes", "-L", forward), timeout=20)
        self.forward = forward
        return f"http://127.0.0.1:{local_port}"

    def check_relay(self):
        if self.relay is not None and self.relay.poll() is not None:
            self.relay_log.seek(0)
            detail = self.relay_log.read()[-3000:].strip()
            raise LaunchError(f"The private SSH tunnel stopped. {detail}")

    def close(self):
        # A reused session has no cancellation lease and must remain untouched.
        if self.job and self.job.get("created") and self.job.get("lease"):
            try:
                self.ssh.remote("cancel", job_id=self.job["job_id"], lease=self.job["lease"])
                print(f"Released A100 job {self.job['job_id']}.", flush=True)
            except Exception as exc:
                print(f"Could not confirm job cleanup: {exc}\n"
                      f"If needed, connect to UHeM and run: scancel {self.job['job_id']}\n"
                      "The job also has a one-hour scheduler limit.", file=sys.stderr)
        if self.forward:
            try:
                checked(self.ssh.command("-O", "cancel", "-L", self.forward), timeout=20)
            except Exception:
                pass
        if self.relay is not None:
            self.relay.terminate()
            try:
                self.relay.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.relay.kill()
                self.relay.wait(timeout=5)
        if self.relay_log is not None:
            self.relay_log.close()


def wait_for_node(session, deadline):
    previous = None
    while time.monotonic() < deadline:
        job = session.ssh.remote("describe", job_id=session.job["job_id"])
        status = job.get("status", "UNKNOWN")
        if status != previous:
            print(f"UHeM job {session.job['job_id']}: {status}", flush=True)
            previous = status
        if status == "RUNNING" and job.get("node"):
            return job
        if not job.get("exists", True) or status not in {"PENDING", "CONFIGURING", "RUNNING"}:
            raise LaunchError(f"Job ended before the model was ready ({status}).")
        time.sleep(5)
    raise LaunchError("Timed out waiting for an A100. You can try again later or increase --wait-minutes.")


def wait_for_model(session, url, deadline):
    last_notice = 0
    while time.monotonic() < deadline:
        session.check_relay()
        try:
            health = read_health(url)
        except (OSError, ValueError, urllib.error.URLError):
            health = None
        if health is not None:
            if not validate_health(health):
                raise LaunchError("The server's model identity does not match the expected SFT checkpoint.")
            return health
        job = session.ssh.remote("describe", job_id=session.job["job_id"])
        if not job.get("exists", True) or job.get("status") != "RUNNING":
            raise LaunchError("The UHeM job ended during model startup.")
        if time.monotonic() - last_notice > 30:
            print("Verifying checkpoint and loading the model; this can take a few minutes...", flush=True)
            last_notice = time.monotonic()
        time.sleep(5)
    raise LaunchError("Timed out waiting for model startup.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=os.environ.get("UHEM_HOST", "nunal@altay.uhem.itu.edu.tr"))
    p.add_argument("--control-path", default=os.environ.get("UHEM_CONTROL_PATH"), help="Reuse an authenticated SSH master without closing it")
    p.add_argument("--attach-only", action="store_true", help="Never submit a new GPU job")
    p.add_argument("--job-id", help="Attach to this exact matching user-owned job; never submit")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--verify-only", action="store_true", help="Attach, verify readiness and exit; never submit or cancel a job")
    p.add_argument("--port", type=int, default=0, help="Local port; 0 chooses a free port")
    p.add_argument("--wait-minutes", type=float, default=20)
    p.add_argument("--dry-run", action="store_true", help="Check the local bundle and show the plan without connecting")
    args = p.parse_args(argv)
    if not re.fullmatch(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9_.-]*", args.host):
        p.error("--host must be an SSH alias or user@host")
    if args.job_id and not re.fullmatch(r"[0-9]+", args.job_id):
        p.error("--job-id must be numeric")
    if args.port != 0 and not 1024 <= args.port <= 65535:
        p.error("--port must be 0 or between 1024 and 65535")
    if not 0 < args.wait_minutes <= 1440:
        p.error("--wait-minutes must be between 0 and 1440")
    args.attach_only = args.attach_only or bool(args.job_id) or args.verify_only
    return args


def interrupted(_signum, _frame):
    raise KeyboardInterrupt


def main(argv=None):
    args = parse_args(argv)
    bundle = load_bundle()
    if args.dry_run:
        print(json.dumps({"host": args.host, "model_tag": MODEL_TAG, "step": 32,
                          "attach_only": args.attach_only, "gpu_count": 1,
                          "time_limit": "01:00:00", "access": "private SSH tunnel"}, indent=2))
        return 0
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    ssh = SSHConnection(args.host, args.control_path)
    session = ChatSession(ssh)
    try:
        ssh.connect()
        session.job = ssh.remote("find_or_start", **bundle, attach_only=args.attach_only, job_id=args.job_id)
        if session.job.get("created"):
            print(f"Requested one A100: job {session.job['job_id']} (one-hour limit).", flush=True)
        else:
            print(f"Reusing UHeM job {session.job['job_id']}; no additional GPU requested.", flush=True)
        deadline = time.monotonic() + args.wait_minutes * 60
        job = wait_for_node(session, deadline)
        url = session.attach_tunnel(job["node"], args.port)
        wait_for_model(session, url, deadline)
        print(f"\nDSAI-3B Instruct (SFT trial) is ready: {url}", flush=True)
        if job.get("end_time"):
            print(f"Scheduled end time on UHeM: {job['end_time']}", flush=True)
        if args.verify_only:
            print("Readiness verified. Disconnecting; the existing job remains running.", flush=True)
            return 0
        if not args.no_browser:
            webbrowser.open(url)
        action = "release this A100" if session.job.get("created") else "disconnect (the existing job stays running)"
        print(f"Keep this Terminal window open. Press Control-C to {action}.", flush=True)
        while True:
            time.sleep(15)
            session.check_relay()
            job = ssh.remote("describe", job_id=session.job["job_id"])
            if not job.get("exists", True) or job.get("status") != "RUNNING":
                print("The UHeM session has ended.", flush=True)
                return 0
    except KeyboardInterrupt:
        print("\nDisconnecting...", flush=True)
        return 0
    except (LaunchError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"\nUnable to open the chat: {exc}", file=sys.stderr)
        if session.job:
            print(f"UHeM job: {session.job['job_id']}; logs: {session.job.get('control_dir', 'see scontrol show job')}", file=sys.stderr)
        return 1
    finally:
        session.close()
        try:
            ssh.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
