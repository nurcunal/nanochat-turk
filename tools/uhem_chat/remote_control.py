"""Private UHeM job control for the downloadable DSAI SFT launcher.

Run with ``python3 -c <this file's text>`` and one JSON request on stdin.
Only the two packaged runtime files may be deployed. Standard output is one
JSON response; Slurm output is captured rather than forwarded.
"""

import base64
import datetime
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile


MODEL_ROOT = Path("/ari/users/nunal/nanochat-turk-d32-fineweb2-bpe32k")
CONTROL_ROOT = MODEL_ROOT / "control" / "dsai-uhem-launcher-v1"
RUNTIME_FILES = ("chat_web.py", "launch_chat.sbatch")
HISTORICAL_LAUNCHER_SHA256 = (
    "7a737bb0a5ff77a69f6c38865ea31ec955b6392a32657a5e9d82b756b061f5c0"
)
ACTIVE_STATES = {"RUNNING", "PENDING", "CONFIGURING"}
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RUNTIME_BYTES = 1024 * 1024


def run_command(arguments, env=None, allow_failure=False):
    result = subprocess.run(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=env,
        timeout=45,
    )
    if result.returncode and not allow_failure:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError("{} failed: {}".format(arguments[0], detail))
    return result


def valid_job_id(value):
    value = str(value or "")
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError("A numeric Slurm job ID is required")
    return value


def valid_sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected lowercase SHA-256 is required")
    return value


def file_sha256(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular, non-symlink file: {}".format(path))
    if path.stat().st_uid != os.getuid():
        raise ValueError("File is not owned by the connected UHeM user")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private_directory(path):
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir() or path.stat().st_uid != os.getuid():
        raise ValueError("Control directory must be owned and not a symlink")
    path.chmod(0o700)
    return path


def queue_jobs():
    result = run_command(["squeue", "--me", "--noheader", "--format=%i|%T"])
    jobs = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|", 1)
        if len(fields) == 2 and re.fullmatch(r"[1-9][0-9]*", fields[0]):
            jobs[fields[0]] = fields[1].strip()
    return jobs


def parse_job_record(text):
    # Slurm's one-line format can contain spaces in Command/WorkDir paths.
    return dict(
        re.findall(
            r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=(.*?)(?=\s[A-Za-z][A-Za-z0-9_]*=|$)",
            text.strip(),
        )
    )


def owned_job_record(job_id):
    job_id = valid_job_id(job_id)
    result = run_command(["scontrol", "show", "job", job_id, "-o"], allow_failure=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if "Invalid job id" in detail or "Invalid job ID" in detail:
            return None
        raise RuntimeError("Cannot inspect Slurm job {}: {}".format(job_id, detail[-2000:]))
    record = parse_job_record(result.stdout)
    if record.get("JobId") != job_id:
        raise ValueError("Slurm returned a different job identity")
    owner = re.search(r"\(([0-9]+)\)$", record.get("UserId", ""))
    if not owner or int(owner.group(1)) != os.getuid():
        raise ValueError("Job is not owned by the connected UHeM user")
    return record


def job_launcher_hash(record):
    command = record.get("Command", "")
    if not command.startswith("/"):
        return None
    try:
        return file_sha256(Path(command))
    except (OSError, ValueError):
        return None


def receipt_path(job_id):
    return CONTROL_ROOT / "jobs" / (valid_job_id(job_id) + ".json")


def load_receipt(job_id):
    path = receipt_path(job_id)
    if not path.exists():
        return None
    if path.is_symlink() or path.stat().st_uid != os.getuid():
        raise ValueError("Invalid job lease file")
    if path.stat().st_mode & 0o077:
        raise ValueError("Job lease file must have mode 0600")
    receipt = json.loads(path.read_text())
    if receipt.get("job_id") != valid_job_id(job_id):
        raise ValueError("Job lease identity mismatch")
    valid_sha256(receipt.get("launcher_sha256"))
    return receipt


def known_job(record, expected_launcher=None):
    allowed = {HISTORICAL_LAUNCHER_SHA256}
    if expected_launcher:
        allowed.add(valid_sha256(expected_launcher))
    receipt = load_receipt(record["JobId"])
    if receipt:
        allowed.add(receipt["launcher_sha256"])
    return job_launcher_hash(record) in allowed


def public_job(record, created=False, lease=None):
    node = record.get("NodeList")
    if node in {None, "", "(null)", "None", "Unknown"}:
        node = None
    elif not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", node):
        raise ValueError("Expected a single allocated compute node")
    end_time = record.get("EndTime")
    if end_time in {None, "", "Unknown", "N/A", "None"}:
        end_time = None
    command = record.get("Command", "")
    return {
        "job_id": record["JobId"],
        "created": created,
        "lease": lease,
        "status": record.get("JobState", "UNKNOWN"),
        "node": node,
        "end_time": end_time,
        "control_dir": str(Path(command).parent) if command.startswith("/") else None,
    }


def validate_bundle(payload):
    bundle = payload.get("bundle")
    expected = payload.get("expected")
    if not isinstance(bundle, dict) or set(bundle) != set(RUNTIME_FILES):
        raise ValueError("Bundle must contain exactly the two runtime files")
    if not isinstance(expected, dict) or set(expected) != set(RUNTIME_FILES):
        raise ValueError("Both runtime SHA-256 hashes are required")
    decoded = {}
    for name in RUNTIME_FILES:
        valid_sha256(expected[name])
        raw = base64.b64decode(bundle[name], validate=True)
        if len(raw) > MAX_RUNTIME_BYTES:
            raise ValueError("Runtime file is too large")
        if hashlib.sha256(raw).hexdigest() != expected[name]:
            raise ValueError("Runtime checksum mismatch: {}".format(name))
        decoded[name] = raw
    return decoded, expected


def deploy_bundle(decoded, expected):
    bundle_hash = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    target = CONTROL_ROOT / ("bundle-" + bundle_hash)
    if not target.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".deploy-", dir=str(CONTROL_ROOT)))
        try:
            for name in RUNTIME_FILES:
                path = temporary / name
                with path.open("xb") as handle:
                    handle.write(decoded[name])
                path.chmod(0o600)
            temporary.rename(target)
        finally:
            if temporary.exists():
                shutil.rmtree(str(temporary))
    if target.is_symlink() or not target.is_dir() or target.stat().st_uid != os.getuid():
        raise ValueError("Invalid immutable deployment directory")
    for name in RUNTIME_FILES:
        if file_sha256(target / name) != expected[name]:
            raise ValueError("Existing immutable bundle differs; refusing overwrite")
    return target


def save_new_receipt(receipt):
    private_directory(CONTROL_ROOT / "jobs")
    destination = receipt_path(receipt["job_id"])
    descriptor, temporary = tempfile.mkstemp(prefix=".lease-", dir=str(destination.parent))
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(receipt, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic publication without replacing another lease for the same ID.
        os.link(temporary, str(destination))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def submit_job(control_dir, launcher_hash):
    environment = dict(os.environ)
    environment.update(
        DSAI_CHAT_CONTROL=str(control_dir),
        EXPECTED_LAUNCH_SHA256=launcher_hash,
    )
    result = run_command(
        [
            "sbatch", "--parsable",
            "--account=nakane", "--partition=a100q", "--nodes=1", "--ntasks=1",
            "--cpus-per-task=8", "--gres=gpu:a100:1", "--mem=64G",
            "--time=01:00:00", "--no-requeue", "--export=ALL",
            "--output=" + str(control_dir / "slurm-%j.out"),
            "--error=" + str(control_dir / "slurm-%j.err"),
            str(control_dir / "launch_chat.sbatch"),
        ],
        env=environment,
    )
    ids = []
    for line in result.stdout.splitlines():
        match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", line.strip())
        if match:
            ids.append(match.group(1))
    if len(ids) != 1:
        raise RuntimeError("Cannot identify submitted job from sbatch response: " + result.stdout[-1000:])
    return ids[0]


def find_or_start(payload):
    decoded, expected = validate_bundle(payload)
    launcher_hash = expected["launch_chat.sbatch"]
    jobs = queue_jobs()
    requested = payload.get("job_id")
    if requested is not None:
        requested = valid_job_id(requested)
        if jobs.get(requested) not in ACTIVE_STATES:
            raise ValueError("Requested job is not an active job owned by this user")
        candidates = [requested]
    else:
        priority = {"RUNNING": 0, "CONFIGURING": 1, "PENDING": 2}
        candidates = sorted(
            (job_id for job_id, state in jobs.items() if state in ACTIVE_STATES),
            key=lambda job_id: (priority[jobs[job_id]], int(job_id)),
        )
    for job_id in candidates:
        record = owned_job_record(job_id)
        if record and record.get("JobState") in ACTIVE_STATES:
            actual_hash = job_launcher_hash(record)
            if actual_hash in {HISTORICAL_LAUNCHER_SHA256, launcher_hash}:
                return public_job(record)
            if record.get("JobName") in {"dsai-3b-instruct", "tr-d32-eff020-chat"}:
                raise ValueError(
                    "Existing SFT job {} cannot be verified against this launcher; "
                    "refusing to allocate another GPU".format(job_id)
                )
    if requested is not None:
        raise ValueError("Requested job does not use a verified DSAI SFT launcher")
    if payload.get("attach_only"):
        raise ValueError("No matching active SFT session exists; attach-only mode cannot allocate a GPU")
    control_dir = deploy_bundle(decoded, expected)
    job_id = submit_job(control_dir, launcher_hash)
    lease = secrets.token_urlsafe(32)
    receipt = {
        "job_id": job_id,
        "lease": lease,
        "launcher_sha256": launcher_hash,
        "control_dir": str(control_dir),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    try:
        save_new_receipt(receipt)
    except Exception as error:
        # This ID came from this invocation's successful sbatch call only.
        cleanup = run_command(["scancel", job_id], allow_failure=True)
        raise RuntimeError(
            "Could not save lease for newly submitted job {}; cancellation {}: {}".format(
                job_id, "requested" if cleanup.returncode == 0 else "FAILED", error
            )
        )
    inspection_warning = None
    try:
        record = owned_job_record(job_id)
        if record:
            return public_job(record, created=True, lease=lease)
    except Exception as error:
        # Submission and the lease are already durable. Never discard the
        # client's cleanup handle just because Slurm inspection is transiently
        # unavailable or returns an unexpected record immediately afterward.
        inspection_warning = str(error)
    result = {
        "job_id": job_id, "created": True, "lease": lease,
        "status": "PENDING", "node": None, "end_time": None,
        "control_dir": str(control_dir),
    }
    if inspection_warning:
        result["inspection_warning"] = inspection_warning
    return result


def describe(payload):
    job_id = valid_job_id(payload.get("job_id"))
    if job_id not in queue_jobs():
        return {"exists": False, "status": "ENDED", "job_id": job_id,
                "node": None, "end_time": None, "control_dir": None}
    record = owned_job_record(job_id)
    if not record:
        return {"exists": False, "status": "ENDED", "job_id": job_id,
                "node": None, "end_time": None, "control_dir": None}
    result = public_job(record)
    result["exists"] = True
    result.pop("created")
    result.pop("lease")
    return result


def cancel(payload):
    job_id = valid_job_id(payload.get("job_id"))
    lease = payload.get("lease")
    receipt = load_receipt(job_id)
    if not receipt or not isinstance(lease, str) or not hmac.compare_digest(lease, receipt.get("lease", "")):
        raise ValueError("Cancellation requires the exact lease of a job created by this launcher")
    if job_id not in queue_jobs():
        return {"job_id": job_id, "cancelled": False, "status": "ENDED"}
    record = owned_job_record(job_id)
    if not record:
        return {"job_id": job_id, "cancelled": False, "status": "ENDED"}
    if job_launcher_hash(record) != receipt["launcher_sha256"]:
        raise ValueError("Job launcher identity changed; refusing cancellation")
    if str(Path(record.get("Command", "")).parent) != receipt["control_dir"]:
        raise ValueError("Job deployment directory changed; refusing cancellation")
    run_command(["scancel", job_id])
    return {"job_id": job_id, "cancelled": True, "status": "CANCEL_REQUESTED"}


def logs(payload):
    job_id = valid_job_id(payload.get("job_id"))
    record = owned_job_record(job_id)
    if not record:
        return {"job_id": job_id, "stdout": "", "stderr": "", "status": "UNAVAILABLE"}
    expected = payload.get("expected", {}).get("launch_chat.sbatch")
    if not known_job(record, expected):
        raise ValueError("Logs require a verified DSAI SFT launcher")
    control = Path(record["Command"]).parent.resolve()
    result = {"job_id": job_id, "status": record.get("JobState", "UNKNOWN")}
    for field, key in (("StdOut", "stdout"), ("StdErr", "stderr")):
        path = Path(record.get(field, "/nonexistent"))
        if not path.is_file() or path.is_symlink():
            result[key] = ""
            continue
        if os.path.commonpath([str(control), str(path.resolve())]) != str(control):
            raise ValueError("Job log is outside its verified deployment directory")
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - 8192))
            result[key] = "\n".join(handle.read().decode("utf-8", errors="replace").splitlines()[-50:])
    return result


def dispatch(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    action = payload.get("action")
    if action not in {"find_or_start", "describe", "cancel", "logs"}:
        raise ValueError("Unknown action")
    if action in {"find_or_start", "cancel"}:
        private_directory(CONTROL_ROOT)
        lock_path = CONTROL_ROOT / "control.lock"
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return find_or_start(payload) if action == "find_or_start" else cancel(payload)
    return describe(payload) if action == "describe" else logs(payload)


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Request is too large")
        result = dispatch(json.loads(raw.decode("utf-8")))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
