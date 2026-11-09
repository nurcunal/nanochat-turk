# Try the SFT model on a UHeM A100

[**Download the Mac launcher**](https://github.com/nurcunal/nanochat-turk/releases/download/dsai-uhem-launcher-v1/DSAI-3B-Instruct-macOS.zip)

This launcher opens the surviving DSAI-3B SFT trial checkpoint through your
UHeM account. It connects from your Mac, requests one A100 for up to one hour
when needed, and opens a private browser chat. A matching running or queued
session is reused instead of allocating another GPU.

## Start a session

1. Connect your Mac to the UHeM VPN.
2. Download and unzip the launcher. Keep the extracted files together.
3. Double-click **`DSAI-3B-Instruct.command`** and complete your usual SSH login
   in Terminal. Python 3 and OpenSSH must be available on your Mac.
4. Wait for the scheduler and model loading. The browser opens when the exact
   SFT checkpoint is ready.

Keep the Terminal window open while chatting. Press **Control-C** when done.
If the launcher created the A100 job, it releases that job. If it reused an
existing job, it disconnects without stopping that session. The scheduler
also enforces the one-hour limit on newly created jobs.

If the connection drops during submission before a job ID is returned, a new
job may continue until that limit. Reconnect the VPN and run with
`--attach-only` to find the existing session. If you need to stop it manually,
connect to UHeM and run `scancel JOB_ID` using the ID printed by the launcher.

You can also start it from Terminal:

```bash
bash /path/to/DSAI-3B-Instruct-macOS/DSAI-3B-Instruct.command
```

The default SSH destination is `nunal@altay.uhem.itu.edu.tr`. This bundle is
configured for the existing model files and runtime in that UHeM account;
downloading it does not grant access to the account or compute allocation.
Passwords and private keys are supplied through your own SSH setup and are
not included in the download or sent to GitHub.

## Optional settings

```bash
# Use an SSH alias from your own ~/.ssh/config.
bash DSAI-3B-Instruct.command --host uhem-altay

# Attach to an existing matching job without ever requesting a GPU.
bash DSAI-3B-Instruct.command --attach-only

# Attach to a particular matching job.
bash DSAI-3B-Instruct.command --job-id JOB_ID

# Check local bundle integrity and show the plan without connecting.
bash DSAI-3B-Instruct.command --dry-run

# Verify an existing session, then disconnect without opening a browser.
bash DSAI-3B-Instruct.command --verify-only --no-browser
```

`UHEM_HOST` can supply the SSH destination. `UHEM_CONTROL_PATH` or
`--control-path` can reuse an existing authenticated SSH master; the launcher
does not close a master that it did not create. `--port` selects a local
port, and `--wait-minutes` changes the default 20-minute startup wait.

## Model and runtime

- SFT checkpoint: `tr_d32_bpe32k_capability_calibration_eff020_v1`, step 32.
- Model SHA-256: `4238d6271c8723460d20abec1ae88bb95da44cdae2087e01c90bd75d7bc49bf0`.
- Context: 2,048 tokens including the response. Start a new conversation when
  the context fills up.
- Hardware: one A100 in `a100q`, account `nakane`, eight requested CPU cores,
  64 GB host memory, one-hour limit.
- Runtime: the archived capability-calibration code and existing Python
  environment on UHeM, with SDPA attention and pinned inference-file hashes.

This is the surviving SFT candidate used for the interactive trial. The
launcher does not publish or imply benchmark results for it.

The server listens only on the compute node's loopback address. Two SSH
forwards carry the connection through the login node to a local browser port.
The link is a downloadable launcher, not an always-running public website;
GitHub itself does not start the GPU or handle the SSH login.

## Bundle contents

- `DSAI-3B-Instruct.command`: macOS entry point.
- `uhem_chat_launcher.py`: local login, session lifecycle and browser access.
- `remote_control.py`: owner-checked job discovery/submission and cancellation.
- `launch_chat.sbatch`: one-A100 inference job.
- `chat_web.py`: SFT chat server using the archived model's chat contract.
- `manifest.json`: checksums of the distributed files.

Validation on 9 November 2026 passed 33 automated tests covering job
ownership, reuse, cleanup and model identity. A live one-A100 test verified
startup and a generated response; a second launch reused that same job, and
Control-C in the original launcher released it while preserving the existing
SSH connection.
