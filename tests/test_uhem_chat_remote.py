"""Offline lifecycle checks for the UHeM chat controller.

All Slurm interactions are mocked. These tests must never allocate or cancel
an actual cluster job.
"""

import base64
import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "uhem_chat_remote_under_test", REPO_ROOT / "tools/uhem_chat/remote_control.py"
)
remote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote)


class RemoteControlLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root_patch = patch.object(remote, "CONTROL_ROOT", Path(temporary.name) / "control")
        root_patch.start()
        self.addCleanup(root_patch.stop)
        remote.private_directory(remote.CONTROL_ROOT)
        # Any forgotten command mock fails the test before subprocess is used.
        command_guard = patch.object(
            remote, "run_command", side_effect=AssertionError("Unmocked Slurm command")
        )
        command_guard.start()
        self.addCleanup(command_guard.stop)
        self.runtime_bytes = {
            "chat_web.py": b'print("test fixture")\n',
            "launch_chat.sbatch": b"#!/bin/bash\ntrue\n",
        }
        self.expected = {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in self.runtime_bytes.items()
        }
        self.payload = {
            "action": "find_or_start",
            "bundle": {
                name: base64.b64encode(raw).decode()
                for name, raw in self.runtime_bytes.items()
            },
            "expected": self.expected,
            "attach_only": False,
            "job_id": None,
        }

    def record(self, job_id="558074", **changes):
        record = {
            "JobId": job_id,
            "JobName": "tr-d32-eff020-chat",
            "JobState": "RUNNING",
            "UserId": "test({})".format(os.getuid()),
            "Command": "/trusted/historical/launch_chat.sbatch",
            "NodeList": "a123",
            "EndTime": "2026-11-09T16:58:47",
        }
        record.update(changes)
        return record

    def created_record(self, job_id):
        receipt = remote.load_receipt(job_id)
        return self.record(
            job_id,
            JobName="dsai-3b-instruct",
            Command=receipt["control_dir"] + "/launch_chat.sbatch",
        )

    def create_fixture_job(self, job_id="12345", inspection=None):
        with patch.object(remote, "queue_jobs", return_value={}), patch.object(
            remote, "submit_job", return_value=job_id
        ), patch.object(
            remote,
            "owned_job_record",
            side_effect=inspection or self.created_record,
        ):
            return remote.find_or_start(self.payload)

    def test_running_historical_job_reused_without_lease_or_submission(self):
        with patch.object(remote, "queue_jobs", return_value={"558074": "RUNNING"}), patch.object(
            remote, "owned_job_record", return_value=self.record()
        ), patch.object(
            remote, "job_launcher_hash", return_value=remote.HISTORICAL_LAUNCHER_SHA256
        ), patch.object(remote, "submit_job") as submit:
            result = remote.find_or_start(self.payload)
        self.assertEqual(result["job_id"], "558074")
        self.assertFalse(result["created"])
        self.assertIsNone(result["lease"])
        submit.assert_not_called()

    def test_matching_pending_job_prevents_duplicate_gpu_submission(self):
        with patch.object(remote, "queue_jobs", return_value={"558074": "PENDING"}), patch.object(
            remote,
            "owned_job_record",
            return_value=self.record(JobState="PENDING", NodeList="(null)"),
        ), patch.object(
            remote, "job_launcher_hash", return_value=self.expected["launch_chat.sbatch"]
        ), patch.object(remote, "submit_job") as submit:
            result = remote.find_or_start(self.payload)
        self.assertEqual(result["status"], "PENDING")
        self.assertIsNone(result["node"])
        self.assertFalse(result["created"])
        submit.assert_not_called()

    def test_attach_only_does_not_submit_when_no_session_exists(self):
        with patch.object(remote, "queue_jobs", return_value={}), patch.object(
            remote, "submit_job"
        ) as submit:
            with self.assertRaisesRegex(ValueError, "attach-only"):
                remote.find_or_start(dict(self.payload, attach_only=True))
        submit.assert_not_called()

    def test_named_session_with_changed_launcher_prevents_duplicate_submission(self):
        with patch.object(remote, "queue_jobs", return_value={"558074": "RUNNING"}), patch.object(
            remote, "owned_job_record", return_value=self.record()
        ), patch.object(remote, "job_launcher_hash", return_value="f" * 64), patch.object(
            remote, "submit_job"
        ) as submit:
            with self.assertRaisesRegex(ValueError, "refusing to allocate another GPU"):
                remote.find_or_start(self.payload)
        submit.assert_not_called()

    def test_new_job_receipt_is_private_and_binds_exact_job_and_launcher(self):
        result = self.create_fixture_job()
        receipt = remote.load_receipt("12345")
        self.assertTrue(result["created"])
        self.assertEqual(receipt["lease"], result["lease"])
        self.assertEqual(receipt["job_id"], "12345")
        self.assertEqual(receipt["launcher_sha256"], self.expected["launch_chat.sbatch"])
        self.assertEqual(remote.receipt_path("12345").stat().st_mode & 0o777, 0o600)
        self.assertEqual(Path(receipt["control_dir"]).stat().st_mode & 0o777, 0o700)

    def test_wrong_lease_cannot_cancel_a_job(self):
        self.create_fixture_job()
        with patch.object(remote, "run_command") as command:
            with self.assertRaisesRegex(ValueError, "exact lease"):
                remote.cancel({"job_id": "12345", "lease": "wrong"})
        command.assert_not_called()

    def test_valid_lease_cancels_only_exact_created_job_id(self):
        created = self.create_fixture_job()
        with patch.object(remote, "queue_jobs", return_value={"12345": "RUNNING"}), patch.object(
            remote, "owned_job_record", side_effect=self.created_record
        ), patch.object(remote, "run_command") as command:
            result = remote.cancel({"job_id": "12345", "lease": created["lease"]})
        self.assertTrue(result["cancelled"])
        command.assert_called_once_with(["scancel", "12345"])

    def test_reused_historical_job_has_no_cancellable_lease(self):
        with patch.object(remote, "run_command") as command:
            with self.assertRaisesRegex(ValueError, "exact lease"):
                remote.cancel({"job_id": "558074", "lease": "anything"})
        command.assert_not_called()

    def test_modified_immutable_bundle_is_rejected_without_overwriting_it(self):
        directory = remote.deploy_bundle(self.runtime_bytes, self.expected)
        changed = directory / "chat_web.py"
        changed.write_bytes(b"changed content")
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            remote.deploy_bundle(self.runtime_bytes, self.expected)
        self.assertEqual(changed.read_bytes(), b"changed content")

    def test_postsubmission_inspection_failures_keep_cleanup_handle(self):
        def inspection_error(_job_id):
            raise RuntimeError("Slurm temporarily unavailable")

        def invalid_node(job_id):
            record = self.created_record(job_id)
            record["NodeList"] = "a[123-124]"
            return record

        for job_id, inspection in (("12345", inspection_error), ("12346", invalid_node)):
            with self.subTest(job_id=job_id):
                result = self.create_fixture_job(job_id, inspection)
                receipt = remote.load_receipt(job_id)
                self.assertTrue(result["created"])
                self.assertEqual(result["job_id"], job_id)
                self.assertEqual(result["lease"], receipt["lease"])
                self.assertEqual(result["control_dir"], receipt["control_dir"])
                self.assertEqual(result["status"], "PENDING")
                self.assertIn("inspection_warning", result)

    def test_explicit_missing_job_does_not_fall_back_to_allocating(self):
        with patch.object(remote, "queue_jobs", return_value={}), patch.object(
            remote, "submit_job"
        ) as submit:
            with self.assertRaisesRegex(ValueError, "owned by this user"):
                remote.find_or_start(dict(self.payload, job_id="99999"))
        submit.assert_not_called()

    def test_existing_session_inspection_error_never_allocates_duplicate(self):
        with patch.object(remote, "queue_jobs", return_value={"558074": "RUNNING"}), patch.object(
            remote, "owned_job_record", side_effect=RuntimeError("Controller unavailable")
        ), patch.object(remote, "submit_job") as submit:
            with self.assertRaisesRegex(RuntimeError, "Controller unavailable"):
                remote.find_or_start(self.payload)
        submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
