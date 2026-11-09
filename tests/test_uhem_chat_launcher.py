"""Exercise launcher ownership and identity checks without SSH or a GPU job."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import importlib.util
import io
from pathlib import Path
import shlex
import subprocess
import unittest
from unittest import mock


LAUNCHER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "uhem_chat"
    / "uhem_chat_launcher.py"
)
SPEC = importlib.util.spec_from_file_location("uhem_chat_launcher_under_test", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def healthy_server():
    return {
        "ready": True,
        "source": "sft",
        "model_tag": launcher.MODEL_TAG,
        "step": 32,
        "num_gpus": 1,
        "chat_contract": {"contract_sha256": launcher.CONTRACT_SHA256},
    }


class LauncherTestCase(unittest.TestCase):
    def setUp(self):
        self.stack = self.enterContext(ExitStack())
        # Any overlooked external effect is a test failure, not a real command.
        self.run = self.stack.enter_context(
            mock.patch.object(
                launcher.subprocess, "run", side_effect=AssertionError("Unexpected subprocess")
            )
        )
        self.popen = self.stack.enter_context(
            mock.patch.object(
                launcher.subprocess, "Popen", side_effect=AssertionError("Unexpected process")
            )
        )
        self.stack.enter_context(
            mock.patch.object(launcher.HTTP, "open", side_effect=AssertionError("Unexpected HTTP"))
        )
        self.stack.enter_context(
            mock.patch.object(launcher.webbrowser, "open", side_effect=AssertionError("Unexpected browser"))
        )
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(redirect_stderr(io.StringIO()))


class TestArgumentValidation(LauncherTestCase):
    def test_read_only_modes_imply_attach_only(self):
        for arguments in (["--verify-only"], ["--job-id", "558074"], ["--attach-only"]):
            with self.subTest(arguments=arguments):
                self.assertTrue(launcher.parse_args(arguments).attach_only)

    def test_rejects_invalid_or_shell_shaped_hosts(self):
        for host in ("", "-oProxyCommand=id", "altay other", "altay\nother", "altay;id", "$(id)", "`id`", "altay|id"):
            with self.subTest(host=host), self.assertRaises(SystemExit) as error:
                launcher.parse_args([f"--host={host}"])
            self.assertEqual(error.exception.code, 2)

    def test_accepts_normal_ssh_alias_and_user_host(self):
        for host in ("uhem", "uhem-login", "nunal@altay.uhem.itu.edu.tr", "nunal@10.128.2.40"):
            with self.subTest(host=host):
                self.assertEqual(launcher.parse_args(["--host", host]).host, host)

    def test_rejects_invalid_job_ids(self):
        for job_id in ("-1", "558074;id", "$(id)", "558074 558075", "558074.batch", "٥٥٨٠٧٤"):
            with self.subTest(job_id=job_id), self.assertRaises(SystemExit) as error:
                launcher.parse_args([f"--job-id={job_id}"])
            self.assertEqual(error.exception.code, 2)

    def test_rejects_ports_outside_local_tcp_range_or_nonintegers(self):
        for port in ("-1", "1", "1023", "65536", "8787;id", "$(id)"):
            with self.subTest(port=port), self.assertRaises(SystemExit) as error:
                launcher.parse_args([f"--port={port}"])
            self.assertEqual(error.exception.code, 2)

    def test_accepts_auto_and_unprivileged_ports(self):
        for port in (0, 1024, 8787, 65535):
            with self.subTest(port=port):
                self.assertEqual(launcher.parse_args(["--port", str(port)]).port, port)


class TestModelIdentity(LauncherTestCase):
    def test_accepts_the_expected_model(self):
        self.assertTrue(launcher.validate_health(healthy_server()))

    def test_rejects_other_checkpoint_or_contract(self):
        cases = (
            ("ready", False),
            ("source", "base"),
            ("model_tag", "tr_d32_bpe32k_smoltalk2_turk_full_v9_then_fast_stage_b_v1"),
            ("step", 84),
            ("num_gpus", 2),
            ("chat_contract", {"contract_sha256": "0" * 64}),
            ("chat_contract", None),
        )
        for key, value in cases:
            health = healthy_server()
            health[key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(launcher.validate_health(health))

    def test_identity_mismatch_aborts_wait_instead_of_accepting_another_server(self):
        session = mock.Mock()
        session.job = {"job_id": "558074"}
        health = healthy_server()
        health["step"] = 84
        with mock.patch.object(launcher.time, "monotonic", return_value=1), mock.patch.object(
            launcher, "read_health", return_value=health
        ), mock.patch.object(launcher.time, "sleep") as sleep:
            with self.assertRaisesRegex(launcher.LaunchError, "identity does not match"):
                launcher.wait_for_model(session, "http://127.0.0.1:8787", 100)
        sleep.assert_not_called()
        session.ssh.remote.assert_not_called()


class TestSessionOwnership(LauncherTestCase):
    def test_new_job_cleanup_uses_only_exact_job_and_cancellation_lease(self):
        ssh = mock.Mock()
        session = launcher.ChatSession(ssh)
        session.job = {"created": True, "job_id": "558999", "lease": "unique-owner-lease"}
        session.close()
        ssh.remote.assert_called_once_with("cancel", job_id="558999", lease="unique-owner-lease")

    def test_reused_or_unleased_jobs_are_never_cancelled(self):
        for job in (
            None,
            {"created": False, "job_id": "558074"},
            {"created": False, "job_id": "558074", "lease": "someone-elses-lease"},
            {"created": True, "job_id": "558999"},
        ):
            with self.subTest(job=job):
                ssh = mock.Mock()
                session = launcher.ChatSession(ssh)
                session.job = job
                session.close()
                ssh.remote.assert_not_called()

    def test_cancellation_failure_still_closes_owned_tunnel_resources(self):
        ssh = mock.Mock()
        ssh.remote.side_effect = launcher.LaunchError("SSH unavailable")
        session = launcher.ChatSession(ssh)
        session.job = {"created": True, "job_id": "558999", "lease": "owner-lease"}
        session.forward = "127.0.0.1:8787:127.0.0.1:30000"
        session.relay = mock.Mock()
        session.relay_log = mock.Mock()
        with mock.patch.object(launcher, "checked") as checked:
            session.close()
        ssh.remote.assert_called_once_with("cancel", job_id="558999", lease="owner-lease")
        ssh.command.assert_called_once_with("-O", "cancel", "-L", session.forward)
        checked.assert_called_once()
        session.relay.terminate.assert_called_once_with()
        session.relay.wait.assert_called_once_with(timeout=10)
        session.relay_log.close.assert_called_once_with()

    def test_external_ssh_master_is_checked_but_never_closed(self):
        connection = launcher.SSHConnection("uhem", "/tmp/existing-user-master")
        with mock.patch.object(launcher, "checked") as checked:
            connection.connect()
        command = checked.call_args.args[0]
        self.assertIn("/tmp/existing-user-master", command)
        self.assertEqual(command[-3:], ["-O", "check", "uhem"])
        connection.close()
        self.run.assert_not_called()
        self.assertFalse(Path(connection.temp_dir.name).exists())

    def test_owned_ssh_master_is_the_only_master_closed(self):
        connection = launcher.SSHConnection("uhem")
        self.run.side_effect = None
        self.run.return_value = subprocess.CompletedProcess([], 0)
        connection.close()
        command = self.run.call_args.args[0]
        self.assertEqual(command[-3:], ["-O", "exit", "uhem"])
        self.assertEqual(command[command.index("-S") + 1], connection.control_path)
        self.assertFalse(Path(connection.temp_dir.name).exists())

    def test_remote_arguments_are_shell_quoted_and_local_call_never_uses_shell(self):
        connection = launcher.SSHConnection("uhem", "/tmp/existing-user-master")
        arguments = ["printf", "%s", "a;$(id)`id`' with spaces"]
        self.run.side_effect = None
        self.run.return_value = subprocess.CompletedProcess([], 0, stdout="ok", stderr="")
        try:
            self.assertEqual(connection.run(arguments), "ok")
            command = self.run.call_args.args[0]
            self.assertEqual(shlex.split(command[-1]), arguments)
            self.assertNotIn("shell", self.run.call_args.kwargs)
        finally:
            connection.close()


class TestMainLifecycle(LauncherTestCase):
    def setUp(self):
        super().setUp()
        self.bundle = {"bundle": {"chat_web.py": "payload"}, "expected": {"chat_web.py": "hash"}}
        self.stack.enter_context(mock.patch.object(launcher, "load_bundle", return_value=self.bundle))
        self.stack.enter_context(mock.patch.object(launcher.signal, "signal"))
        self.ssh = mock.Mock(spec=launcher.SSHConnection)
        self.connection_class = self.stack.enter_context(
            mock.patch.object(launcher, "SSHConnection", return_value=self.ssh)
        )
        self.node = self.stack.enter_context(
            mock.patch.object(launcher, "wait_for_node", return_value={"status": "RUNNING", "node": "a123"})
        )
        self.tunnel = self.stack.enter_context(
            mock.patch.object(launcher.ChatSession, "attach_tunnel", return_value="http://127.0.0.1:8787")
        )
        self.model = self.stack.enter_context(
            mock.patch.object(launcher, "wait_for_model", return_value=healthy_server())
        )
        self.sleep = self.stack.enter_context(mock.patch.object(launcher.time, "sleep", side_effect=KeyboardInterrupt))

    def fake_job(self, created=False):
        job = {"created": created, "job_id": "558999" if created else "558074"}
        if created:
            job["lease"] = "owned-job-lease"

        def remote(action, **payload):
            if action == "find_or_start":
                return copy.deepcopy(job)
            if action == "cancel":
                return {"cancelled": True}
            raise AssertionError(f"Unexpected remote action: {action}")

        self.ssh.remote.side_effect = remote
        return job

    def test_verify_only_never_requests_submission_or_cancels_reused_job(self):
        self.fake_job()
        self.assertEqual(launcher.main(["--verify-only", "--host", "uhem"]), 0)
        self.ssh.remote.assert_called_once_with("find_or_start", **self.bundle, attach_only=True, job_id=None)
        self.ssh.close.assert_called_once_with()
        self.sleep.assert_not_called()

    def test_explicit_job_never_requests_submission_or_cancels_reused_job(self):
        self.fake_job()
        self.assertEqual(launcher.main(["--job-id", "558074", "--no-browser"]), 0)
        self.ssh.remote.assert_called_once_with("find_or_start", **self.bundle, attach_only=True, job_id="558074")
        self.ssh.close.assert_called_once_with()

    def test_missing_attach_only_job_does_not_retry_with_submission(self):
        self.ssh.remote.side_effect = launcher.LaunchError("No matching job")
        self.assertEqual(launcher.main(["--attach-only", "--no-browser"]), 1)
        self.ssh.remote.assert_called_once_with("find_or_start", **self.bundle, attach_only=True, job_id=None)
        self.node.assert_not_called()
        self.ssh.close.assert_called_once_with()

    def test_failure_after_creation_cancels_exact_owned_job(self):
        for phase in ("node", "tunnel", "model"):
            with self.subTest(phase=phase):
                self.ssh.reset_mock()
                self.fake_job(created=True)
                failing = getattr(self, phase)
                failing.side_effect = launcher.LaunchError("Startup failed")
                try:
                    self.assertEqual(launcher.main(["--no-browser"]), 1)
                finally:
                    failing.side_effect = None
                self.assertEqual(
                    self.ssh.remote.call_args_list,
                    [
                        mock.call("find_or_start", **self.bundle, attach_only=False, job_id=None),
                        mock.call("cancel", job_id="558999", lease="owned-job-lease"),
                    ],
                )
                self.ssh.close.assert_called_once_with()

    def test_control_c_after_creation_releases_only_owned_job(self):
        self.fake_job(created=True)
        self.assertEqual(launcher.main(["--no-browser"]), 0)
        self.ssh.remote.assert_has_calls(
            [
                mock.call("find_or_start", **self.bundle, attach_only=False, job_id=None),
                mock.call("cancel", job_id="558999", lease="owned-job-lease"),
            ]
        )
        self.assertEqual(self.ssh.remote.call_count, 2)
        self.ssh.close.assert_called_once_with()

    def test_dry_run_never_connects_or_opens_a_tunnel(self):
        self.assertEqual(launcher.main(["--dry-run", "--verify-only"]), 0)
        self.connection_class.assert_not_called()
        self.ssh.remote.assert_not_called()
        self.tunnel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
