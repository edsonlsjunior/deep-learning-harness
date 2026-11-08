import asyncio
import unittest
from unittest.mock import AsyncMock, patch
from app import harness as H


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        for run in list(H.RUNS.values()):
            H.close(run)

    async def test_policy_matrix(self):
        self.assertEqual(H.decision("consult", "write"), "blocked")
        self.assertEqual(H.decision("confirm", "write"), "awaiting")
        self.assertEqual(H.decision("operate", "write"), "allowed")
        self.assertEqual(H.decision("operate", "delete"), "awaiting")

    async def test_consult_never_executes_write(self):
        run, forward = H.create("consult"), AsyncMock()
        with self.assertRaises(PermissionError):
            await H.execute(run.token, "POST", "/api/projects", "create_project", {}, forward)
        forward.assert_not_awaited()

    async def test_operate_executes_normal_action(self):
        run, forward = H.create("operate"), AsyncMock(return_value={"id": 1})
        self.assertEqual(await H.execute(run.token, "POST", "/api/projects", "create_project", {}, forward), {"id": 1})
        forward.assert_awaited_once()

    async def test_confirmation_waits_then_executes_once(self):
        run, forward = H.create("confirm"), AsyncMock(return_value={"id": 1})
        task = asyncio.create_task(H.execute(run.token, "POST", "/api/projects", "create_project", {"json": {"name": "Aula"}}, forward))
        _, event = await run.queue.get()
        self.assertEqual(event["decision"], "awaiting")
        forward.assert_not_awaited()
        H.approve(run.id, event["id"], True)
        await task
        forward.assert_awaited_once()
        with self.assertRaises(ValueError):
            H.approve(run.id, event["id"], True)

    async def test_delete_in_operate_requires_approval_and_denial_blocks(self):
        run, forward = H.create("operate"), AsyncMock()
        task = asyncio.create_task(H.execute(run.token, "DELETE", "/api/projects/123", "delete_project", {}, forward))
        _, event = await run.queue.get()
        self.assertEqual(event["decision"], "awaiting")
        H.approve(run.id, event["id"], False)
        with self.assertRaises(PermissionError):
            await task
        forward.assert_not_awaited()

    async def test_expiration_blocks(self):
        run, forward = H.create("confirm"), AsyncMock()
        with patch.object(H, "APPROVAL_SECONDS", .001):
            with self.assertRaises(PermissionError):
                await H.execute(run.token, "POST", "/api/projects", "create_project", {}, forward)
        forward.assert_not_awaited()

    async def test_close_cancels_pending_and_revokes_token(self):
        run, forward = H.create("confirm"), AsyncMock()
        task = asyncio.create_task(H.execute(run.token, "POST", "/api/projects", "create_project", {}, forward))
        await run.queue.get()
        H.close(run)
        with self.assertRaises(PermissionError):
            await task
        with self.assertRaises(PermissionError):
            await H.execute(run.token, "GET", "/api/projects", "list_projects", {}, forward)
        forward.assert_not_awaited()

    async def test_wrong_run_cannot_approve(self):
        first, second, forward = H.create("confirm"), H.create("confirm"), AsyncMock()
        task = asyncio.create_task(H.execute(first.token, "POST", "/api/projects", "create_project", {}, forward))
        _, event = await first.queue.get()
        with self.assertRaises(ValueError):
            H.approve(second.id, event["id"], True)
        H.close(first)
        with self.assertRaises(PermissionError):
            await task
        forward.assert_not_awaited()

    async def test_route_allowlist_and_step_limit(self):
        for path in ("http://other/api/projects", "/api/harness/execute", "/api/projects/../harness", "/api/projects?x=1"):
            with self.assertRaises(PermissionError):
                H.classify("GET", path)
        run, forward = H.create("operate"), AsyncMock()
        run.steps = H.MAX_STEPS
        with self.assertRaises(PermissionError):
            await H.execute(run.token, "GET", "/api/projects", "list_projects", {}, forward)
        forward.assert_not_awaited()
