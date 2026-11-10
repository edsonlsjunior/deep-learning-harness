import unittest

from app.session_credentials import SessionCredentialStore


class CredentialStoreTests(unittest.TestCase):
    def test_value_is_isolated_by_session_and_never_listed(self):
        clock = [10.0]
        store = SessionCredentialStore(ttl_seconds=60, clock=lambda: clock[0])
        store.set("alice", "openai", "only-alice")

        self.assertEqual(store.get("alice", "openai"), "only-alice")
        self.assertIsNone(store.get("bob", "openai"))
        self.assertTrue(store.configured("alice", "openai"))

    def test_value_expires_and_can_be_removed(self):
        clock = [0.0]
        store = SessionCredentialStore(ttl_seconds=60, clock=lambda: clock[0])
        store.set("session", "openai", "temporary")
        clock[0] = 61.0

        self.assertIsNone(store.get("session", "openai"))
        self.assertEqual(store.expires_in("session", "openai"), 0)
        store.remove("session", "openai")
        self.assertFalse(store.configured("session", "openai"))

    def test_rejects_unknown_provider_and_invalid_values(self):
        store = SessionCredentialStore()
        with self.assertRaisesRegex(ValueError, "não aceita"):
            store.set("session", "gemini", "key")
        with self.assertRaisesRegex(ValueError, "válida"):
            store.set("session", "openai", "   ")


if __name__ == "__main__":
    unittest.main()
