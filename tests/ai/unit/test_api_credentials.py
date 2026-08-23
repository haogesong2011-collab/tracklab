"""API key helpers must never echo the secret."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai import api_credentials as creds  # noqa: E402


class ApiCredentialsTests(unittest.TestCase):
    def tearDown(self) -> None:
        creds._session_key = None
        os.environ.pop(creds.CONFIG_DIR_ENV, None)
        os.environ.pop(creds.ENV_NAME, None)

    def test_mask_and_session_storage(self) -> None:
        self.assertEqual(creds.mask_key(None), "未配置")
        self.assertTrue(creds.mask_key("sk-abcdefghijk").endswith("hijk"))
        self.assertNotIn("sk-abcdef", creds.mask_key("sk-abcdefghijk"))
        persisted = creds.set_api_key("sk-session-test-key", persist=False)
        self.assertFalse(persisted)
        self.assertEqual(creds._session_key, "sk-session-test-key")
        creds._session_key = None
        self.assertTrue(creds.mask_key("abcd").startswith("•"))

    def test_normalize_strips_chinese_labels(self) -> None:
        self.assertEqual(
            creds.normalize_api_key("API密钥： sk-abcdefghijk （测试）"),
            "sk-abcdefghijk",
        )

    def test_persist_to_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[creds.CONFIG_DIR_ENV] = tmp
            os.environ.pop(creds.ENV_NAME, None)
            self.assertTrue(creds.set_api_key("sk-file-persist-key", persist=True))
            creds._session_key = None
            self.assertEqual(creds.get_api_key(), "sk-file-persist-key")
            stored = Path(tmp) / creds.KEY_FILENAME
            self.assertTrue(stored.is_file())
            self.assertNotIn("sk-file-persist-key", creds.mask_key(creds.get_api_key()))
            creds.clear_api_key()
            self.assertIsNone(creds.get_api_key())
            self.assertFalse(stored.exists())
