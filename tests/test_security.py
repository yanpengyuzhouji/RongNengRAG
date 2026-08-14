import unittest

from src.api.security import (
    ApiKeyAuthenticator,
    AuthConfigurationError,
    AuthenticationError,
    cors_origins,
)


class ApiKeyAuthenticatorTests(unittest.TestCase):
    def test_authentication_is_required_and_fail_closed_by_default(self):
        authenticator = ApiKeyAuthenticator.from_config({}, {})
        self.assertFalse(authenticator.configured)
        with self.assertRaises(AuthConfigurationError):
            authenticator.authenticate()

    def test_json_keys_map_to_stable_principals(self):
        key = "a-secure-api-key-with-32-characters"
        authenticator = ApiKeyAuthenticator.from_config(
            {}, {"RONGNENG_API_KEYS_JSON": '{"alice":"' + key + '"}'}
        )
        self.assertEqual("alice", authenticator.authenticate(f"Bearer {key}"))
        self.assertEqual("alice", authenticator.authenticate(x_api_key=key))
        with self.assertRaises(AuthenticationError):
            authenticator.authenticate(x_api_key="wrong-key")

    def test_weak_keys_and_wildcard_cors_are_rejected(self):
        with self.assertRaises(AuthConfigurationError):
            ApiKeyAuthenticator.from_config(
                {}, {"RONGNENG_API_KEY": "short"}
            )
        with self.assertRaises(AuthConfigurationError):
            cors_origins({"security": {"cors_origins": ["*"]}})


if __name__ == "__main__":
    unittest.main()
