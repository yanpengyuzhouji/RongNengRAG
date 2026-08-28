import unittest

from src.api.main import _casual_reply


class CasualChatTests(unittest.TestCase):
    def test_greeting_and_connection_test_are_handled_without_retrieval(self):
        answer = _casual_reply("你好，测试一下")
        self.assertIsNotNone(answer)
        self.assertIn("榕能知识库助手", answer)

    def test_common_casual_messages(self):
        self.assertIsNotNone(_casual_reply("您好"))
        self.assertIsNotNone(_casual_reply("谢谢你"))
        self.assertIsNotNone(_casual_reply("再见"))

    def test_substantive_question_is_not_treated_as_greeting(self):
        self.assertIsNone(_casual_reply("你好，请说明GB7595-2017的适用范围"))


if __name__ == "__main__":
    unittest.main()
