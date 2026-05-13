import unittest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.preprocess import clean_text, tokenize, preprocess_pipeline

class TestPreprocess(unittest.TestCase):
    def test_lowercase(self):
        self.assertEqual(clean_text("ABC", lower=True), "abc")
    def test_remove_url(self):
        self.assertNotIn("http", clean_text("Check http://example.com"))
    def test_remove_punct(self):
        self.assertEqual(clean_text("Hi!!!", remove_punct=True), "hi")
    def test_tokenize(self):
        self.assertEqual(tokenize("a b c"), ["a","b","c"])
    def test_pipeline(self):
        result = preprocess_pipeline("Hello https://spam.com WINNER!!!")
        self.assertNotIn("https", " ".join(result))
        self.assertIn("winner", result)

if __name__ == "__main__":
    unittest.main()