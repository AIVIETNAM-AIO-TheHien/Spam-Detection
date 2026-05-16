# tests/test_preprocess_fixed.py
import unittest
import sys
import os
import io
import re

# Thêm đường dẫn gốc dự án vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocess_fixed import (
    normalize_entities,
    remove_system_noise,
    remove_mailing_list_boilerplate,
    truncate_text,
    clean_text,
    preprocess_email,
    preprocess_pipeline,
    report_length_stats
)


class TestNormalizeEntities(unittest.TestCase):
    """Unit tests for entity normalization (URL, email, phone, money)"""

    def test_url(self):
        text = "Visit http://example.com and https://secure.com or www.google.com"
        result = normalize_entities(text)
        self.assertEqual(result.count("[URL]"), 3)
        self.assertNotIn("http://", result)
        self.assertNotIn("www.", result)

    def test_email(self):
        text = "Contact me@example.com and support@test.org"
        result = normalize_entities(text)
        self.assertEqual(result.count("[EMAIL]"), 2)
        self.assertNotIn("@", result)

    def test_phone_vietnam_formats(self):
        text = "Call 0909123456 or 0987.654.321 or +84 987 654 321"
        result = normalize_entities(text)
        self.assertEqual(result.count("[PHONE]"), 3)

    def test_phone_simple(self):
        text = "Phone 12345678 or 12345678901"
        result = normalize_entities(text)
        self.assertEqual(result.count("[PHONE]"), 2)

    def test_money(self):
        text = "You won $1000 or 2 million dollars or €50 or £20.5"
        result = normalize_entities(text)
        self.assertEqual(result.count("[MONEY]"), 4)

    def test_no_entities(self):
        text = "This is a normal sentence without any entities."
        result = normalize_entities(text)
        self.assertEqual(result, text)

    def test_overlapping_patterns(self):
        # URL inside email? unlikely but test robustness
        text = "Check https://user@example.com for details"
        result = normalize_entities(text)
        self.assertIn("[URL]", result)
        self.assertIn("[EMAIL]", result)


class TestRemoveSystemNoise(unittest.TestCase):
    """Unit tests for cleaning system noise (exmh, PGP, headers, base64)"""

    def test_exmh_id(self):
        text = "some text _exmh_12345678P more text"
        result = remove_system_noise(text)
        self.assertNotIn("_exmh_", result)
        self.assertIn("some text more text", re.sub(r'\s+', ' ', result))

    def test_pgp_signature(self):
        text = """Hello
-----BEGIN PGP SIGNATURE-----
Version: GnuPG v1
iQEzBAABCAAdFiEE...
-----END PGP SIGNATURE-----
End"""
        result = remove_system_noise(text)
        self.assertNotIn("PGP", result)
        self.assertIn("Hello", result)
        self.assertIn("End", result)

    def test_headers(self):
        text = "Date: 2025-01-01\nFrom: sender@example.com\nSubject: Hello\nThis is body"
        result = remove_system_noise(text)
        self.assertNotIn("Date:", result)
        self.assertNotIn("From:", result)
        self.assertNotIn("Subject:", result)
        self.assertIn("This is body", result)

    def test_base64_string(self):
        long_b64 = "aHR0cHM6Ly9leGFtcGxlLmNvbSBsb25nYmFzZTY0c3RyaW5ndGhhdGlzdmVyeWxvbmc="
        text = f"prefix {long_b64} suffix"
        result = remove_system_noise(text)
        self.assertNotIn(long_b64, result)

    def test_no_noise(self):
        text = "Clean text without any system noise."
        result = remove_system_noise(text)
        self.assertEqual(result, text)


class TestRemoveMailingListBoilerplate(unittest.TestCase):
    """Unit tests for removing mailing list footers"""

    def test_irish_linux_group(self):
        text = "Some content\nIrish Linux Users Group: ilug@linux.ie\nhttp://www.linux.ie/mailman/listinfo/ilug for unsubscription information. List maintainer: listmaster@linux.ie"
        result = remove_mailing_list_boilerplate(text)
        self.assertNotIn("unsubscription", result.lower())
        self.assertIn("some content", result.lower())

    def test_yahoo_sponsor(self):
        text = "Content\n------------------------ Yahoo! Groups Sponsor ---------------------~--> 4 DVDs Free +s&p Join Now"
        result = remove_mailing_list_boilerplate(text)
        self.assertNotIn("yahoo", result.lower())
        self.assertIn("content", result.lower())

    def test_separator_lines(self):
        text = "Main text\n_______________________________________________\nFooter text"
        result = remove_mailing_list_boilerplate(text)
        self.assertNotIn("_______________________________________________", result)
        self.assertIn("main text", result.lower())

    def test_no_boilerplate(self):
        text = "Just a normal email without any footer."
        result = remove_mailing_list_boilerplate(text)
        self.assertEqual(result, text)


class TestTruncateText(unittest.TestCase):
    """Unit tests for text truncation by word count"""

    def test_no_truncate(self):
        text = "one two three four five"
        result = truncate_text(text, max_words=10)
        self.assertEqual(result, text)

    def test_truncate_exact(self):
        text = "one two three four five"
        result = truncate_text(text, max_words=5)
        self.assertEqual(result, text)

    def test_truncate_less(self):
        text = "one two three four five six seven"
        result = truncate_text(text, max_words=4)
        self.assertEqual(result, "one two three four")
        self.assertEqual(len(result.split()), 4)

    def test_truncate_empty(self):
        self.assertEqual(truncate_text("", 5), "")

    def test_truncate_with_extra_spaces(self):
        text = "one  two   three    four"
        result = truncate_text(text, max_words=3)
        self.assertEqual(result, "one two three")


class TestCleanText(unittest.TestCase):
    """Comprehensive tests for the main clean_text pipeline"""

    def test_basic_cleaning(self):
        text = "Hello WORLD!!!"
        result = clean_text(text, lower=True, remove_punct=True)
        self.assertEqual(result, "hello world")

    def test_with_entities_and_noise(self):
        text = "Check https://spam.com or email me@abc.com Call 0909123456. _exmh_123"
        result = clean_text(text)
        self.assertIn("[url]", result)   # because lower is applied after normalization
        self.assertIn("[email]", result)
        self.assertIn("[phone]", result)
        self.assertNotIn("_exmh_", result)

    def test_boilerplate_removal(self):
        text = "Important message\nIrish Linux Users Group unsubscription info"
        result = clean_text(text, remove_boilerplate_flag=True)
        self.assertNotIn("unsubscription", result)

    def test_length_control(self):
        long_text = "word " * 2000
        result = clean_text(long_text, max_words=1024)
        self.assertEqual(len(result.split()), 1024)

    def test_remove_numbers(self):
        text = "Call 12345 and [phone] token"
        result = clean_text(text, remove_numbers=True, normalize_entities_flag=False)
        self.assertNotIn("12345", result)
        self.assertIn("and", result)

    def test_keep_brackets(self):
        # Ensure that after punctuation removal, [URL] remains
        text = "Visit https://example.com"
        result = clean_text(text, remove_punct=True, normalize_entities_flag=True)
        self.assertIn("[url]", result)   # brackets preserved

    def test_empty_input(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "None")  # converts to string

    def test_non_string_input(self):
        self.assertEqual(clean_text(12345, lower=False, remove_punct=False), "12345")


class TestPreprocessEmail(unittest.TestCase):
    """Unit tests for email preprocessing (subject + body)"""

    def test_subject_body(self):
        subject = "Win $500"
        body = "Call 0909123456 now!"
        result = preprocess_email(subject=subject, body=body)
        self.assertIn("win", result)
        self.assertIn("call", result)
        self.assertIn("[money]", result)
        self.assertIn("[phone]", result)

    def test_subject_only(self):
        result = preprocess_email(subject="Hello World")
        self.assertEqual(result, "hello world")

    def test_body_only(self):
        result = preprocess_email(body="Just a body")
        self.assertEqual(result, "just a body")

    def test_text_only_overrides(self):
        result = preprocess_email(subject="ignored", body="ignored", text="direct text")
        self.assertEqual(result, "direct text")

    def test_empty_inputs(self):
        result = preprocess_email()
        self.assertEqual(result, "")
        result = preprocess_email(subject=None, body=None)
        self.assertEqual(result, "")

    def test_custom_kwargs(self):
        # pass custom parameters to clean_text
        result = preprocess_email(subject="Hello!!!", body="World!!!", lower=False, remove_punct=False)
        self.assertEqual(result, "Hello!!! World!!!")


class TestPreprocessPipeline(unittest.TestCase):
    """Unit tests for the simple pipeline (SMS / comments)"""

    def test_pipeline_sms(self):
        raw = "WINNER!!! You've won $1000. Call 0901234567 now!"
        result = preprocess_pipeline(raw)
        self.assertIn("winner", result)
        self.assertIn("[money]", result)
        self.assertIn("[phone]", result)
        self.assertNotIn("!!!", result)

    def test_pipeline_empty(self):
        self.assertEqual(preprocess_pipeline(""), "")
        self.assertEqual(preprocess_pipeline(None), "None")

    def test_pipeline_already_clean(self):
        text = "hello world"
        self.assertEqual(preprocess_pipeline(text), text)


class TestReportLengthStats(unittest.TestCase):
    """Unit test for report_length_stats (capture stdout)"""

    def test_report(self):
        texts = ["short", "a bit longer sentence", "this is a fairly long text that contains several words"]
        captured = io.StringIO()
        sys.stdout = captured
        report_length_stats(texts)
        sys.stdout = sys.__stdout__
        output = captured.getvalue()
        self.assertIn("Length stats (words):", output)
        self.assertIn("min=1", output)
        self.assertIn("max=9", output)
        self.assertIn("mean", output)

    def test_empty_list(self):
        captured = io.StringIO()
        sys.stdout = captured
        report_length_stats([])
        sys.stdout = sys.__stdout__
        self.assertIn("Empty list", captured.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)