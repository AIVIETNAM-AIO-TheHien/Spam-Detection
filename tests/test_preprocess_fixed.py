"""
Unit test cho module src.data.preprocess_fixed
Kiểm tra các hàm chuẩn hóa thực thể, loại bỏ nhiễu, xóa boilerplate, cắt độ dài, pipeline.
"""

import unittest
import sys
import os

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
    report_length_stats  # không test hàm này vì chỉ in ra, nhưng import để đảm bảo tồn tại
)


class TestNormalizeEntities(unittest.TestCase):
    """Kiểm tra hàm thay thế thực thể bằng token"""

    def test_url(self):
        text = "Visit http://example.com and https://secure.com or www.google.com"
        result = normalize_entities(text)
        self.assertIn("[URL]", result)
        self.assertNotIn("http://", result)
        self.assertNotIn("www.", result)

    def test_email(self):
        text = "Contact me@example.com and support@test.org"
        result = normalize_entities(text)
        self.assertIn("[EMAIL]", result)
        self.assertNotIn("@", result)

    def test_phone_vietnam(self):
        text = "Call 0909123456 or 0987.654.321 or +84 987 654 321"
        result = normalize_entities(text)
        self.assertEqual(result.count("[PHONE]"), 3)
        self.assertNotIn("0909123456", result)

    def test_phone_simple(self):
        text = "Phone 12345678 or 12345678901"
        result = normalize_entities(text)
        self.assertEqual(result.count("[PHONE]"), 2)

    def test_money(self):
        text = "You won $1000 or 2 million dollars or €50"
        result = normalize_entities(text)
        self.assertEqual(result.count("[MONEY]"), 3)


class TestRemoveSystemNoise(unittest.TestCase):
    """Kiểm tra xóa nhiễu hệ thống (exmh, PGP, base64, headers)"""

    def test_exmh_id(self):
        text = "some text _exmh_12345678P more text"
        result = remove_system_noise(text)
        self.assertNotIn("_exmh_", result)
        self.assertIn("some text more text", result)

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
        text = "prefix aHR0cHM6Ly9leGFtcGxlLmNvbSBsb25nYmFzZTY0c3RyaW5ndGhhdGlzdmVyeWxvbmc= suffix"
        result = remove_system_noise(text)
        self.assertNotIn("aHR0cHM6Ly9leGFtcGxlLmNvbSBsb25nYmFzZTY0c3RyaW5ndGhhdGlzdmVyeWxvbmc=", result)


class TestRemoveMailingListBoilerplate(unittest.TestCase):
    """Kiểm tra xóa footer mailing list"""

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


class TestTruncateText(unittest.TestCase):
    """Kiểm tra cắt ngắn văn bản theo số từ"""

    def test_no_truncate(self):
        text = "one two three four five"
        result = truncate_text(text, max_words=10)
        self.assertEqual(result, text)

    def test_truncate(self):
        text = "one two three four five six seven eight nine ten"
        result = truncate_text(text, max_words=5)
        self.assertEqual(result, "one two three four five")
        self.assertEqual(len(result.split()), 5)

    def test_truncate_empty(self):
        self.assertEqual(truncate_text("", 5), "")


class TestCleanText(unittest.TestCase):
    """Kiểm tra pipeline clean_text tổng hợp"""

    def test_basic_cleaning(self):
        text = "Hello WORLD!!!"
        result = clean_text(text, lower=True, remove_punct=True)
        self.assertEqual(result, "hello world")

    def test_with_entities_and_noise(self):
        text = "Check https://spam.com or email me@abc.com Call 0909123456. _exmh_123"
        result = clean_text(text)
        self.assertIn("[url]", result)  # lower đã bật
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


class TestPreprocessEmail(unittest.TestCase):
    """Kiểm tra hàm xử lý email riêng subject và body"""

    def test_subject_body_concatenation(self):
        subject = "Win $500"
        body = "Call 0909123456 now!"
        result = preprocess_email(subject=subject, body=body)
        self.assertIn("win", result)
        self.assertIn("call", result)
        self.assertIn("[money]", result)
        self.assertIn("[phone]", result)

    def test_text_only(self):
        result = preprocess_email(text="Just a simple text")
        self.assertEqual(result, "just a simple text")

    def test_empty_input(self):
        result = preprocess_email(subject=None, body=None)
        self.assertEqual(result, "")


class TestPreprocessPipeline(unittest.TestCase):
    """Kiểm tra pipeline đơn giản cho văn bản (SMS, comment)"""

    def test_pipeline_sms(self):
        raw = "WINNER!!! You've won $1000. Call 0901234567 now!"
        result = preprocess_pipeline(raw)
        self.assertIn("winner", result)
        self.assertIn("[money]", result)
        self.assertIn("[phone]", result)
        self.assertNotIn("!!!", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)