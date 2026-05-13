"""
Unit test cho module src.data.preprocess
Kiểm tra các hàm clean_text, tokenize, preprocess_pipeline
Chạy được từ cả thư mục gốc hoặc thư mục tests
"""

import unittest
import sys
import os

# --- Tự động thêm đường dẫn thư mục gốc vào sys.path ---
# Lấy đường dẫn tuyệt đối của thư mục chứa file test này
current_dir = os.path.dirname(os.path.abspath(__file__))
# Đi lên 1 cấp để tới thư mục gốc dự án (Spam-Detection)
project_root = os.path.dirname(current_dir)
# Thêm thư mục gốc vào sys.path để import được src
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import module cần test
from src.data.preprocess import clean_text, tokenize, preprocess_pipeline

class TestCleanText(unittest.TestCase):
    """Kiểm tra hàm clean_text với các tùy chọn khác nhau"""

    def test_lowercase(self):
        """Chuyển chữ hoa thành chữ thường"""
        self.assertEqual(clean_text("Hello WORLD", lowercase=True), "hello world")

    def test_no_lowercase(self):
        """Giữ nguyên chữ hoa khi lowercase=False"""
        self.assertEqual(clean_text("Hello WORLD", lowercase=False), "Hello WORLD")

    def test_remove_urls(self):
        """Xóa URL (http, https, www)"""
        text = "Check https://example.com and http://test.com or www.google.com"
        result = clean_text(text, remove_urls=True)
        self.assertNotIn("https://", result)
        self.assertNotIn("http://", result)
        self.assertNotIn("www.", result)
        self.assertIn("Check and or", result)

    def test_remove_emails(self):
        """Xóa địa chỉ email"""
        text = "Contact me@example.com or support@test.org for help"
        result = clean_text(text, remove_emails=True)
        self.assertNotIn("@", result)
        self.assertIn("Contact or for help", result)

    def test_remove_punctuation(self):
        """Xóa dấu câu"""
        text = "Hello!!! How are you? I'm fine, thanks."
        result = clean_text(text, remove_punct=True)
        self.assertEqual(result, "Hello How are you Im fine thanks")

    def test_remove_html(self):
        """Xóa thẻ HTML"""
        text = "<html><body>Hello <b>World</b></body></html>"
        result = clean_text(text, remove_html=True)
        self.assertEqual(result, "Hello World")

    def test_remove_numbers(self):
        """Xóa số (khi bật)"""
        text = "Call 0901234567 or 123 for support"
        result = clean_text(text, remove_numbers=True)
        self.assertNotIn("0901234567", result)
        self.assertNotIn("123", result)
        self.assertIn("Call or for support", result)

    def test_keep_numbers_by_default(self):
        """Mặc định giữ lại số (quan trọng cho spam)"""
        text = "Call 0901234567 now"
        result = clean_text(text, remove_numbers=False)
        self.assertIn("0901234567", result)

    def test_combination_of_options(self):
        """Kết hợp nhiều tùy chọn cùng lúc"""
        text = "Check https://spam.com and email me@abc.com!!! WINNER 100K"
        result = clean_text(
            text,
            lowercase=True,
            remove_urls=True,
            remove_emails=True,
            remove_punct=True,
            remove_html=True,
            remove_numbers=False,
        )
        self.assertEqual(result, "check and winner 100k")


class TestTokenize(unittest.TestCase):
    """Kiểm tra hàm tokenize"""

    def test_tokenize_simple(self):
        """Tokenize câu đơn giản"""
        self.assertEqual(tokenize("machine learning is fun"), ["machine", "learning", "is", "fun"])

    def test_tokenize_with_extra_spaces(self):
        """Tokenize với nhiều khoảng trắng"""
        self.assertEqual(tokenize("a    b   c"), ["a", "b", "c"])

    def test_tokenize_empty_string(self):
        """Tokenize chuỗi rỗng"""
        self.assertEqual(tokenize(""), [])


class TestPreprocessPipeline(unittest.TestCase):
    """Kiểm tra pipeline tiền xử lý hoàn chỉnh"""

    def test_pipeline_basic(self):
        """Pipeline với email spam điển hình"""
        raw = "FREE MONEY!!! Call 0901234567 or visit https://fake.com"
        result = preprocess_pipeline(raw)
        # Kết quả mong đợi: lowercase, xóa dấu câu, xóa URL, giữ số
        expected = ["free", "money", "call", "0901234567", "or", "visit"]
        self.assertEqual(result, expected)

    def test_pipeline_with_html(self):
        """Pipeline với nội dung HTML"""
        raw = "<b>URGENT</b> Your account <i>will be closed</i>"
        result = preprocess_pipeline(raw)
        self.assertNotIn("<b>", " ".join(result))
        self.assertIn("urgent", result)
        self.assertIn("your", result)
        self.assertIn("account", result)

    def test_pipeline_with_email(self):
        """Pipeline xóa email"""
        raw = "Contact me@example.com for details"
        result = preprocess_pipeline(raw)
        self.assertNotIn("me@example.com", " ".join(result))
        self.assertIn("contact", result)
        self.assertIn("for", result)
        self.assertIn("details", result)

    def test_pipeline_keep_numbers(self):
        """Pipeline vẫn giữ số (phục vụ nhận diện spam)"""
        raw = "You won $1000000. Call 123456789"
        result = preprocess_pipeline(raw)
        self.assertIn("1000000", result)
        self.assertIn("123456789", result)

    def test_pipeline_empty_result(self):
        """Pipeline với text chỉ toàn URL sẽ bị xóa hết -> tokenize thành []"""
        raw = "https://example.com"
        result = preprocess_pipeline(raw)
        self.assertEqual(result, [])

    def test_pipeline_non_string_input(self):
        """Pipeline với đầu vào không phải string (int, float) -> vẫn xử lý"""
        raw = 12345
        result = preprocess_pipeline(raw)
        self.assertEqual(result, ["12345"])

    def test_pipeline_vietnamese(self):
        """Pipeline với tiếng Việt có dấu (không cần bỏ dấu, chỉ chuẩn hóa cơ bản)"""
        raw = "Chúc mừng bạn đã trúng thưởng 100 triệu! Liên hệ ngay 0909123456"
        result = preprocess_pipeline(raw)
        # Dấu câu bị xóa, chữ thường, số giữ lại
        self.assertIn("chúc", result)
        self.assertIn("mừng", result)
        self.assertIn("bạn", result)
        self.assertIn("100", result)
        self.assertIn("0909123456", result)


if __name__ == "__main__":
    # Chạy tất cả test với độ chi tiết cao
    unittest.main(verbosity=2)