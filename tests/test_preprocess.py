"""
Unit test cho module src.data.preprocess
Kiểm tra các hàm clean_text, tokenize, preprocess_pipeline
"""

import unittest
import sys
import os

# Thêm đường dẫn src vào sys.path để import được module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocess import clean_text, tokenize, preprocess_pipeline

class TestCleanText(unittest.TestCase):
    """Kiểm tra hàm clean_text với các tùy chọn khác nhau"""
    
    def test_lowercase(self):
        """Test chuyển đổi chữ thường"""
        result = clean_text("Hello WORLD", lowercase=True)
        self.assertEqual(result, "hello world")
    
    def test_no_lowercase(self):
        """Test giữ nguyên chữ hoa khi lowercase=False"""
        result = clean_text("Hello WORLD", lowercase=False)
        self.assertEqual(result, "Hello WORLD")
    
    def test_remove_urls(self):
        """Test xóa URL (http, https, www)"""
        text = "Check https://example.com and http://test.com or www.google.com"
        result = clean_text(text, remove_urls=True)
        self.assertNotIn("https://", result)
        self.assertNotIn("http://", result)
        self.assertNotIn("www.", result)
        self.assertIn("Check and or", result)
    
    def test_remove_emails(self):
        """Test xóa địa chỉ email"""
        text = "Contact me@example.com or support@test.org for help"
        result = clean_text(text, remove_emails=True)
        self.assertNotIn("@", result)
        self.assertIn("Contact or for help", result)
    
    def test_remove_punctuation(self):
        """Test xóa dấu câu"""
        text = "Hello!!! How are you? I'm fine, thanks."
        result = clean_text(text, remove_punct=True)
        self.assertEqual(result, "Hello How are you Im fine thanks")
    
    def test_remove_html(self):
        """Test xóa thẻ HTML"""
        text = "<html><body>Hello <b>World</b></body></html>"
        result = clean_text(text, remove_html=True)
        self.assertEqual(result, "Hello World")
    
    def test_remove_numbers(self):
        """Test xóa số (khi bật tùy chọn)"""
        text = "Call 0901234567 or 123 for support"
        result = clean_text(text, remove_numbers=True)
        self.assertNotIn("0901234567", result)
        self.assertNotIn("123", result)
        self.assertIn("Call or for support", result)
    
    def test_keep_numbers_by_default(self):
        """Test mặc định giữ lại số (quan trọng cho spam)"""
        text = "Call 0901234567 now"
        result = clean_text(text, remove_numbers=False)
        self.assertIn("0901234567", result)
    
    def test_combination_of_options(self):
        """Test kết hợp nhiều tùy chọn cùng lúc"""
        text = "Check https://spam.com and email me@abc.com!!! WINNER 100K"
        result = clean_text(text, lowercase=True, remove_urls=True, 
                           remove_emails=True, remove_punct=True, 
                           remove_html=True, remove_numbers=False)
        self.assertEqual(result, "check and winner 100k")

class TestTokenize(unittest.TestCase):
    """Kiểm tra hàm tokenize"""
    
    def test_tokenize_simple(self):
        """Test tokenize câu đơn giản"""
        result = tokenize("machine learning is fun")
        self.assertEqual(result, ["machine", "learning", "is", "fun"])
    
    def test_tokenize_with_extra_spaces(self):
        """Test tokenize với nhiều khoảng trắng"""
        result = tokenize("a    b   c")
        self.assertEqual(result, ["a", "b", "c"])
    
    def test_tokenize_empty_string(self):
        """Test tokenize chuỗi rỗng"""
        result = tokenize("")
        self.assertEqual(result, [])

class TestPreprocessPipeline(unittest.TestCase):
    """Kiểm tra pipeline tiền xử lý hoàn chỉnh"""
    
    def test_pipeline_basic(self):
        """Test pipeline với email spam điển hình"""
        raw = "FREE MONEY!!! Call 0901234567 or visit https://fake.com"
        result = preprocess_pipeline(raw)
        # Kết quả mong đợi: lowercase, xóa dấu câu, xóa URL, giữ số
        expected = ["free", "money", "call", "0901234567", "or", "visit"]
        self.assertEqual(result, expected)
    
    def test_pipeline_with_html(self):
        """Test pipeline với nội dung HTML"""
        raw = "<b>URGENT</b> Your account <i>will be closed</i>"
        result = preprocess_pipeline(raw)
        self.assertNotIn("<b>", " ".join(result))
        self.assertIn("urgent", result)
        self.assertIn("your", result)
        self.assertIn("account", result)
    
    def test_pipeline_with_email(self):
        """Test pipeline xóa email"""
        raw = "Contact me@example.com for details"
        result = preprocess_pipeline(raw)
        self.assertNotIn("me@example.com", " ".join(result))
        self.assertIn("contact", result)
        self.assertIn("for", result)
        self.assertIn("details", result)
    
    def test_pipeline_keep_numbers(self):
        """Test pipeline vẫn giữ số (phục vụ nhận diện spam)"""
        raw = "You won $1000000. Call 123456789"
        result = preprocess_pipeline(raw)
        self.assertIn("1000000", result)
        self.assertIn("123456789", result)
    
    def test_pipeline_empty_result(self):
        """Test pipeline với text rỗng hoặc chỉ toàn ký tự bị loại"""
        raw = "https://example.com"
        result = preprocess_pipeline(raw)
        # Sau khi xóa URL, text trống -> tokenize trả về []
        self.assertEqual(result, [])
    
    def test_pipeline_non_string_input(self):
        """Test pipeline với đầu vào không phải string (float, int)"""
        raw = 12345
        result = preprocess_pipeline(raw)
        self.assertEqual(result, ["12345"])  # Chuyển thành string và giữ số

if __name__ == "__main__":
    # Chạy tất cả các test
    unittest.main(verbosity=2)
