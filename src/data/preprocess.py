"""
Module tiền xử lý dữ liệu spam detection.
Đầu ra: một dòng text duy nhất đã làm sạch, sẵn sàng cho model.
"""

import re
import string
from typing import List, Optional, Union

# 1. Token hóa thực thể (Entity Tokenization)
def normalize_entities(text: str) -> str:
    """
    Thay thế các thực thể cụ thể bằng token đại diện.
    - Số điện thoại (VN và US) -> [PHONE]
    - URL -> [URL]
    - Email -> [EMAIL]
    - Số tiền (có dấu $, €, £, hoặc kèm 'triệu', 'nghìn') -> [MONEY]
    """
    # Email
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', ' [EMAIL] ', text)
    # URL
    text = re.sub(r'https?://\S*|www\.\S+', ' [URL] ', text, flags=re.IGNORECASE)
    # Số điện thoại (dạng 10-11 chữ số, có thể có +84, 0, dấu cách, dấu gạch)
    phone_pattern = r'(\+?84|0)[\s.-]?([0-9]{2,3})[\s.-]?([0-9]{3})[\s.-]?([0-9]{3,4})'
    text = re.sub(phone_pattern, ' [PHONE] ', text)
    # Số điện thoại đơn giản 8-11 chữ số liền
    text = re.sub(r'\b\d{8,11}\b', ' [PHONE] ', text)
    # Số tiền: $, €, £, hoặc số + "triệu/nghìn"
    currency_symbol = r'(?:\$|€|£|â‚¬|Â£)'
    money_unit = r'(?:triệu|nghìn|triá»‡u|nghÃ¬n|million|thousand|dollars?|usd|vnd)'
    money_pattern = rf'(?:{currency_symbol}\s*\d+(?:[\.,]\d+)?|\b\d+(?:[\.,]\d+)?\s*{money_unit}\b)'
    text = re.sub(money_pattern, ' [MONEY] ', text, flags=re.IGNORECASE)
    return text

# 2. Loại bỏ nhiễu hệ thống (Noise Stripping)

def remove_system_noise(text: str) -> str:
    """
    Xóa các rác kỹ thuật:
    - ID phiên exmh (_exmh_12345678P)
    - PGP signature blocks
    - Base64 strings (dài > 40 ký tự không dấu cách)
    - Header lines (date, from, subject, etc.)
    """
    # Xóa exmh IDs
    text = re.sub(r'_exmh_[a-zA-Z0-9]+', '', text)
    # Xóa PGP signature blocks (bắt đầu bằng -----BEGIN PGP...)
    text = re.sub(r'-----BEGIN PGP.*?-----END PGP.*?-----', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Xóa các dòng header phổ biến (Date:, From:, To:, Subject:, Message-ID:...)
    text = re.sub(r'^(date|from|to|subject|message-id|content-type|charset):.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Xóa các chuỗi base64 dài (>40 ký tự liên tiếp không khoảng trắng)
    text = re.sub(r'\b[a-zA-Z0-9+/]{40,}\b', '', text)
    return text

# 3. Xóa boilerplate mailing list

def remove_mailing_list_boilerplate(text: str) -> str:
    """
    Xóa các footer phổ biến của mailing list.
    Dùng pattern linh hoạt để match các biến thể.
    """
    patterns = [
        r'irish linux users group.*?list maintainer.*?(?:\n|$)',
        r'irish linux users group.*?(?:\n|$)',
        r'to unsubscribe from this group.*?(?:\n|$)',
        r'yahoo! groups sponsor.*?(?:\n|$)',
        r'please read the faq.*?(?:\n|$)',
        r'_______________________________________________.*?(?:\n|$)',
        r'^--\s*$.*?\n',  # dòng -- có thể là separator
        r'^_+.*$',        # dòng gạch dưới
    ]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.DOTALL | re.IGNORECASE)
    return text


# 4. Kiểm soát độ dài

def truncate_text(text: str, max_words: int = 1024) -> str:
    """Cắt văn bản nếu số từ vượt quá max_words, giữ phần đầu."""
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
        text = ' '.join(words)
    return text

# 5. Hàm chính làm sạch văn bản

def clean_text(
    text: str,
    lower: bool = True,
    remove_punct: bool = True,
    remove_numbers: bool = False,   # mặc định giữ số (sẽ thay bằng token)
    normalize_entities_flag: bool = True,
    remove_noise_flag: bool = True,
    remove_boilerplate_flag: bool = True,
    max_words: int = 1024
) -> str:
    """
    Pipeline xử lý văn bản hoàn chỉnh.
    Đầu ra là một dòng string duy nhất đã được làm sạch.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Chuẩn hóa thực thể (URL, email, phone, money)
    if normalize_entities_flag:
        text = normalize_entities(text)

    # 2. Loại bỏ nhiễu hệ thống (PGP, exmh, base64, headers)
    if remove_noise_flag:
        text = remove_system_noise(text)

    # 3. Xóa footer mailing list
    if remove_boilerplate_flag:
        text = remove_mailing_list_boilerplate(text)

    # 4. Xóa thẻ HTML (nếu còn sót)
    text = re.sub(r'<.*?>', '', text)

    # 5. Xóa dấu câu (nếu yêu cầu)
    if remove_punct:
        punctuation_to_remove = string.punctuation.replace('[', '').replace(']', '')
        text = text.translate(str.maketrans('', '', punctuation_to_remove))

    # 6. Xóa số (nếu yêu cầu) - nhưng thường đã thay bằng token
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # 7. Chuyển về chữ thường
    if lower:
        text = text.lower()

    # 8. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Cắt độ dài nếu cần
    if max_words > 0:
        text = truncate_text(text, max_words)

    return text

def preprocess_email(subject: Optional[str] = None, body: Optional[str] = None, text: Optional[str] = None, **kwargs) -> str:
    """
    Xử lý email có riêng subject và body, hoặc văn bản đã gộp.
    Trả về một dòng text duy nhất.
    """
    if text is not None:
        combined = text
    else:
        subject = subject or ""
        body = body or ""
        combined = subject + " " + body
    return clean_text(combined, **kwargs)

def preprocess_pipeline(text: str) -> str:
    """
    Pipeline đơn giản cho văn bản đầu vào là string duy nhất (SMS, comment, ...).
    Trả về string đã làm sạch, sẵn sàng đưa vào model.
    """
    return clean_text(text)

# 6. Hàm tiện ích báo cáo độ dài

def report_length_stats(texts: List[str]) -> None:
    """In thống kê độ dài văn bản (số từ) của một danh sách."""
    if not texts:
        print("Empty list")
        return
    lengths = [len(t.split()) for t in texts]
    print(f"Length stats (words): min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}, p95={sorted(lengths)[int(0.95*len(lengths))]}")
