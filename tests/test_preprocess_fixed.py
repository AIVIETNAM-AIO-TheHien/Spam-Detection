"""
Module tiền xử lý dữ liệu spam detection.
Đầu ra: một dòng text duy nhất đã làm sạch, sẵn sàng cho model.
"""

import re
import string
from typing import List, Optional

# 1. Token hóa thực thể (Entity Tokenization)
def normalize_entities(text: str) -> str:
    """
    Thay thế các thực thể cụ thể bằng token đại diện.
    - URL -> [URL]
    - Email -> [EMAIL]
    - Số điện thoại -> [PHONE]
    - Số tiền -> [MONEY]
    """
    # URL
    text = re.sub(r'http\S+|www\.\S+|https\S+', ' [URL] ', text, flags=re.IGNORECASE)
    # Email
    text = re.sub(r'\S+@\S+', ' [EMAIL] ', text)
    # Số điện thoại (VN và quốc tế)
    phone_pattern = r'(\+?84|0)[\s.-]?([0-9]{2,3})[\s.-]?([0-9]{3})[\s.-]?([0-9]{3,4})'
    text = re.sub(phone_pattern, ' [PHONE] ', text)
    text = re.sub(r'\b\d{8,11}\b', ' [PHONE] ', text)
    # Số tiền
    money_pattern = r'(\$|€|£)?\s*\d+(?:[\.,]\d+)?\s*(triệu|nghìn|million|thousand)?'
    text = re.sub(money_pattern, ' [MONEY] ', text, flags=re.IGNORECASE)
    return text

# 2. Loại bỏ nhiễu hệ thống (Noise Stripping)
def remove_system_noise(text: str) -> str:
    """
    Xóa rác kỹ thuật: exmh ID, PGP signature, header, base64.
    """
    # Xóa exmh IDs
    text = re.sub(r'_exmh_[a-zA-Z0-9]+', '', text)
    # Xóa PGP signature blocks
    text = re.sub(r'-----BEGIN PGP.*?-----END PGP.*?-----', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Xóa header dòng
    text = re.sub(r'^(date|from|to|subject|message-id|content-type|charset):.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Xóa chuỗi base64 dài (bao gồm cả dấu =)
    text = re.sub(r'\b[a-zA-Z0-9+/=]{40,}\b', '', text)
    return text

# 3. Xóa boilerplate mailing list
def remove_mailing_list_boilerplate(text: str) -> str:
    """
    Xóa footer phổ biến của mailing list.
    """
    patterns = [
        r'irish linux users group.*?list maintainer.*?(?:\n|$)',
        r'to unsubscribe from this group.*?(?:\n|$)',
        r'yahoo! groups sponsor.*?(?:\n|$)',
        r'please read the faq.*?(?:\n|$)',
        r'_______________________________________________.*?(?:\n|$)',
        r'^--\s*$.*?\n',
        r'^_+.*$',
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

# Hàm giữ lại dấu ngoặc vuông khi xóa dấu câu
def _remove_punctuation_keep_brackets(text: str) -> str:
    """Xóa dấu câu nhưng giữ lại [ và ]."""
    keep = set('[]')
    return ''.join(ch for ch in text if ch not in string.punctuation or ch in keep)

# 5. Hàm chính làm sạch văn bản
def clean_text(
    text: str,
    lower: bool = True,
    remove_punct: bool = True,
    remove_numbers: bool = False,
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

    # 1. Chuẩn hóa thực thể (URL, email, phone, money) – ưu tiên trước khi xóa dấu câu
    if normalize_entities_flag:
        text = normalize_entities(text)

    # 2. Loại bỏ nhiễu hệ thống
    if remove_noise_flag:
        text = remove_system_noise(text)

    # 3. Xóa footer mailing list
    if remove_boilerplate_flag:
        text = remove_mailing_list_boilerplate(text)

    # 4. Xóa thẻ HTML (nếu còn)
    text = re.sub(r'<.*?>', '', text)

    # 5. Xóa dấu câu nhưng giữ [ và ]
    if remove_punct:
        text = _remove_punctuation_keep_brackets(text)

    # 6. Xóa số (nếu yêu cầu) – nhưng token [PHONE] và [MONEY] không bị ảnh hưởng (không chứa số)
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # 7. Chuyển về chữ thường
    if lower:
        text = text.lower()

    # 8. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # 9. Cắt ngắn nếu cần
    if max_words > 0:
        text = truncate_text(text, max_words)

    return text

def preprocess_email(subject: Optional[str] = None, body: Optional[str] = None, text: Optional[str] = None, **kwargs) -> str:
    """Xử lý email có subject và body riêng, hoặc văn bản đã gộp."""
    if text is not None:
        combined = text
    else:
        subject = subject or ""
        body = body or ""
        combined = subject + " " + body
    return clean_text(combined, **kwargs)

def preprocess_pipeline(text: str) -> str:
    """Pipeline đơn giản cho SMS / comment."""
    return clean_text(text)

def report_length_stats(texts: List[str]) -> None:
    """In thống kê độ dài văn bản (số từ)."""
    if not texts:
        print("Empty list")
        return
    lengths = [len(t.split()) for t in texts]
    print(f"Length stats (words): min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}, p95={sorted(lengths)[int(0.95*len(lengths))]}")