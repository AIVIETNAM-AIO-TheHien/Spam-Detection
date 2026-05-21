#!/usr/bin/env python3
"""
Kịch bản tạo kế hoạch làm giàu dữ liệu (augmentation) cho bài toán phát hiện spam.
Mục tiêu: Bổ sung các domain spam tiếng Anh còn thiếu và cân bằng lớp.

Chạy: python scripts/augment_data.py
"""

import json
import os

# Các domain spam tiếng Anh cần bổ sung (dựa trên điểm yếu của dataset hiện tại)
MISSING_DOMAINS = {
    "spam_phishing": {
        "description": "Email lừa đảo (giả mạo ngân hàng, PayPal, yêu cầu xác minh tài khoản, v.v.)",
        "quantity": 1000,
        "priority": "cao"
    },
    "spam_lottery": {
        "description": "Email trúng thưởng, xổ số, quà tặng miễn phí, kế thừa tài sản",
        "quantity": 800,
        "priority": "cao"
    },
    "spam_product": {
        "description": "Quảng cáo sản phẩm không mong muốn (thuốc, tình dục, chứng khoán, v.v.)",
        "quantity": 800,
        "priority": "trung bình"
    },
    "spam_mlm": {
        "description": "Email đa cấp, làm giàu nhanh, cơ hội kinh doanh",
        "quantity": 500,
        "priority": "trung bình"
    },
    "ham_english": {
        "description": "Email hợp pháp bằng tiếng Anh (đối trọng để tránh mất cân bằng)",
        "quantity": 1500,
        "priority": "cao"
    }
}

# Các kỹ thuật augmentation sẽ sử dụng (áp dụng cho văn bản tiếng Anh)
AUGMENTATION_TECHNIQUES = {
    "back_translation": {
        "description": "Dịch văn bản qua lại giữa tiếng Anh và một ngôn ngữ khác (ví dụ: Pháp, Đức) để tạo biến thể ngữ nghĩa",
        "libraries": ["transformers (Helsinki-NLP/opus-mt)", "googletrans"],
        "cost": "Miễn phí (dùng model mở)",
        "expected_output": "Mỗi câu gốc sinh ra 2-3 câu mới"
    },
    "llm_synthetic": {
        "description": "Dùng mô hình ngôn ngữ lớn (GPT, Gemini) để sinh email spam mới từ prompt",
        "prompt_example": "Generate 10 spam emails about lottery scams with urgent tone and fake prize claims",
        "cost": "~$5/5000 mẫu (Gemini free tier có giới hạn)",
        "tool": "Google AI Studio hoặc OpenAI API"
    },
    "paraphrase": {
        "description": "Dùng mô hình paraphrase (T5, Pegasus) để diễn đạt lại câu nhưng giữ nguyên ý nghĩa",
        "libraries": ["pegasus", "t5-paraphrase", "parrot"],
        "cost": "Miễn phí (chạy local)",
        "note": "Cần GPU nếu số lượng lớn"
    },
    "rule_based_noise": {
        "description": "Thêm nhiễu có kiểm soát: thay thế URL, số điện thoại, tên công ty, email giả",
        "libraries": ["re", "faker"],
        "cost": "Miễn phí",
        "note": "Phù hợp để tạo thêm biến thể từ dữ liệu gốc"
    }
}

def generate_augmentation_plan():
    print("=" * 70)
    print("KẾ HOẠCH LÀM GIÀU DỮ LIỆU - SPAM DETECTION (TIẾNG ANH)")
    print("=" * 70)
    
    print("\n1. CÁC DOMAIN CẦN BỔ SUNG:")
    total = 0
    for domain, info in MISSING_DOMAINS.items():
        qty = info["quantity"]
        total += qty
        print(f"   - {domain} ({info['priority']}): {qty} mẫu")
        print(f"     Mô tả: {info['description']}")
    
    print(f"\n   📊 Tổng số mẫu dự kiến bổ sung: {total}")
    
    print("\n2. KỸ THUẬT AUGMENTATION SẼ SỬ DỤNG:")
    for tech, details in AUGMENTATION_TECHNIQUES.items():
        print(f"\n   🔹 {tech.upper()}")
        print(f"      - {details['description']}")
        print(f"      - Thư viện/Công cụ: {', '.join(details['libraries']) if 'libraries' in details else details.get('tool', 'N/A')}")
        if 'cost' in details:
            print(f"      - Chi phí ước tính: {details['cost']}")
        if 'note' in details:
            print(f"      - Lưu ý: {details['note']}")
    
    print("\n3. LỘ TRÌNH THỰC HIỆN:")
    print("   - Tuần 2: Thu thập 200-300 mẫu seed cho mỗi domain từ các nguồn công khai")
    print("   - Tuần 3: Áp dụng back-translation và paraphrase để nhân bản")
    print("   - Tuần 4: Sử dụng LLM để sinh thêm mẫu mới, kiểm tra chất lượng bằng tay")

    os.makedirs("configs", exist_ok=True)
    plan = {
        "domains_to_add": MISSING_DOMAINS,
        "techniques": AUGMENTATION_TECHNIQUES,
        "total_samples": total,
        "timeline": {
            "week2": "Collect 200-300 seed samples per domain from public sources",
            "week3": "Apply back-translation and paraphrase",
            "week4": "Generate with LLM and manual quality check",
        },
        "notes": "Manually verify 10% of generated samples; keep source metadata.",
    }
    with open("configs/augmentation_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    
    os.makedirs("docs", exist_ok=True)
    with open("docs/augmentation_plan.md", "w", encoding="utf-8") as f:
        f.write("# Augmentation Plan\n\n")
        f.write("## Domains to Add\n\n")
        f.write("| Domain | Priority | Quantity | Description |\n")
        f.write("| :--- | :---: | ---: | :--- |\n")
        for domain, info in MISSING_DOMAINS.items():
            f.write(
                f"| {domain} | {info['priority']} | {info['quantity']} | "
                f"{info['description']} |\n"
            )

        f.write(f"\nTotal planned samples: {total}\n\n")
        f.write("## Techniques\n\n")
        for tech, details in AUGMENTATION_TECHNIQUES.items():
            f.write(f"### {tech}\n\n")
            f.write(f"- Description: {details['description']}\n")
            tools = ", ".join(details["libraries"]) if "libraries" in details else details.get("tool", "N/A")
            f.write(f"- Tools: {tools}\n")
            if "cost" in details:
                f.write(f"- Cost: {details['cost']}\n")
            if "expected_output" in details:
                f.write(f"- Expected output: {details['expected_output']}\n")
            if "note" in details:
                f.write(f"- Note: {details['note']}\n")
            if "prompt_example" in details:
                f.write(f"- Prompt example: `{details['prompt_example']}`\n")
            f.write("\n")

        f.write("## Timeline\n\n")
        f.write("- Week 2: Collect 200-300 seed samples per domain from public sources.\n")
        f.write("- Week 3: Apply back-translation and paraphrase.\n")
        f.write("- Week 4: Generate with LLM and manually check quality.\n\n")
        f.write("## Notes\n\n")
        f.write("- Manually verify 10% of generated samples.\n")
        f.write("- Keep source metadata for generated samples.\n")

    print("\n✅ Đã lưu kế hoạch vào configs/augmentation_plan.json và docs/augmentation_plan.md")

if __name__ == "__main__":
    generate_augmentation_plan()
