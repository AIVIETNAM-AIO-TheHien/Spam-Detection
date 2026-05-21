# Augmentation Plan

## Domains to Add

| Domain | Priority | Quantity | Description |
| :--- | :---: | ---: | :--- |
| spam_phishing | cao | 1000 | Email lua dao gia mao ngan hang, PayPal, yeu cau xac minh tai khoan. |
| spam_lottery | cao | 800 | Email trung thuong, xo so, qua tang mien phi, ke thua tai san. |
| spam_product | trung binh | 800 | Quang cao san pham khong mong muon nhu thuoc, chung khoan, hoac san pham nhay cam. |
| spam_mlm | trung binh | 500 | Email da cap, lam giau nhanh, co hoi kinh doanh. |
| ham_english | cao | 1500 | Email hop phap bang tieng Anh de can bang tap du lieu. |

Total planned samples: 4600

## Techniques

### back_translation

- Tools: transformers (Helsinki-NLP/opus-mt), googletrans
- Cost: Mien phi neu chay model mo.
- Expected output: Moi cau goc sinh 2-3 cau bien the.

### llm_synthetic

- Tools: Google AI Studio hoac OpenAI API
- Cost: Tuy provider va quota.
- Prompt example: `Generate 10 spam emails about lottery scams with urgent tone and fake prize claims`

### paraphrase

- Tools: pegasus, t5-paraphrase, parrot
- Cost: Mien phi neu chay local.
- Note: Can GPU khi sinh so luong lon.

### rule_based_noise

- Tools: re, faker
- Cost: Mien phi.
- Note: Phu hop de tao bien the co kiem soat tu du lieu goc.

## Timeline

- Week 2: Collect 200-300 seed samples per domain from public sources.
- Week 3: Apply back-translation and paraphrase.
- Week 4: Generate with LLM and manually check quality.

## Notes

- Manually verify 10% of generated samples.
- Keep source metadata for generated samples.
