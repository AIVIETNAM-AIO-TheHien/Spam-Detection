#!/usr/bin/env python3
"""
Data Augmentation Plan for Spam Detection.
Target: Add missing Vietnamese spam domains and balance classes.

"""

import json
import os

# Missing domains and required sample counts
MISSING_DOMAINS = {
    "spam_real_estate": {
        "description": "Messages/emails about property sales, rentals, project promotions",
        "quantity": 500,
        "priority": "high"
    },
    "spam_lottery_scam": {
        "description": "Prize scams, lottery wins, gift cards",
        "quantity": 500,
        "priority": "high"
    },
    "spam_mlm": {
        "description": "Multi‑level marketing invitations, 'get rich quick' schemes",
        "quantity": 500,
        "priority": "medium"
    },
    "spam_fake_courses": {
        "description": "Fake course ads, fake certificates, exam cheating offers",
        "quantity": 500,
        "priority": "medium"
    },
    "ham_vietnamese": {
        "description": "Normal Vietnamese messages (counterbalance)",
        "quantity": 1000,
        "priority": "high"
    }
}

AUGMENTATION_TECHNIQUES = {
    "back_translation": {
        "description": "Translate text back and forth between English and Vietnamese to create semantic variants",
        "libraries": ["transformers (Helsinki-NLP/opus-mt)", "googletrans"],
        "cost": "Free (open models)",
        "expected_output": "2‑3 variants per original sentence"
    },
    "llm_synthetic": {
        "description": "Use LLM (Gemini/GPT) to generate new spam samples from prompts",
        "prompt_example": "Generate 10 Vietnamese spam messages about real estate: 'urgent sale, cheap apartment, call now'",
        "cost": "~$5 per 5000 samples (Gemini free tier limited)",
        "tool": "Google AI Studio or OpenAI API"
    },
    "paraphrase": {
        "description": "Use paraphrase models to rephrase sentences while keeping meaning",
        "libraries": ["pegasus", "t5-paraphrase", "parrot"],
        "cost": "Free (local execution)",
        "note": "Requires GPU for large volumes"
    },
    "rule_based_noise": {
        "description": "Add controlled noise: replace phone numbers, company names, fake URLs",
        "libraries": ["re", "faker"],
        "cost": "Free",
        "note": "Good for generating variants from existing data"
    }
}

def generate_augmentation_plan():
    print("=" * 70)
    print("DATA AUGMENTATION PLAN - SPAM DETECTION")
    print("=" * 70)
    
    print("\n1. DOMAINS TO ADD:")
    total = 0
    for domain, info in MISSING_DOMAINS.items():
        qty = info["quantity"]
        total += qty
        print(f"   - {domain} ({info['priority']}): {qty} samples")
        print(f"     Description: {info['description']}")
    
    print(f"\n   📊 Total planned samples: {total}")
    
    print("\n2. AUGMENTATION TECHNIQUES:")
    for tech, details in AUGMENTATION_TECHNIQUES.items():
        print(f"\n   🔹 {tech.upper()}")
        print(f"      - {details['description']}")
        if 'libraries' in details:
            print(f"      - Libraries: {', '.join(details['libraries'])}")
        if 'tool' in details:
            print(f"      - Tool: {details['tool']}")
        print(f"      - Cost estimate: {details['cost']}")
        if 'note' in details:
            print(f"      - Note: {details['note']}")
    
    print("\n3. TIMELINE:")
    print("   - Week 2: Manually collect 100 seed samples per domain")
    print("   - Week 3: Apply back‑translation and paraphrase to expand")
    print("   - Week 4: Generate new samples with LLM, manual quality check")
    
    # Save plan to JSON
    plan = {
        "domains_to_add": MISSING_DOMAINS,
        "techniques": AUGMENTATION_TECHNIQUES,
        "total_samples": total,
        "timeline": {
            "week2": "Collect 100 seed samples per domain",
            "week3": "Apply back‑translation and paraphrase",
            "week4": "Generate with LLM and manual quality check"
        },
        "notes": "Manually verify 10% of generated samples; keep source metadata for each sample."
    }
    
    os.makedirs("configs", exist_ok=True)
    with open("configs/augmentation_plan.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    
    print("\n Detailed plan saved to configs/augmentation_plan.json")

if __name__ == "__main__":
    generate_augmentation_plan()