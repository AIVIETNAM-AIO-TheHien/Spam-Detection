# Guide to Collecting Real‑World English Spam Data

## Purpose
Supplement existing datasets with fresh English spam examples from domains not well covered (e.g., phishing, lottery scams, MLM, fake product reviews) and add diverse English ham (legitimate messages).

## Public Data Sources

| Spam Type | Source | Collection Method | Target |
|-----------|--------|-------------------|--------|
| **Phishing emails** | PhishTank, OpenPhish, Kaggle Phishing Datasets | Download public CSV/JSON (API available) | 500+ |
| **Lottery/Prize scams** | SpamAssassin public corpus, 419scam.org | Direct download or manual copy from public forums | 300 |
| **MLM / Get rich quick** | Reddit (r/antiMLM, r/scams), public Facebook groups | Reddit API (pushshift), manual extraction | 200 |
| **Fake product reviews** | Amazon, Trustpilot (public reviews flagged as fake) | Pre‑existing datasets (e.g., Fake Reviews Dataset on Kaggle) | 500 |
| **General English ham** | Enron email corpus, Apache public mailing lists, NEWSGROUP datasets | Download from open repositories | 1000+ |

## Detailed Collection Steps

1. **Phishing emails**  
   - Use PhishTank API to fetch verified phishing URLs and associated emails (if available).  
   - Alternatively, download ready‑to‑use datasets from Kaggle: "Phishing Email Dataset" or "CEAS 2008".

2. **Lottery/419 scams**  
   - The SpamAssassin corpus (already in the project) contains many 419 samples.  
   - Supplement with manually collected emails from online forums like "Scamwarners".

3. **MLM / pyramid schemes**  
   - Use Reddit’s Pushshift API to query comments and posts from subreddits like `r/antiMLM` (look for quoted messages).  
   - Ensure only publicly available content is collected; no private messages.

4. **Fake product reviews**  
   - Use existing academic datasets: "Fake Reviews Dataset" (UCI) or "Deceptive Opinion Spam Corpus".  
   - These are already labeled and can be merged directly.

5. **English ham (legitimate)**  
   - Enron email corpus (∼500k emails) provides a large source of real business communications.  
   - Apache mailing lists (e.g., SpamAssassin‑talk, Exmh‑workers) are also public and contain clean discussions.

## Legal & Ethical Notes

- Only collect **publicly available** data. Do not scrape private inboxes or require authentication.
- Respect robots.txt and rate limits (use APIs where available).
- For Reddit/Facebook, only collect from **public groups/pages** and respect the platform’s terms of service.
- Anonymize any personal information (names, emails, phone numbers) before storing.

## Output Format

Store collected data as JSON or CSV with at least two columns: `text` (the message content) and `label` (`spam` or `ham`). Include optional metadata like `source` and `date`.

Example:
```json
{
  "text": "You have won $1,000,000! Click here to claim your prize now.",
  "label": "spam",
  "source": "phishTank_sample_123",
  "date": "2025-05-14"
}