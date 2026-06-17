# TrustSeal — AI-Powered ID Verification for Notaries

The Application is live at : https://plasma-motivator-peso.ngrok-free.dev/static/index.html


**Built at VillageHacks 2026 · Notary Everyday Track**

TrustSeal is a multi-provider AI pipeline that verifies identity documents for notaries — replacing the manual, inconsistent process of squinting at a driver's license with a structured, scored, auditable verification report.

---

## What it does

### Stage 1 — Primary document verification
Upload the front and back of a Driver License (or other ID). Three agents independently analyse it:

| Agent | Provider | Task |
|---|---|---|
| Agent 1 | Anthropic Claude | 
| Agent 2 | Anthropic Claude |

Each agent does **everything** in one pass:
1. Identifies the issuing state and document type
2. Checks layout, zones, and placement against the **AAMVA 2020 standard** (loaded from local DB)
3. Extracts every visible field with strict formatting rules
4. Decodes the PDF417 barcode (locally, no AI) and compares every extracted field vs barcode
5. Scores: barcode match = +1, mismatch = −1
6. Returns a complete scored JSON

The Judge then polls: if both agents agree → high confidence. If they conflict → Judge checks the original image and decides. Barcode is ground truth.

### Stage 2 — Cross-document verification
Upload a second document (utility bill, bank statement, loan doc). Same two-agent + judge pattern extracts its identity fields, then cross-matches against the verified ID fields.

### Output
- Field poll table showing agreement level per field and source (agents / barcode / judge image check)
- Barcode match % with per-field mismatch breakdown
- Rule-based checks from AAMVA standard (deterministic, no AI)
- Downloadable full verification report JSON
- Final confidence score and APPROVE / REVIEW / REJECT recommendation

---

## Architecture

```
notary_verify/
├── main.py            # FastAPI backend
├── agents.py          # Multi-provider agent pipeline
├── rules_engine.py    # Deterministic rule checks (no AI)
├── standards/
│   ├── aamva_universal.json    # AAMVA 2020 universal standard (all 50 states)
│   ├── passport_icao.json      # ICAO Doc 9303 passport standard
│   └── states/
│       └── AZ.json             # Arizona-specific rules
└── static/
    └── index.html     # Frontend UI
```

### Provider assignment

```
→ Agent 1 (Claude)    — primary analysis
→ Agent 2 (Claude)   — independent redundant analysis  

```

---

## Standards DB

The `standards/` directory is a local database built from:
- **AAMVA 2020 DL/ID Card Design Standard** — mandatory fields, zone placements, barcode element IDs (DCS, DAC, DBB...), physical dimensions, security feature requirements
- **ICAO Doc 9303** — passport MRZ structure and check digit rules

This means the rule-based checks are **deterministic and auditable** — no AI needed to verify that a date is in the wrong format or that the portrait is on the wrong side.

---

## Setup

### Prerequisites
```bash
pip install fastapi uvicorn python-multipart anthropic openai zxing-cpp Pillow opencv-python-headless numpy
```

### Environment variables
Copy `.env.example` to `.env` and fill in your API keys:
```
```

### Run
```bash
# Test your API keys first
python test_keys.py

# Start the server
uvicorn main:app --reload --port 8000

# Open the UI
open http://localhost:8000
```

---

## Barcode scanning

PDF417 barcodes are decoded **entirely locally** using `zxing-cpp` — no image data leaves your machine for barcode processing. The scanner handles:
- Rotated images (0°, 90°, 180°, 270°)
- Low-contrast images (CLAHE preprocessing)
- Low-resolution images (2x upscale)
- Glare detection with user-friendly re-upload prompts

---

## PII handling

| Operation | Where | PII exposure |
|---|---|---|
| Barcode decode | Local Python | Zero |
| Rule-based checks | Local Python | Zero |
| Visual AI analysis | Claude / OpenAI APIs | Image sent — Anthropic zero data retention by default |
| Cross-match logic | Local Python | Zero |

---

## Demo

Tested with Arizona Driver Licenses. The system:
- Correctly identifies LIMITED-TERM status (DDD=1 in barcode)
- Flags glare-obscured barcodes with specific re-upload instructions
- Detects address conflicts between extracted text and barcode data
- Falls back to working providers when a key fails

---

## Built with

- **FastAPI** — backend
- **Anthropic Claude** — Agent 1,2 (vision + analysis)
- **zxing-cpp** — PDF417 barcode decoding
- **AAMVA 2020 Standard** — ground truth for all rule checks

---

