"""
agents.py — Single-provider pipeline using Anthropic Claude only.
Agent 1 → Claude (independent call)
Agent 2 → Claude (same prompt, independent call)
Judge   → Claude (receives both outputs + images, polls and decides)
"""

import anthropic
import base64
import json
import os
import re
from pathlib import Path
from typing import Optional
from rules_engine import RulesEngine, CrossDocumentMatcher, PDF417Parser

# ─── Single client (lazy) ─────────────────────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
        print(f"[client] Anthropic key loaded ({key[:12]}...)")
        _client = anthropic.Anthropic(api_key=key)
    return _client

CLAUDE_MODEL  = "claude-opus-4-5"
STANDARDS_DIR = Path(__file__).parent / "standards"


# ─── Image helper ─────────────────────────────────────────────────────────────
def _img(b: bytes, mt: str) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mt,
            "data": base64.standard_b64encode(b).decode()
        }
    }

def _clean(t: str) -> str:
    t = t.strip()
    t = re.sub(r'^```json\s*', '', t)
    t = re.sub(r'^```\s*',     '', t)
    t = re.sub(r'\s*```$',     '', t)
    return t.strip()


# ─── Single call wrapper ──────────────────────────────────────────────────────
def _call(system: str, content: list, label: str = "") -> dict:
    """Call Claude with vision. Returns parsed JSON dict."""
    client = _get_client()
    try:
        r = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": content}]
        )
        raw = r.content[0].text
        print(f"  [{label}] {raw[:100]}...")
        return json.loads(_clean(raw))
    except json.JSONDecodeError as e:
        raw_text = r.content[0].text if 'r' in dir() else ''
        print(f"  [{label} JSON ERR] {e} — raw: {raw_text[:200]}")
        # Try to extract JSON from prose
        m = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except:
                pass
        return {"error": f"JSON parse failed: {e}", "raw": raw_text[:200]}
    except Exception as e:
        print(f"  [{label} ERR] {e}")
        return {"error": str(e)}


# ─── Standards loader ─────────────────────────────────────────────────────────
def _standards_text(state: str = "AZ") -> str:
    universal   = {}
    state_rules = {}
    up = STANDARDS_DIR / "aamva_universal.json"
    sp = STANDARDS_DIR / "states" / f"{state.upper()}.json"
    if up.exists():    universal   = json.loads(up.read_text())
    if sp.exists():    state_rules = json.loads(sp.read_text())

    dims   = universal.get("physical", {}).get("dimensions_mm", {})
    fields = [f["label"] for f in universal.get("mandatory_human_readable_fields", [])]
    iin    = state_rules.get("iin", "")

    return f"""AAMVA 2020 AZ Driver License standard:
- Size: {dims.get('width',85.6)}mm x {dims.get('height',53.98)}mm (credit card)
- Portrait: MUST be on LEFT side
- Over 21: HORIZONTAL. Under 21: VERTICAL
- Mandatory fields: {', '.join(fields[:8])}
- IIN: {iin} | Zone I: state name + doc type"""


# ─── Agent prompt (identical for Agent 1 and Agent 2) ────────────────────────
_AGENT_SYSTEM = """You are an expert ID document verification agent for notary services.
Analyse the ID document image(s) and return ONLY valid JSON.
Be precise and conservative. Never guess — use empty string if not clearly visible."""

def _agent_prompt(standard_text: str, barcode_fields: Optional[dict],
                  back_provided: bool, doc_type: str) -> str:

    barcode_expected = doc_type.upper() in ("DL", "ID", "CDL", "UNKNOWN", "")

    if barcode_fields:
        bc = f"""BARCODE DATA (decoded locally — ground truth, prefer over visual read):
{json.dumps({k:v for k,v in list(barcode_fields.items())[:15]}, indent=2)}
If printed value differs from barcode for any field, note it in barcode_mismatches."""
    elif back_provided and barcode_expected:
        bc = "BARCODE: Back provided but could NOT be decoded. Flag — AZ DLs must have PDF417."
    elif not back_provided and barcode_expected:
        bc = "BARCODE: No back image — barcode check skipped."
    else:
        bc = "BARCODE: Not expected for this document type."

    return f"""Analyse this ID document. Return ONLY this JSON, nothing else:

STANDARD:
{standard_text}

{bc}

RULES: Dates MM/DD/YYYY | Names UPPERCASE | Eye: BRO/BLU/GRN/GRY/BLK/HAZ | Sex: M/F/X | Unknown: ""

{{
  "identified_state": "",
  "document_type": "DL or ID or CDL or PP or OTHER",
  "orientation": "horizontal or vertical",
  "portrait_on_left": true,
  "zone_i_text": "",
  "real_id_star": false,
  "limited_term": false,
  "family_name": "",
  "given_name": "",
  "middle_name": "",
  "dob": "",
  "issue_date": "",
  "expiry": "",
  "id_number": "",
  "address": "",
  "city": "",
  "state": "",
  "zip": "",
  "sex": "",
  "height": "",
  "weight": "",
  "eye_color": "",
  "hair_color": "",
  "vehicle_class": "",
  "restrictions": "",
  "endorsements": "",
  "veteran": false,
  "organ_donor": false,
  "barcode_mismatches": [],
  "uncertain_fields": [],
  "tamper_signs": [],
  "confidence": 0.0
}}"""


# ─── Judge prompt ─────────────────────────────────────────────────────────────
_JUDGE_SYSTEM = """You are the Judge in an ID verification pipeline.
You receive two independent analyses of the same document plus the original images.
Poll each field, resolve conflicts by examining the images yourself.
Return ONLY valid JSON."""

def _judge_prompt(r1: dict, r2: dict, barcode: Optional[dict]) -> str:
    bc = f"\nBARCODE GROUND TRUTH (100% reliable):\n{json.dumps(barcode, indent=2)}" if barcode else ""
    return f"""You have the original document image(s) above.{bc}

AGENT 1 output:
{json.dumps(r1, indent=2)}

AGENT 2 output:
{json.dumps(r2, indent=2)}

POLLING RULES:
1. Both agents agree → use that value, high confidence
2. Barcode has the field → barcode wins over both agents
3. Agents disagree → look at the image yourself, pick correct value
4. Agent says something not visible in image → ignore it, note as agent error

Return ONLY this JSON:
{{
  "identified_state": "",
  "document_type": "",
  "orientation": "",
  "final_extracted_fields": {{
    "family_name": "", "given_name": "", "middle_name": "",
    "dob": "", "issue_date": "", "expiry": "", "id_number": "",
    "address": "", "city": "", "state": "", "zip": "",
    "sex": "", "height": "", "weight": "", "eye_color": "", "hair_color": "",
    "vehicle_class": "", "restrictions": "", "endorsements": "",
    "portrait_on_left": true, "real_id_star": false, "limited_term": false,
    "veteran": false, "organ_donor": false
  }},
  "field_poll_results": {{
    "family_name": {{"value":"","agreement":"both","source":"agents"}},
    "given_name":  {{"value":"","agreement":"both","source":"agents"}},
    "dob":         {{"value":"","agreement":"both","source":"agents"}},
    "expiry":      {{"value":"","agreement":"both","source":"agents"}},
    "id_number":   {{"value":"","agreement":"both","source":"agents"}},
    "address":     {{"value":"","agreement":"both","source":"agents"}}
  }},
  "barcode_summary": {{
    "match_count": 0,
    "mismatch_count": 0,
    "mismatches": {{}}
  }},
  "agent1_errors": [],
  "agent2_errors": [],
  "flags": [],
  "confidence": 0.0,
  "is_authentic": true,
  "verdict": "PASS or WARN or FAIL",
  "verdict_reason": ""
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — Primary document
# ═══════════════════════════════════════════════════════════════════════════════
def run_stage1(front_bytes: bytes, back_bytes: Optional[bytes],
               barcode_raw: Optional[str], media_type: str = "image/jpeg",
               doc_type: str = "unknown") -> dict:

    barcode_fields   = PDF417Parser().parse(barcode_raw).get("fields",{}) if barcode_raw else None
    barcode_present  = barcode_fields is not None
    barcode_expected = doc_type.upper() in ("DL","ID","CDL","UNKNOWN","")

    std      = _standards_text("AZ")
    prompt   = _agent_prompt(std, barcode_fields, back_bytes is not None, doc_type)

    # Build image content for each agent
    images = [_img(front_bytes, media_type)]
    if back_bytes:
        images.append(_img(back_bytes, media_type))

    print("  [S1] Agent 1 → Claude...")
    r1 = _call(_AGENT_SYSTEM, images + [{"type":"text","text":prompt}], "Agent1")

    print("  [S1] Agent 2 → Claude...")
    r2 = _call(_AGENT_SYSTEM, images + [{"type":"text","text":prompt}], "Agent2")

    # Judge sees both outputs + images
    print("  [S1] Judge → Claude...")
    rj = _call(_JUDGE_SYSTEM,
               images + [{"type":"text","text":_judge_prompt(r1, r2, barcode_fields)}],
               "Judge")

    extracted = rj.get("final_extracted_fields", {})
    extracted["portrait_side"] = "left" if extracted.get("portrait_on_left") else "right"
    rules = RulesEngine("AZ").check_document(extracted, barcode_raw)

    raw_conf = min(rj.get("confidence", 0.0), rules.confidence())
    if not barcode_present and barcode_expected and back_bytes:
        final_conf = min(raw_conf, 0.65)
        barcode_note = {"severity":"WARN",
                        "message":"Barcode unreadable — AZ DLs must have PDF417. Manual inspection required."}
    elif not barcode_present and barcode_expected:
        final_conf = min(raw_conf, 0.75)
        barcode_note = {"severity":"INFO","message":"No back image — barcode verification skipped."}
    else:
        final_conf = raw_conf
        barcode_note = None if barcode_present else \
                       {"severity":"INFO","message":"Barcode not applicable for this document type."}

    fails = rules.summary()["fail_count"]
    verdict = "FAIL" if fails>=2 or final_conf<0.40 else \
              "WARN" if fails>=1 or final_conf<0.70 else "PASS"

    return {
        "stage":            1,
        "agent_1":          r1,
        "agent_2":          r2,
        "judge":            rj,
        "rules_report":     rules.summary(),
        "extracted_fields": extracted,
        "field_poll":       rj.get("field_poll_results", {}),
        "barcode_summary":  rj.get("barcode_summary", {}),
        "barcode_present":  barcode_present,
        "barcode_expected": barcode_expected,
        "barcode_note":     barcode_note,
        "flags":            rj.get("flags", []),
        "verdict":          verdict,
        "confidence":       round(final_conf, 2),
        "is_authentic":     verdict != "FAIL",
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — Cross-document
# ═══════════════════════════════════════════════════════════════════════════════
_DOC2_SYSTEM = """You are an expert document reader for identity verification.
Extract all identity fields from the document image. Return ONLY valid JSON."""

_DOC2_PROMPT = """Extract identity fields from this document. Return ONLY this JSON:
{
  "document_type": "utility_bill or bank_statement or loan_doc or lease or other",
  "issuing_org": "",
  "document_date": "MM/DD/YYYY or null",
  "family_name": "",
  "given_name": "",
  "full_name_as_printed": "",
  "address": "",
  "city": "",
  "state": "",
  "zip": "",
  "dob": "MM/DD/YYYY or null",
  "account_last4": "",
  "uncertain_fields": [],
  "confidence": 0.0
}"""

_XMATCH_SYSTEM = """You are the cross-document Judge for identity verification.
Compare the primary ID fields against the second document.
Return ONLY valid JSON."""

def _xmatch_prompt(id_fields: dict, doc2_a: dict, doc2_b: dict, local: dict) -> str:
    return f"""You have both documents in the images above.

PRIMARY ID FIELDS:
{json.dumps(id_fields, indent=2)}

AGENT 1 extraction of second document:
{json.dumps(doc2_a, indent=2)}

AGENT 2 extraction of second document:
{json.dumps(doc2_b, indent=2)}

RULE-BASED match result (deterministic — trust this):
{json.dumps(local, indent=2)}

Compare identity fields. Return ONLY this JSON:
{{
  "cross_match": {{
    "name":    {{"id_value":"","doc2_value":"","match":true,"detail":""}},
    "address": {{"id_value":"","doc2_value":"","match":true,"detail":""}},
    "dob":     {{"id_value":"","doc2_value":"","match":true,"detail":""}}
  }},
  "compliance_flags": [{{"flag":"","severity":"FAIL or WARN"}}],
  "agent_disagreements": [],
  "confidence": 0.0,
  "overall_verdict": "PASS or WARN or FAIL",
  "summary": "",
  "recommended_action": "APPROVE or REVIEW or REJECT"
}}"""


def run_stage2(id_image_bytes: bytes, doc2_image_bytes: bytes,
               id_extracted: dict, media_type: str = "image/jpeg") -> dict:

    doc2_img = [_img(doc2_image_bytes, media_type)]

    print("  [S2] Agent 1 → Claude (doc 2)...")
    ra = _call(_DOC2_SYSTEM, doc2_img + [{"type":"text","text":_DOC2_PROMPT}], "S2-Agent1")

    print("  [S2] Agent 2 → Claude (doc 2)...")
    rb = _call(_DOC2_SYSTEM, doc2_img + [{"type":"text","text":_DOC2_PROMPT}], "S2-Agent2")

    # Poll doc2 agents
    doc2, disagree = {}, []
    for f in ("family_name","given_name","address","city","state","zip","dob",
              "document_type","full_name_as_printed","issuing_org","document_date"):
        v1 = str(ra.get(f,"") or "").strip().upper()
        v2 = str(rb.get(f,"") or "").strip().upper()
        if v1==v2:          doc2[f]=v1
        elif v1 and not v2: doc2[f]=v1
        elif v2 and not v1: doc2[f]=v2
        elif v1 and v2:     doc2[f]=v1; disagree.append(f"{f}: {v1} vs {v2}")
        else:               doc2[f]=""

    local = CrossDocumentMatcher().match(id_extracted, doc2)

    print("  [S2] Judge → Claude (cross-match)...")
    both_images = [_img(id_image_bytes, media_type), _img(doc2_image_bytes, media_type)]
    rj = _call(_XMATCH_SYSTEM,
               both_images + [{"type":"text","text":_xmatch_prompt(id_extracted, ra, rb, local)}],
               "S2-Judge")

    cross_match = rj.get("cross_match", {
        "name":    {"id_value":"","doc2_value":"","match":None,"detail":""},
        "address": {"id_value":"","doc2_value":"","match":None,"detail":""},
        "dob":     {"id_value":"","doc2_value":"","match":None,"detail":""},
    })

    flags = rj.get("compliance_flags", []) + \
            [{"flag":f"Agent disagreement: {d}","severity":"WARN"} for d in disagree]

    return {
        "stage":              2,
        "agent_1":            ra,
        "agent_2":            rb,
        "judge":              rj,
        "doc2_fields":        doc2,
        "cross_match":        cross_match,
        "local_match":        local,
        "compliance_flags":   flags,
        "agent_disagreements":disagree,
        "verdict":            rj.get("overall_verdict", local["verdict"]),
        "confidence":         rj.get("confidence", local["overall_confidence"]),
        "summary":            rj.get("summary", ""),
        "recommended_action": rj.get("recommended_action", "REVIEW"),
    }
