"""
main.py  —  FastAPI backend for Notary ID Verify
"""

# Load .env file FIRST before any other imports that read env vars
from pathlib import Path
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            import os
            os.environ.setdefault(key.strip(), val.strip())
    print(f"[.env] Loaded from {_env_file}")
else:
    print(f"[.env] No .env file found at {_env_file}")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Optional, List
import json
import base64
from pathlib import Path

from agents import run_stage1, run_stage2

def detect_media_type(data: bytes) -> str:
    """Detect image media type from magic bytes — never trust browser content-type."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    if data[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return 'image/jpeg'  # fallback

from rules_engine import PDF417Parser, RulesEngine

app = FastAPI(title="TrustSeal", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

# Serve static files at /static/*
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ─── Root redirect ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect root to the UI."""
    return HTMLResponse(
        '<html><head><meta http-equiv="refresh" content="0;url=/static/index.html"></head></html>'
    )


# ─── Barcode scan (local, no AI) ───────────────────────────────────────────────

def _detect_image_issues(img_pil) -> dict:
    """
    Analyse image quality BEFORE attempting barcode decode.
    Returns dict with issues and a suggested fix message.
    """
    import numpy as np
    arr = np.array(img_pil.convert('RGB'))
    h, w = arr.shape[:2]

    issues = []

    # 1. Overall brightness
    mean = arr.mean()
    if mean > 210:
        issues.append({"type": "overexposed", "severity": "critical",
                        "msg": "Image is overexposed — too much light or flash. "
                               "Turn off flash and avoid direct light on the card."})
    elif mean < 40:
        issues.append({"type": "underexposed", "severity": "critical",
                        "msg": "Image is too dark. Move to better lighting."})

    # 2. Glare hotspot — check for large bright patch
    gray = np.array(img_pil.convert('L'))
    overlit_pixels = (gray > 240).sum()
    overlit_pct = overlit_pixels / gray.size * 100
    if overlit_pct > 8:
        issues.append({"type": "glare", "severity": "critical",
                        "msg": f"Strong glare detected ({overlit_pct:.0f}% of image is blown out). "
                               "Tilt the card slightly or move away from light sources."})
    elif overlit_pct > 3:
        issues.append({"type": "glare", "severity": "warn",
                        "msg": f"Mild glare ({overlit_pct:.0f}% overlit). "
                               "Try tilting the card slightly to reduce reflection."})

    # 3. Low contrast (card face nearly uniform — focus/blur issue)
    std = gray.std()
    if std < 15:
        issues.append({"type": "low_contrast", "severity": "critical",
                        "msg": "Image has very low contrast — card may be out of focus or "
                               "the background matches the card. Place on a dark surface."})

    # 4. Card not filling frame (too much background)
    # Rough check: if the centre strip is high variance but edges are low
    centre = gray[h//3:2*h//3, w//4:3*w//4]
    edge_top = gray[:h//8, :]
    if edge_top.std() < 8 and centre.std() > 20:
        pass  # card centred, fine
    
    return {
        "has_issues": len(issues) > 0,
        "critical": any(i["severity"] == "critical" for i in issues),
        "issues": issues,
        "stats": {
            "mean_brightness": round(float(mean), 1),
            "contrast_std": round(float(std), 1),
            "overlit_pct": round(overlit_pct, 1),
        }
    }


@app.post("/scan-barcode")
async def scan_barcode(file: UploadFile = File(...)):
    """
    Decode PDF417 barcode from an ID image using zxing-cpp.
    Runs ENTIRELY LOCALLY — no AI, no PII sent anywhere.
    Detects glare/lighting issues and asks user to re-upload if needed.
    """
    image_bytes = await file.read()

    try:
        import zxingcpp
        import numpy as np
        from PIL import Image
        import io
        import cv2

        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')

        # ── Step 1: Check image quality BEFORE attempting decode ──────────────
        quality = _detect_image_issues(img)
        if quality["critical"]:
            issue_msgs = [i["msg"] for i in quality["issues"]]
            return {
                "success": False,
                "reupload_required": True,
                "error": "Image quality too poor to scan barcode.",
                "user_message": issue_msgs[0],   # Show the most important issue
                "all_issues": issue_msgs,
                "quality_stats": quality["stats"],
                "tip": "For best results: place the card flat on a dark surface, "
                       "turn off flash, and ensure even lighting with no reflections."
            }

        # ── Step 2: Attempt decode with progressive preprocessing ─────────────
        raw = None

        # Pass 1: raw orientations
        for angle in [0, 90, 180, 270]:
            test = img.rotate(angle, expand=True) if angle else img
            results = zxingcpp.read_barcodes(np.array(test.convert('RGB')))
            for r in results:
                if 'PDF417' in str(r.format):
                    raw = r.text
                    print(f"  [barcode] PDF417 decoded at {angle}° (pass 1)")
                    break
            if raw: break

        # Pass 2: CLAHE contrast normalisation (handles uneven lighting)
        if not raw:
            img_gray = np.array(img.convert('L'))
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(img_gray)
            enhanced_pil = Image.fromarray(enhanced)
            for angle in [0, 90, 180, 270]:
                test = enhanced_pil.rotate(angle, expand=True) if angle else enhanced_pil
                results = zxingcpp.read_barcodes(np.array(test))
                for r in results:
                    if 'PDF417' in str(r.format):
                        raw = r.text
                        print(f"  [barcode] PDF417 decoded at {angle}° (pass 2 CLAHE)")
                        break
                if raw: break

        # Pass 3: 2x upscale + CLAHE (handles low-res scans)
        if not raw:
            w, h = img.size
            big = img.resize((w*2, h*2), Image.LANCZOS)
            big_gray = np.array(big.convert('L'))
            big_enhanced = clahe.apply(big_gray)
            big_pil = Image.fromarray(big_enhanced)
            for angle in [0, 90, 180, 270]:
                test = big_pil.rotate(angle, expand=True) if angle else big_pil
                results = zxingcpp.read_barcodes(np.array(test))
                for r in results:
                    if 'PDF417' in str(r.format):
                        raw = r.text
                        print(f"  [barcode] PDF417 decoded at {angle}° (pass 3 2x+CLAHE)")
                        break
                if raw: break

        # ── Step 3: Handle decode failure ─────────────────────────────────────
        if not raw:
            # Soft glare might not be critical but still blocks decode
            if quality["stats"]["overlit_pct"] > 1.5:
                return {
                    "success": False,
                    "reupload_required": True,
                    "error": "Barcode not readable — glare on card surface.",
                    "user_message": "The barcode is obscured by glare. "
                                    "Try tilting the card 10-15° away from the light source, "
                                    "or move to a different angle.",
                    "quality_stats": quality["stats"],
                    "tip": "Hold the card at a slight angle so light doesn't reflect "
                           "directly into the camera."
                }
            return {
                "success": False,
                "reupload_required": True,
                "error": "PDF417 barcode not detected after all preprocessing attempts.",
                "user_message": "Could not read the barcode. Please ensure:\n"
                                "• The BACK of the ID is photographed\n"
                                "• The barcode is fully visible and not cut off\n"
                                "• The image is in focus\n"
                                "• No fingers or objects are covering the barcode",
                "quality_stats": quality["stats"],
            }

        # ── Step 4: Clean and parse the raw barcode ───────────────────────────
        raw_clean = (raw
            .replace('<LF>', '\n')
            .replace('<CR>', '\r')
            .replace('<RS>', '\x1e')
            .replace('<GS>', '\x1d')
        )

        parser = PDF417Parser()
        parsed = parser.parse(raw_clean)
        fields = parsed.get("fields", {})

        extracted = {
            "family_name": fields.get("DCS", ""),
            "given_name":  fields.get("DAC", ""),
            "dob":         _barcode_date_to_us(fields.get("DBB", "")),
            "expiry":      _barcode_date_to_us(fields.get("DBA", "")),
            "issue_date":  _barcode_date_to_us(fields.get("DBD", "")),
            "id_number":   fields.get("DAQ", ""),
            "address":     fields.get("DAG", ""),
            "apt":         fields.get("DAH", ""),
            "city":        fields.get("DAI", ""),
            "state":       fields.get("DAJ", ""),
            "zip":         fields.get("DAK", ""),
            "sex":         "M" if fields.get("DBC") == "1" else "F" if fields.get("DBC") == "2" else "",
            "eye_color":   fields.get("DAY", ""),
            "height":      fields.get("DAU", ""),
            "vehicle_class": fields.get("DCA", ""),
            "iin":         parsed.get("header", {}).get("iin", ""),
            "aamva_version": parsed.get("header", {}).get("aamva_version", ""),
        }

        field_count = sum(1 for v in extracted.values() if v)
        print(f"  [barcode] Parsed {field_count} fields. Name: {extracted['given_name']} {extracted['family_name']}")

        return {
            "success": True,
            "raw_barcode": raw_clean,
            "parsed_fields": fields,
            "header": parsed.get("header", {}),
            "extracted": extracted,
            "field_count": field_count,
            "quality_stats": quality["stats"],
        }

    except Exception as e:
        import traceback
        print(f"  [barcode ERR] {traceback.format_exc()}")
        return {"success": False, "reupload_required": False, "error": str(e)}


def _barcode_date_to_us(s: str) -> str:
    """Convert MMDDCCYY to MM/DD/YYYY"""
    if len(s) == 8 and s.isdigit():
        return f"{s[:2]}/{s[2:4]}/{s[4:]}"
    return s


# ─── Stage 1 — Primary document verification ──────────────────────────────────
# Both agents do everything: state ID, standard check, extraction, barcode compare
# Judge polls field by field and produces final document JSON

@app.post("/verify/stage1")
async def verify_stage1(
    front: UploadFile = File(...),
    back:  Optional[UploadFile] = File(None),
    barcode_raw: Optional[str] = Form(None),
    doc_type: Optional[str] = Form("unknown"),
):
    """
    Complete primary document verification.
    Agent 1 + Agent 2: same prompt, independent full analysis (state ID → check → extract → barcode)
    Judge: polls field by field, resolves conflicts, produces final JSON
    """
    front_bytes = await front.read()
    back_bytes  = await back.read() if back else None
    media_type  = detect_media_type(front_bytes)

    result = run_stage1(front_bytes, back_bytes, barcode_raw, media_type, doc_type or "unknown")
    return result


# ─── Stage 2 — Cross-document verification ────────────────────────────────────

@app.post("/verify/stage2")
async def verify_stage2(
    id_file:   UploadFile = File(...),
    doc2_file: UploadFile = File(...),
    id_extracted: str = Form(...),
):
    """Cross-document match — same two-agent + judge pattern on second document."""
    id_bytes   = await id_file.read()
    doc2_bytes = await doc2_file.read()
    media_type = detect_media_type(id_bytes)

    try:
        extracted = json.loads(id_extracted)
    except json.JSONDecodeError:
        raise HTTPException(400, "id_extracted must be valid JSON")

    result = run_stage2(id_bytes, doc2_bytes, extracted, media_type)
    return result


# ─── Legacy stage3 alias (maps to run_stage2) ────────────────────────────────

@app.post("/verify/stage3")
async def verify_stage3(
    id_file: UploadFile = File(...),
    doc2_file: UploadFile = File(...),
    id_extracted: str = Form(...),
):
    """Cross-document match — alias for /verify/stage2."""
    id_bytes = await id_file.read()
    doc2_bytes = await doc2_file.read()
    media_type = detect_media_type(id_bytes)
    try:
        extracted = json.loads(id_extracted)
    except json.JSONDecodeError:
        raise HTTPException(400, "id_extracted must be valid JSON")
    result = run_stage2(id_bytes, doc2_bytes, extracted, media_type)
    return result


# ─── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


# ─── Debug: test stage1 with a dummy image to check API keys ───────────────────

@app.get("/debug/keys")
def debug_keys():
    """Check which env vars are set (values hidden)."""
    import os
    return {
        "ANTHROPIC_API_KEY":  "SET" if os.environ.get("ANTHROPIC_API_KEY") else "MISSING",
        "OPENAI_API_KEY_1":   "SET" if os.environ.get("OPENAI_API_KEY_1")  else "MISSING",
        "OPENAI_API_KEY_2":   "SET" if os.environ.get("OPENAI_API_KEY_2")  else "MISSING",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)