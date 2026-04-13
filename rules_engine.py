"""
rules_engine.py
Pure rule-based checks — no AI, no network calls, zero PII leakage.
Loads AAMVA universal + state-specific standards and validates extracted fields.
"""

import json
import re
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

STANDARDS_DIR = Path(__file__).parent / "standards"

# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    rule_id: str
    severity: str          # FAIL | WARN | INFO
    passed: bool
    description: str
    detail: str = ""

@dataclass
class RulesReport:
    results: list[RuleResult] = field(default_factory=list)

    @property
    def failed(self):   return [r for r in self.results if not r.passed and r.severity == "FAIL"]
    @property
    def warned(self):   return [r for r in self.results if not r.passed and r.severity == "WARN"]
    @property
    def passed_all(self): return len(self.failed) == 0

    def confidence(self) -> float:
        """
        Simple confidence score:
        - Each FAIL deducts 0.15
        - Each WARN deducts 0.05
        - Floor at 0.0
        """
        score = 1.0
        score -= len(self.failed) * 0.15
        score -= len(self.warned) * 0.05
        return max(0.0, round(score, 2))

    def summary(self) -> dict:
        return {
            "passed": self.passed_all,
            "confidence": self.confidence(),
            "fail_count": len(self.failed),
            "warn_count": len(self.warned),
            "failures": [{"rule": r.rule_id, "desc": r.description, "detail": r.detail}
                         for r in self.failed],
            "warnings": [{"rule": r.rule_id, "desc": r.description, "detail": r.detail}
                         for r in self.warned],
        }


# ─── Barcode parser ───────────────────────────────────────────────────────────

class PDF417Parser:
    """
    Parse raw AAMVA PDF417 barcode string into structured fields.
    Format: @[LF][RS][CR]ANSI [IIN][version][jver][nfiles][subfile_type][offset][length]...[fields]
    """

    def parse(self, raw: str) -> dict:
        if not raw or not raw.startswith("@"):
            return {"error": "Not a valid AAMVA barcode (must start with @)"}

        result = {"raw": raw, "fields": {}, "header": {}}

        # Find ANSI header
        ansi_idx = raw.find("ANSI ")
        if ansi_idx == -1:
            return {"error": "ANSI header not found"}

        header_start = ansi_idx + 5
        try:
            result["header"]["iin"] = raw[header_start:header_start+6]
            result["header"]["aamva_version"] = raw[header_start+6:header_start+8]
            result["header"]["jurisdiction_version"] = raw[header_start+8:header_start+10]
            result["header"]["num_entries"] = raw[header_start+10:header_start+12]
        except IndexError:
            return {"error": "Header too short"}

        # Parse all 3-letter element IDs
        # Pattern: [A-Z]{3}followed by data until next element or end
        element_pattern = re.compile(r'([A-Z]{3})([^\r\n]*?)(?=[A-Z]{3}|\r|\n|$)')
        
        # Work on the data portion after the subfile designator
        dl_idx = raw.find("DL")
        id_idx = raw.find("\nID")
        data_start = max(dl_idx, id_idx) if dl_idx > 0 or id_idx > 0 else ansi_idx
        
        # Simple line-by-line parse
        lines = raw.replace('\r', '\n').split('\n')
        for line in lines:
            line = line.strip()
            if len(line) >= 3:
                key = line[:3]
                val = line[3:].strip()
                if re.match(r'^[A-Z]{3}$', key) and val:
                    result["fields"][key] = val

        return result


# ─── Main rules engine ────────────────────────────────────────────────────────

class RulesEngine:
    def __init__(self, state: str = "AZ"):
        self.state = state.upper()
        self.universal = self._load_json("aamva_universal.json")
        self.state_rules = self._load_json(f"states/{self.state}.json")
        self.parser = PDF417Parser()

    def _load_json(self, filename: str) -> dict:
        path = STANDARDS_DIR / filename
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _add(self, report: RulesReport, rule_id: str, severity: str,
             passed: bool, description: str, detail: str = ""):
        report.results.append(RuleResult(rule_id, severity, passed, description, detail))

    # ── Date helpers ──────────────────────────────────────────────────────────

    def _parse_date_us(self, s: str) -> Optional[date]:
        """Parse MM/DD/CCYY"""
        try:
            return datetime.strptime(s.strip(), "%m/%d/%Y").date()
        except:
            return None

    def _parse_date_barcode(self, s: str) -> Optional[date]:
        """Parse MMDDCCYY (barcode format)"""
        try:
            return datetime.strptime(s.strip(), "%m%d%Y").date()
        except:
            return None

    # ── Core checks ───────────────────────────────────────────────────────────

    def check_document(
        self,
        printed_fields: dict,
        barcode_raw: Optional[str] = None,
        dimensions_mm: Optional[dict] = None,
    ) -> RulesReport:
        """
        Run all rule-based checks.

        printed_fields: what was extracted visually from the card face
            {
              "family_name": str, "given_name": str, "dob": str (MM/DD/YYYY),
              "expiry": str (MM/DD/YYYY), "issue_date": str,
              "id_number": str, "sex": str, "eye_color": str,
              "height": str, "address": str,
              "portrait_side": str ("left"|"right"),
              "orientation": str ("horizontal"|"vertical"),
              "zone_I_text": str,
              "document_type": str ("DL"|"ID"|"CDL"),
            }
        barcode_raw: raw decoded PDF417 string
        dimensions_mm: {"width": float, "height": float}
        """
        report = RulesReport()
        today = date.today()

        # ── R001: Portrait on left ────────────────────────────────────────────
        portrait_side = printed_fields.get("portrait_side", "").lower()
        self._add(report, "R001", "FAIL",
                  portrait_side == "left" or portrait_side == "",
                  "Portrait must be on LEFT side (Zone III)",
                  f"Detected: {portrait_side or 'unknown'}")

        # ── R002: Orientation vs age ──────────────────────────────────────────
        dob = self._parse_date_us(printed_fields.get("dob", ""))
        orientation = printed_fields.get("orientation", "").lower()
        if dob and orientation:
            age = (today - dob).days // 365
            if age < 21:
                self._add(report, "R005", "FAIL",
                          orientation == "vertical",
                          "Under-21 DL must be VERTICAL orientation",
                          f"Age: {age}, orientation: {orientation}")
            else:
                self._add(report, "R004", "FAIL",
                          orientation == "horizontal",
                          "Over-21 DL must be HORIZONTAL orientation",
                          f"Age: {age}, orientation: {orientation}")

        # ── R003: Date formats ────────────────────────────────────────────────
        for field_name, rule_id in [("dob", "R006"), ("expiry", "R006b"), ("issue_date", "R006c")]:
            val = printed_fields.get(field_name, "")
            if val:
                parsed = self._parse_date_us(val)
                self._add(report, rule_id, "FAIL",
                          parsed is not None,
                          f"{field_name} must be MM/DD/YYYY format",
                          f"Value: '{val}'")

        # ── R004: Expiry checks ───────────────────────────────────────────────
        expiry = self._parse_date_us(printed_fields.get("expiry", ""))
        if expiry:
            days_left = (expiry - today).days
            self._add(report, "R007", "FAIL",
                      days_left >= 0,
                      "Document must not be expired",
                      f"Expired {abs(days_left)} days ago" if days_left < 0 else f"Valid for {days_left} days")
            if days_left >= 0:
                self._add(report, "R008", "WARN",
                          days_left > 30,
                          "Document expires within 30 days — notary should flag",
                          f"Expires in {days_left} days")

        # ── R005: Sex field ───────────────────────────────────────────────────
        sex = printed_fields.get("sex", "").upper()
        if sex:
            self._add(report, "R009", "FAIL",
                      sex in ["M", "F", "X"],
                      "Sex field must be M, F, or X",
                      f"Value: '{sex}'")

        # ── R006: Eye color ───────────────────────────────────────────────────
        eye = printed_fields.get("eye_color", "").upper()
        valid_eyes = ["BLU", "BRO", "BLK", "HAZ", "GRN", "GRY", "PNK", "MAR", "DIC"]
        if eye:
            self._add(report, "R010", "FAIL",
                      eye in valid_eyes,
                      "Eye color must use AAMVA 3-letter code",
                      f"Value: '{eye}', valid: {valid_eyes}")

        # ── R007: Physical dimensions ─────────────────────────────────────────
        if dimensions_mm:
            w = dimensions_mm.get("width", 0)
            h = dimensions_mm.get("height", 0)
            # ISO ID-1: 85.6 × 53.98mm, tolerance ±0.12mm
            w_ok = abs(w - 85.6) <= 2.0   # generous tolerance for scanned images
            h_ok = abs(h - 53.98) <= 2.0
            self._add(report, "R_DIM_001", "FAIL",
                      w_ok and h_ok,
                      "Card dimensions must match ISO ID-1 (85.6 × 53.98mm)",
                      f"Detected: {w:.1f}mm × {h:.1f}mm, expected 85.6 × 53.98mm")

        # ── R008: Zone I text ─────────────────────────────────────────────────
        zone_i = printed_fields.get("zone_I_text", "").upper()
        valid_labels = ["DRIVER LICENSE", "DRIVER'S LICENSE", "IDENTIFICATION CARD",
                        "COMMERCIAL DRIVER LICENSE", "CDL"]
        if zone_i:
            found = any(lbl in zone_i for lbl in valid_labels)
            self._add(report, "R020", "INFO",
                      found,
                      "Zone I must contain standard document type label",
                      f"Value: '{zone_i}'")

        # ── R009: AZ-specific DL number format ───────────────────────────────
        if self.state == "AZ":
            id_number = printed_fields.get("id_number", "")
            if id_number:
                az_format = bool(re.match(r'^[A-Z][0-9]{8}$', id_number.upper()))
                self._add(report, "R_AZ_004", "WARN",
                          az_format,
                          "Arizona DL number format: 1 letter + 8 digits (e.g. D12345678)",
                          f"Value: '{id_number}'")

        # ── Barcode checks ────────────────────────────────────────────────────
        if barcode_raw:
            self._check_barcode(report, barcode_raw, printed_fields)

        return report

    def _check_barcode(self, report: RulesReport, raw: str, printed: dict):
        """All barcode-specific rule checks."""

        # R011: starts with @
        self._add(report, "R003", "FAIL",
                  raw.startswith("@"),
                  "Barcode must start with @ (AAMVA compliance indicator)",
                  f"Starts with: '{raw[:3]}'")

        parsed = self.parser.parse(raw)
        if "error" in parsed:
            self._add(report, "R_BAR_ERR", "FAIL",
                      False, "Barcode could not be parsed", parsed["error"])
            return

        fields = parsed.get("fields", {})
        header = parsed.get("header", {})

        # R012: mandatory fields present
        mandatory = ["DCS", "DAC", "DBB", "DBA", "DAQ", "DCF", "DCG"]
        for elem_id in mandatory:
            self._add(report, f"R_BAR_{elem_id}", "FAIL",
                      elem_id in fields,
                      f"Mandatory barcode field {elem_id} ({self._elem_name(elem_id)}) must be present",
                      f"Fields found: {list(fields.keys())[:10]}")

        # R013: IIN = AZ (636004)
        iin = header.get("iin", "")
        if self.state == "AZ":
            self._add(report, "R_AZ_005", "FAIL",
                      iin == "636004",
                      "Arizona document IIN must be 636004",
                      f"Found IIN: '{iin}'")

        # R014: Country = USA
        country = fields.get("DCG", "")
        if country:
            self._add(report, "R_BAR_COUNTRY", "FAIL",
                      country == "USA",
                      "Barcode country (DCG) must be USA",
                      f"Value: '{country}'")

        # R015: Barcode name matches printed name
        bc_last = fields.get("DCS", "").upper().strip()
        bc_first = fields.get("DAC", "").upper().strip()
        pr_last = printed.get("family_name", "").upper().strip()
        pr_first = printed.get("given_name", "").upper().strip()

        if bc_last and pr_last:
            self._add(report, "R015_LAST", "FAIL",
                      self._fuzzy_name_match(bc_last, pr_last),
                      "Barcode last name must match printed last name",
                      f"Barcode: '{bc_last}', Printed: '{pr_last}'")

        if bc_first and pr_first:
            self._add(report, "R015_FIRST", "FAIL",
                      self._fuzzy_name_match(bc_first, pr_first),
                      "Barcode first name must match printed first name",
                      f"Barcode: '{bc_first}', Printed: '{pr_first}'")

        # R016: Barcode DOB matches printed DOB
        bc_dob_raw = fields.get("DBB", "")
        if bc_dob_raw:
            bc_dob = self._parse_date_barcode(bc_dob_raw)
            pr_dob = self._parse_date_us(printed.get("dob", ""))
            if bc_dob and pr_dob:
                self._add(report, "R016", "FAIL",
                          bc_dob == pr_dob,
                          "Barcode DOB must match printed DOB",
                          f"Barcode: {bc_dob}, Printed: {pr_dob}")

        # R017: Barcode ID number matches printed
        bc_id = fields.get("DAQ", "").strip()
        pr_id = printed.get("id_number", "").strip()
        if bc_id and pr_id:
            self._add(report, "R017", "FAIL",
                      bc_id.upper() == pr_id.upper(),
                      "Barcode ID number (DAQ) must match printed ID number",
                      f"Barcode: '{bc_id}', Printed: '{pr_id}'")

        # R018: Barcode expiry matches printed
        bc_exp_raw = fields.get("DBA", "")
        if bc_exp_raw:
            bc_exp = self._parse_date_barcode(bc_exp_raw)
            pr_exp = self._parse_date_us(printed.get("expiry", ""))
            if bc_exp and pr_exp:
                self._add(report, "R018", "FAIL",
                          bc_exp == pr_exp,
                          "Barcode expiry must match printed expiry",
                          f"Barcode: {bc_exp}, Printed: {pr_exp}")

        # R019: DDA compliance flag
        dda = fields.get("DDA", "")
        if dda:
            self._add(report, "R_DDA", "WARN",
                      dda == "F",
                      "DDA=N means card is NOT REAL ID / DHS compliant",
                      f"DDA value: '{dda}' (F=compliant, N=non-compliant)")

        # R020: DDD limited duration
        ddd = fields.get("DDD", "")
        if ddd == "1":
            self._add(report, "R_DDD", "WARN",
                      False,
                      "DDD=1: This is a LIMITED DURATION document (temporary lawful status)",
                      "Notary must verify additional documentation")

    def _elem_name(self, code: str) -> str:
        mapping = {
            "DCS": "Family name", "DAC": "First name", "DAD": "Middle name",
            "DBB": "Date of birth", "DBA": "Expiry", "DBD": "Issue date",
            "DAQ": "ID number", "DCF": "Doc discriminator", "DCG": "Country",
            "DBC": "Sex", "DAY": "Eye color", "DAU": "Height",
            "DAG": "Street", "DAI": "City", "DAJ": "State", "DAK": "ZIP"
        }
        return mapping.get(code, code)

    def _fuzzy_name_match(self, a: str, b: str) -> bool:
        """Allow truncation: barcode may truncate long names."""
        a, b = a.upper().strip(), b.upper().strip()
        if a == b:
            return True
        # Truncation: one starts with the other
        return a.startswith(b[:20]) or b.startswith(a[:20])


# ─── Cross-document matcher ───────────────────────────────────────────────────

class CrossDocumentMatcher:
    """
    Stage 3: Compare fields from document 1 (DL/ID) vs document 2 (utility bill, etc.)
    No AI, pure field comparison.
    """

    def match(self, doc1_fields: dict, doc2_fields: dict) -> dict:
        results = {}
        today = date.today()

        checks = [
            ("name_last",    "family_name",  "family_name"),
            ("name_first",   "given_name",   "given_name"),
            ("dob",          "dob",          "dob"),
            ("address_street","address",     "address"),
            ("address_city", "city",         "city"),
            ("address_state","state",        "state"),
        ]

        for check_id, key1, key2 in checks:
            v1 = (doc1_fields.get(key1) or "").upper().strip()
            v2 = (doc2_fields.get(key2) or "").upper().strip()
            if v1 and v2:
                match = self._loose_match(v1, v2)
                results[check_id] = {
                    "doc1_value": doc1_fields.get(key1, ""),
                    "doc2_value": doc2_fields.get(key2, ""),
                    "match": match,
                    "confidence": 0.95 if v1 == v2 else (0.75 if match else 0.2),
                    "flag": not match
                }

        flags = [k for k, v in results.items() if v.get("flag")]
        overall_confidence = (
            sum(v["confidence"] for v in results.values()) / len(results)
            if results else 0.0
        )

        return {
            "field_results": results,
            "flags": flags,
            "overall_confidence": round(overall_confidence, 2),
            "verdict": "PASS" if not flags else ("WARN" if len(flags) <= 1 else "FAIL"),
        }

    def _loose_match(self, a: str, b: str) -> bool:
        if a == b:
            return True
        # Remove common punctuation and retry exact match
        clean_a = re.sub(r'[^A-Z0-9 ]', '', a).strip()
        clean_b = re.sub(r'[^A-Z0-9 ]', '', b).strip()
        if clean_a == clean_b:
            return True
        # For addresses only: allow one containing the other (abbreviations like ST vs STREET)
        # But NOT for names — different name = different person
        return False


if __name__ == "__main__":
    # Quick smoke test
    engine = RulesEngine("AZ")
    
    test_fields = {
        "family_name": "SMITH",
        "given_name": "JOHN",
        "dob": "06/15/1990",
        "expiry": "06/15/2028",
        "issue_date": "06/15/2024",
        "id_number": "D12345678",
        "sex": "M",
        "eye_color": "BRO",
        "height": "5'-10\"",
        "portrait_side": "left",
        "orientation": "horizontal",
        "zone_I_text": "ARIZONA DRIVER LICENSE",
        "document_type": "DL",
    }

    test_barcode = "@\n\x1e\rANSI 636004100002DL00410278ZA03190008DLDAQ D12345678\nDCSSMITH\nDACJOHN\nDAD\nDBD06152024\nDBB06151990\nDBA06152028\nDBC1\nDAU070 in\nDAYBRO\nDAG123 MAIN ST\nDAITEMPE\nDAJAZ\nDAK852811234 \nDCF24242447474\nDCGUSA\nDDAF\n"

    report = engine.check_document(test_fields, test_barcode, {"width": 85.6, "height": 53.98})
    import json
    print(json.dumps(report.summary(), indent=2))
    print(f"\nConfidence: {report.confidence()}")