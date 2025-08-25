# bloodfood.py — Hybrid BioAge + Per-Marker ML food recommender
# - Parses labs (PDF/CSV) → computes PhenoAge + per-marker severity
# - Loads per-food LightGBM models (joblib) + scaler → predicts marker deltas per food
# - Blends BioAge score with per-marker impact + dietary preferences

# ---- imports (keep yours) ----
import os, re, io, json, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Paths (MUST be defined before loaders) ----------------
# ---------------- Paths ----------------
REPO_ROOT = Path(__file__).resolve().parent
env_root  = os.environ.get("BIOAGE_ROOT")
root      = Path(env_root) if env_root else REPO_ROOT

# ADD THESE 3 LINES ↓
MODELS_DIR   = root / "models"
PERFOOD_DIR  = MODELS_DIR / "PerFood"
PERFOOD_ALT  = MODELS_DIR / "Perfood"


PROC   = root / "processed"
ASSETS = root / "app_assets"
CORE   = root / "models" / "RewardModel" / "core_scoring"

FND_PARQUET    = PROC / "FNDDS_MASTER_PER100G.parquet"
CAT_PARQUET    = ASSETS / "food_catalog.parquet"
LAB_SCHEMA_JS  = ASSETS / "lab_schema.json"
TEMPLATE_CSV   = ASSETS / "labs_upload_template.csv"
CONSENSUS_CSV  = CORE / "consensus_food_scores.csv"
GUARDRAILS_CSV = CORE / "core_food_scores_guardrails.csv"
ATTR_CSV       = CORE / "core_food_attribution_top50_compact.csv"  # optional

# Per-marker ML bundle (case-insensitive fallback for mac→linux)
PERFOOD_DIR = root / "models" / "PerFood"   # canonical
PERFOOD_ALT = root / "models" / "Perfood"   # common Mac casing

# File names expected inside the folder
PERFOOD_META   = "meta.json"
PERFOOD_SCALER = "X_scaler.joblib"

# ---------------- Optional dependency: pdfplumber ----------------
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

# ---------------- Small helpers ----------------
def robust_z(x: pd.Series) -> pd.Series:
    """Robust z-score using median & IQR; falls back to std if needed."""
    x = pd.to_numeric(x, errors="coerce").astype(float)
    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    scale = iqr / 1.349 if iqr and np.isfinite(iqr) and iqr > 0 else np.nanstd(x)
    if not (scale and np.isfinite(scale) and scale > 0):
        scale = 1.0
    return (x - med) / scale

def clip01(v): return float(np.clip(v, 0.0, 1.0))

# ---------------- PhenoAge (Levine) ----------------
def phenoage_from_row(row: pd.Series):
    """
    Uses official coefficients + Gompertz mapping.
    Inputs (clinical units): albumin g/dL, creat mg/dL, glucose mg/dL, CRP mg/L,
    lymphocyte %, MCV fL, RDW %, ALP U/L, WBC 10^3/µL, age years
    """
    need = ["age_years","albumin","creatinine","glucose","crp_mgL",
            "lymphocyte_pct","mcv","rdw","alk_phosphatase","wbc"]
    if any(pd.isna(row.get(k)) for k in need):
        return (None, None)

    albumin_gL   = float(row["albumin"]) * 10.0
    creat_umol   = float(row["creatinine"]) * 88.4
    glucose_mmol = float(row["glucose"]) / 18.0
    lncrp        = np.log(max(float(row["crp_mgL"]), 0.0) + 1e-6)
    lymph        = float(row["lymphocyte_pct"])
    mcv          = float(row["mcv"])
    rdw          = float(row["rdw"])
    alp          = float(row["alk_phosphatase"])
    wbc          = float(row["wbc"])
    age          = float(row["age_years"])

    xb = (
        -19.90667
        + (-0.03359355 * albumin_gL)
        + ( 0.009506491 * creat_umol)
        + ( 0.1953192  * glucose_mmol)
        + ( 0.09536762 * lncrp)
        + (-0.01199984 * lymph)
        + ( 0.02676401 * mcv)
        + ( 0.3306156  * rdw)
        + ( 0.001868778* alp)
        + ( 0.05542406 * wbc)
        + ( 0.08035356 * age)
    )
    m = 1.0 - np.exp( (-1.51714 * np.exp(xb)) / 0.007692696 )
    m = float(np.clip(m, 1e-15, 1 - 1e-15))
    pheno = (np.log(-0.0055305 * np.log(1.0 - m)) / 0.09165) + 141.50225
    accel = pheno - age
    return float(np.clip(pheno, 0.0, 140.0)), float(np.clip(accel, -60.0, 90.0))

# ---------------- Loaders ----------------
# ---------------- Loaders ----------------
# ---------------- Loaders ----------------
@st.cache_data(show_spinner=False)
def load_lab_schema():
    if LAB_SCHEMA_JS.exists():
        with open(LAB_SCHEMA_JS, "r") as f:
            return json.load(f)
    # minimal fallback schema
    return {
        "age_years":{"unit":"years","aliases":["age","age_yrs"]},
        "albumin":{"unit":"g/dL","aliases":["LBXSAL","albumin"]},
        "creatinine":{"unit":"mg/dL","aliases":["LBXSCR","creatinine","creat"]},
        "glucose":{"unit":"mg/dL","aliases":["LBXSGL","glucose"]},
        "crp_mgL":{"unit":"mg/L","aliases":["CRP","hsCRP","hs-crp","LBXCRP"]},
        "lymphocyte_pct":{"unit":"%","aliases":["lymphs","lymphocyte %","lymphocytes %"]},
        "mcv":{"unit":"fL","aliases":["LBXMCVSI","mcv"]},
        "rdw":{"unit":"%","aliases":["LBXRDW","rdw"]},
        "alk_phosphatase":{"unit":"U/L","aliases":["LBXSAPSI","alk phos","alkaline phosphatase"]},
        "wbc":{"unit":"10^3/µL","aliases":["LBXWBCSI","wbc","white blood cells"]},
    }

@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    cat = pd.read_parquet(CAT_PARQUET)
    cat["FoodCode"] = pd.to_numeric(cat["FoodCode"], errors="coerce").astype("Int64")
    if "tags" not in cat.columns:
        cat["tags"] = np.nan
    return cat

@st.cache_data(show_spinner=False)
def load_fnd_features() -> pd.DataFrame | None:
    """Return FoodCode + NUTR_* (prefer per-100kcal parquet)."""
    if FND_PARQUET.exists():
        fnd = pd.read_parquet(FND_PARQUET)
        fnd["FoodCode"] = pd.to_numeric(fnd["FoodCode"], errors="coerce").astype("Int64")
        nutr_cols = [c for c in fnd.columns if str(c).startswith("NUTR_")]
        return fnd[["FoodCode"] + nutr_cols].copy() if nutr_cols else None
    # fallback: if catalog already contains NUTR_* columns
    if CAT_PARQUET.exists():
        c = pd.read_parquet(CAT_PARQUET)
        nutr_cols = [c for c in c.columns if str(c).startswith("NUTR_")]
        if nutr_cols:
            out = c[["FoodCode"] + nutr_cols].copy()
            out["FoodCode"] = pd.to_numeric(out["FoodCode"], errors="coerce").astype("Int64")
            return out
    return None

schema   = load_lab_schema()
catalog  = load_catalog() if CAT_PARQUET.exists() else None
fnd_nutr = load_fnd_features()

# ---------------- Dedup + categories used in UI ----------------
_STOPWORDS = re.compile(
    r"\b(ns as to form|nsf|nfs|assume.*?|fat not added in cooking|no added fat|"
    r"from (?:fresh|frozen|canned)|fresh|frozen|canned|raw|cooked|reconstituted|"
    r"for use on a sandwich|reduced sodium|low(?:fat| sodium)|diet)\b", re.I
)
def _normalize_desc(desc: str) -> str:
    s = str(desc or "").lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = _STOPWORDS.sub(" ", s)
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("water cress", "watercress").replace("beet green", "beet greens").replace("turnip green", "turnip greens")
    return s

_VEG_RE   = re.compile(r"\b(kale|chard|lettuce|greens?|watercress|parsley|basil|cilantro|spinach|broccoli|cabbage|cauliflower|asparagus|zucchini|squash|okra|tomato|mushroom|onion|pepper|beet|cucumber|pickle|radish|artichoke|brussels|celery|collards|mustard greens|turnip greens)\b", re.I)
_FRUIT_RE = re.compile(r"\b(apple|banana|orange|berry|berries|strawberry|blueberry|raspberry|blackberry|grape|pear|peach|plum|cherry|pineapple|mango|papaya|melon|watermelon|cantaloupe|honeydew|kiwi|lemon|lime|grapefruit|pomegranate|date|fig|raisin|prune|avocado)\b", re.I)
_LEG_RE   = re.compile(r"\b(bean|lentil|chickpea|pea|soy|tofu|tempeh|edamame|peanut|hummus|bread|rice|pasta|noodle|oat|oatmeal|barley|quinoa|corn|tortilla|wheat|bran|cereal|bulgur|couscous|polenta)\b", re.I)
_PROT_RE  = re.compile(r"\b(beef|steak|veal|lamb|pork|bacon|sausage|ham|chicken|turkey|duck|"
                       r"fish|seafood|salmon|tuna|sardine|sardines|anchovy|anchovies|mackerel|herring|trout|cod|halibut|tilapia|"
                       r"shrimp|prawn|oyster|clam|scallop|crab|lobster|egg|eggs|cheese|yogurt|kefir|milk|cottage cheese)\b", re.I)
_FAT_RE   = re.compile(r"\b(oil|olive oil|canola|avocado oil|sunflower oil|safflower|sesame oil|peanut oil|coconut oil|"
                       r"butter|ghee|margarine|shortening|lard|mayonnaise|mayo|aioli|tahini|"
                       r"nut butter|peanut butter|almond butter|cashew butter|seed butter|tallow|"
                       r"walnut|walnuts|almond|almonds|pecan|pecans|cashew|cashews|pistachio|pistachios|hazelnut|hazelnuts|"
                       r"macadamia|sunflower seed|sunflower seeds|pumpkin seed|pumpkin seeds|flaxseed|chia|sesame seed|sesame seeds)\b", re.I)
_BEV_RE   = re.compile(r"\b(almond milk|soy milk|oat milk|rice milk|coconut milk|hemp milk|cashew milk)\b", re.I)

def coarse_category(desc: str, tags: str) -> str:
    t = (tags or "")
    s = str(desc or "").lower()
    if "cereal_fortified" in t: return "legume_grains"
    if "oil_fat_sauce" in t:    return "fats"
    if "leafy_green" in t or "pickled_veg" in t or "seaweed_algae" in t: return "vegetables"
    if _PROT_RE.search(s) and not _BEV_RE.search(s): return "protein"
    if _FAT_RE.search(s):  return "fats"
    if _VEG_RE.search(s):  return "vegetables"
    if _FRUIT_RE.search(s):return "fruit"
    if _LEG_RE.search(s):  return "legume_grains"
    return "other"

def dedup_rank(df: pd.DataFrame, include_tags: str = "", exclude_tags: str = "") -> pd.DataFrame:
    R = df.copy()
    if include_tags.strip():
        inc = [t.strip() for t in include_tags.split(",") if t.strip()]
        R = R[R["tags"].fillna("").apply(lambda s: any(t in s for t in inc))]
    if exclude_tags.strip():
        exc = [t.strip() for t in exclude_tags.split(",") if t.strip()]
        R = R[~R["tags"].fillna("").apply(lambda s: any(t in s for t in exc))]
    R["dedup_key"] = R["Desc"].map(_normalize_desc)
    R = R.sort_values(["score", "kcal_per_100g"], ascending=[True, True]).drop_duplicates("dedup_key", keep="first")
    return R.reset_index(drop=True)

# ---------------- PDF → labs → sanitize → PhenoAge ----------------
# ---------------- Robust PDF parser + sanitizer ----------------
def parse_pdf_labs(file_like) -> dict:
    """
    Returns a dict of raw extracted labs. Tries tables first, then text.
    Handles ranges, units (mg/dL vs mg/L), and lymphocytes (# and %).
    Keys returned when found (raw, before sanitize_labs):
      albumin (g/dL)
      creatinine (mg/dL)
      fasting_glucose or glucose (mg/dL)
      crp (mg/L or mg/dL handled later)
      wbc (10^3/µL or cells/µL handled later)
      lymphs (percent) or lymphs_abs (cells/µL)
      mcv (fL)
      rdw (%)
      alp (U/L)
    """
    if not HAS_PDFPLUMBER:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")

    # ---- label synonyms (longest-first wins) ----
    synonyms = [
        # Albumin
        ("serum albumin", "albumin"), ("albumin, serum", "albumin"),
        ("albumin (serum)", "albumin"), ("alb", "albumin"), ("albumin", "albumin"),

        # Alkaline phosphatase (many ways)
        ("alkaline phosphatase (total)", "alp"), ("alkaline phosphatase, total", "alp"),
        ("alkaline phosphatase (alp)", "alp"), ("alkaline phosphatase (alk phos)", "alp"),
        ("alk phosphatase", "alp"), ("alk. phosphatase", "alp"),
        ("alk phos", "alp"), ("alk-phos", "alp"),
        ("alkaline phosphatase", "alp"), ("alkaline phosphate", "alp"), ("alp", "alp"),

        # CRP
        ("c-reactive protein, cardiac", "crp"), ("c-reactive protein (cardiac)", "crp"),
        ("c reactive protein, cardiac", "crp"), ("crp, cardiac", "crp"),
        ("high sensitivity crp", "crp"), ("hs-crp", "crp"), ("hscrp", "crp"),
        ("c-reactive protein", "crp"), ("c reactive protein", "crp"), ("crp", "crp"),

        # Glucose
        ("glucose, fasting", "fasting_glucose"), ("glucose (fasting)", "fasting_glucose"),
        ("fasting glucose", "fasting_glucose"), ("glucose fasting", "fasting_glucose"),
        ("glucose", "fasting_glucose"),

        # WBC
        ("white blood cell count", "wbc"), ("white blood cells", "wbc"),
        ("white blood cell", "wbc"), ("wbc", "wbc"),

        # Lymphocytes
        ("lymphocytes absolute", "lymphs_abs"), ("absolute lymphocytes", "lymphs_abs"),
        ("abs lymphocytes", "lymphs_abs"), ("lymphs #", "lymphs_abs"),
        ("lymphocyte percent", "lymphs_pct"), ("lymphocytes percent", "lymphs_pct"),
        ("lymphocyte %", "lymphs_pct"), ("lymphocytes %", "lymphs_pct"),
        ("lymphocytes", "lymphs_pct"), ("lymphs", "lymphs_pct"),

        # RBC indices
        ("mean corpuscular volume", "mcv"), ("mcv", "mcv"),
        ("red cell distribution width", "rdw"), ("rdw", "rdw"),

        # Extras
        ("creatinine", "creatinine"), ("creat", "creatinine"),
        ("blood urea nitrogen", "bun"), ("bun", "bun"),
    ]

    # ---- regex helpers ----
    range_pat   = re.compile(r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\b")
    number_pat  = re.compile(r"(-?\d+(?:\.\d+)?)")
    percent_pat = re.compile(r"%")

    def match_key(lbl: str) -> str | None:
        s = (lbl or "").strip().lower()
        for name, key in sorted(synonyms, key=lambda kv: -len(kv[0])):  # longest first
            if name in s:
                return key
        return None

    def first_numeric(txt: str, prefer_percent=False):
        """Return (value, score) if numeric present and not a range; prefer '%' when asked."""
        if not txt:
            return None, None
        if range_pat.search(txt):
            return None, None
        m = number_pat.search(txt)
        if not m:
            return None, None
        val = float(m.group(1))
        has_pct = bool(percent_pat.search(txt))
        score = (1 if (prefer_percent and has_pct) else 0, -len(txt))
        return val, score

    labs: dict = {}
    aux: dict  = {}      # holds lymphs_abs, lymphs_pct
    evidence = {}

    # ---- Pass 1: tables ----
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            for tbl in (page.extract_tables() or []):
                for r_i, row in enumerate(tbl or []):
                    if not row or all(c is None or str(c).strip() == "" for c in row):
                        continue
                    label = str(row[0] or "")
                    key = match_key(label)
                    if not key:
                        continue

                    prefer_pct = (key == "lymphs_pct")
                    best = None  # (score, value, raw_text)

                    # cells to the right
                    for cell in row[1:]:
                        txt = str(cell or "")
                        val, score = first_numeric(txt, prefer_percent=prefer_pct)
                        if val is None:
                            continue
                        cand = (score, val, txt)
                        if (best is None) or (cand > best):
                            best = cand

                    # sometimes the value is directly below the header
                    if best is None and len(row) > 1 and r_i + 1 < len(tbl):
                        below = str((tbl[r_i + 1] or [""])[1] or "")
                        val, score = first_numeric(below, prefer_percent=prefer_pct)
                        if val is not None:
                            best = (score, val, below)

                    if best is None:
                        continue

                    _, value, rawtxt = best
                    lowtxt = (rawtxt or "").lower()

                    # unit fixes at capture time
                    if key == "crp" and ("mg/dl" in lowtxt) and ("mg/l" not in lowtxt):
                        value *= 10.0
                    if key == "albumin" and "g/l" in lowtxt:
                        value /= 10.0

                    if key == "lymphs_abs":
                        aux["lymphs_abs"] = value
                        evidence["lymphs_abs"] = (value, rawtxt)
                    elif key == "lymphs_pct":
                        aux["lymphs_pct"] = value
                        evidence["lymphs_pct"] = (value, rawtxt)
                    elif key == "wbc":
                        labs["wbc"] = value
                        evidence["wbc"] = (value, rawtxt)
                    else:
                        labs[key] = value
                        evidence[key] = (value, rawtxt)

    # ---- Pass 2: text fallback ----
    file_like.seek(0)
    full_text = ""
    with pdfplumber.open(file_like) as pdf:
        for p in pdf.pages:
            full_text += (p.extract_text() or "").replace("\n", " ") + " "

    def from_text(label, std_key):
        if std_key in labs or std_key in aux:
            return
        pat = re.compile(rf"(?i)\b{re.escape(label)}\b" + r".{0,120}?" + r"(-?\d+(?:\.\d+)?)")
        m = pat.search(full_text)
        if not m:
            return
        val = float(m.group(1))
        window = full_text[max(0, m.start() - 40): m.end() + 40].lower()

        if std_key == "crp" and ("mg/dl" in window) and ("mg/l" not in window):
            val *= 10.0
        if std_key == "albumin" and "g/l" in window:
            val /= 10.0

        if std_key in ("lymphs_pct", "lymphs_abs"):
            aux[std_key] = val
            evidence[std_key] = (val, window.strip())
        else:
            labs[std_key] = val
            evidence[std_key] = (val, window.strip())

    # scan all synonyms in free text
    for label, key in synonyms:
        from_text(label, key)

    # --- Extra ALP fallbacks ---
    if "alp" not in labs:
        m = re.search(r"(?i)\bALP\b[^\n]{0,40}?(-?\d+(?:\.\d+)?)", full_text)
        if m:
            with contextlib.suppress(Exception):
                labs["alp"] = float(m.group(1))

    if "alp" not in labs:
        m = re.search(r"(?i)alk[^\n]{0,80}?phos[^\n]{0,80}?(-?\d+(?:\.\d+)?)", full_text)
        if m:
            with contextlib.suppress(Exception):
                labs["alp"] = float(m.group(1))

    if "alp" not in labs:
        m = re.search(r"(?i)(alk(?:aline)?\s+phosph(?:atase|ate))[^\n]{0,60}?(-?\d+(?:\.\d+)?)", full_text)
        if m:
            with contextlib.suppress(Exception):
                labs["alp"] = float(m.group(2))

    # --- derive lymph % from absolute if needed (needs WBC; sanitize will rescale WBC if cells/µL) ---
    if "lymphs_pct" in aux and "lymphs" not in labs:
        labs["lymphs"] = aux["lymphs_pct"]
        evidence["lymphs"] = evidence.get("lymphs_pct", (labs["lymphs"], "% cell"))

    if ("lymphs" not in labs) and ("lymphs_abs" in aux) and ("wbc" in labs):
        try:
            abs_cells = float(aux["lymphs_abs"])   # cells/µL
            wbc_k    = float(labs["wbc"])         # may still be cells/µL; fixed in sanitize_labs
            pct = (abs_cells / (wbc_k * 1000.0)) * 100.0
            labs["lymphs"] = pct
            evidence["lymphs"] = (pct, f"derived from abs {aux['lymphs_abs']} and wbc {labs['wbc']}")
        except Exception:
            pass

    # for optional UI debugging
    labs["_evidence"] = evidence
    return labs



def sanitize_labs(labs: dict) -> dict:
    """
    Normalize units and clamp to plausible ranges.
    - WBC: cells/µL → 10^3/µL if needed
    - Lymphocytes: fix absolute vs percent confusion; clamp 1–80%
    - CRP: assume mg/L; if extreme, treat as mg/dL*10
    """
    x = dict(labs) if labs else {}

    # WBC normalization
    if "wbc" in x and x["wbc"] is not None:
        try:
            w = float(x["wbc"])
            # If large, it's almost certainly cells/µL (e.g., 5400) → convert to 10^3/µL (5.4)
            if w > 100:
                w = w / 1000.0
            x["wbc"] = float(np.clip(w, 0.1, 50.0))
        except Exception:
            pass

    # Lymphocytes normalization
    # a) compute % from abs if needed (requires wbc already normalized)
    if x.get("lymphs") is None and x.get("lymphs_abs") is not None and x.get("wbc") not in (None, 0):
        try:
            abs_cells = float(x["lymphs_abs"])
            wbc_k     = float(x["wbc"])
            pct = (abs_cells / (wbc_k * 1000.0)) * 100.0
            x["lymphs"] = pct
        except Exception:
            pass

    # b) fix when a percent looks like absolute or is off by ×10 or ×100
    if x.get("lymphs") is not None:
        try:
            l = float(x["lymphs"])
            if l > 1000 and x.get("wbc"):  # absolute mistaken as percent
                l = (l / (float(x["wbc"]) * 1000.0)) * 100.0
            if l > 100:                     # e.g., 175.3 → 17.53
                l = l / 10.0 if l <= 300 else l / 100.0
            x["lymphs"] = float(np.clip(l, 1.0, 80.0))
        except Exception:
            pass

    # CRP normalization to mg/L
    if x.get("crp") is not None:
        try:
            c = float(x["crp"])
            # if someone reported mg/dL and we missed it, values like 0.3 mg/dL → 3 mg/L
            if 0 < c < 2 and ("_evidence" in x) and isinstance(x["_evidence"], dict):
                # leave small values alone unless evidence said mg/dL; handled in parser
                pass
            if c > 200:  # crazy large values → probably mg/dL; convert
                c = c * 10.0
            x["crp"] = float(np.clip(c, 0.0, 500.0))
        except Exception:
            pass

    # Clip typical ranges (prevents model blow-ups)
    def _clip(k, lo, hi):
        if x.get(k) is None:
            return
        with contextlib.suppress(Exception):
            x[k] = float(np.clip(float(x[k]), lo, hi))

    _clip("albumin", 2.0, 6.5)       # g/dL
    _clip("glucose", 40.0, 500.0)    # mg/dL (fasting or random — handled same)
    _clip("creatinine", 0.2, 12.0)   # mg/dL
    _clip("alp", 10.0, 1500.0)       # U/L
    _clip("mcv", 60.0, 130.0)        # fL
    _clip("rdw", 8.0, 30.0)          # %
    # wbc already clipped
    return x

# ---------------- Marker map & severity ----------------
MARKER_MAP = {
    # marker_key -> specs + model target code for per-food predictions
    "glucose":         {"label":"Glucose",          "units":"mg/dL",  "dir":"high", "goal":(70, 99),   "target":"LBXSGL"},
    "crp_mgL":         {"label":"CRP (hs)",         "units":"mg/L",   "dir":"high", "goal":(0.0, 3.0), "target":"LBXCRP"},  # model may be missing; handled gracefully
    "albumin":         {"label":"Albumin",          "units":"g/dL",   "dir":"low",  "goal":(3.8, 5.0), "target":"LBXSAL"},
    "lymphocyte_pct":  {"label":"Lymphocytes",      "units":"%",      "dir":"low",  "goal":(20, 40),   "target":None},
    "rdw":             {"label":"RDW",              "units":"%",      "dir":"high", "goal":(11.5,14.5),"target":"LBXRDW"},
    "alk_phosphatase": {"label":"Alk Phosphatase",  "units":"U/L",    "dir":"high", "goal":(44, 120),  "target":"LBXSAPSI"},
    "wbc":             {"label":"WBC",              "units":"10^3/µL","dir":"high", "goal":(4.0,10.5), "target":"LBXWBCSI"},
    "creatinine":      {"label":"Creatinine",       "units":"mg/dL",  "dir":"high", "goal":(0.6,1.3),  "target":"LBXSCR"},
    "mcv":             {"label":"MCV",              "units":"fL",     "dir":"high", "goal":(80, 100),  "target":"LBXMCVSI"},
}

def _status_and_severity(value: float | None, goal: tuple[float, float] | None, direc: str) -> tuple[str, float]:
    if value is None or goal is None: return ("ok", 0.0)
    lo, hi = goal
    if direc == "high":
        if value <= hi: return ("ok", 0.0)
        sev = (value - hi) / max(1.0, hi); return ("high", clip01(sev))
    if direc == "low":
        if value >= lo: return ("ok", 0.0)
        sev = (lo - value) / max(1.0, lo); return ("low", clip01(sev))
    return ("ok", 0.0)

def compute_marker_severity(labs_row: pd.Series) -> dict:
    out = {}
    for key, meta in MARKER_MAP.items():
        val = labs_row.get(key)
        try: val = None if pd.isna(val) else float(val)
        except Exception: val = None
        status, sev = _status_and_severity(val, meta.get("goal"), meta.get("dir","ok"))
        out[key] = {"value":val, "status":status, "severity":sev, "label":meta["label"], "units":meta["units"]}
    return out

# ---------------- Per-food predictor ----------------
@st.cache_data(show_spinner=False)
def load_perfood_bundle():
    """Load the per-marker ML bundle from models/PerFood (or models/Perfood)."""
    mdl_dir = (root / "models" / "PerFood")
    if not mdl_dir.exists():
        mdl_dir = (root / "models" / "Perfood")
    if not mdl_dir.exists():
        return None

    meta = {}
    meta_path = mdl_dir / "meta.json"
    if meta_path.exists():
        with contextlib.suppress(Exception):
            meta = json.load(open(meta_path))

    try:
        from joblib import load as joblib_load
    except Exception:
        st.error("Missing dependency 'joblib'. Add it to requirements.txt.")
        return None

    scaler = None
    scaler_path = mdl_dir / "X_scaler.joblib"
    if scaler_path.exists():
        with contextlib.suppress(Exception):
            scaler = joblib_load(scaler_path)

    models = {}
    for p in mdl_dir.glob("*.joblib"):
        if p.name == "X_scaler.joblib":
            continue
        with contextlib.suppress(Exception):
            models[p.stem] = joblib_load(p)

    if not models:
        return None

    return {"dir": str(mdl_dir), "meta": meta, "scaler": scaler, "models": models}



    scal_path = mdl_dir / PERFOOD_SCALER
    if scal_path.exists():
        with contextlib.suppress(Exception):
            scaler = joblib_load(scal_path)

    # models: load all *.joblib except the scaler
    models = {}
    for p in mdl_dir.glob("*.joblib"):
        if p.name == PERFOOD_SCALER:
            continue
        with contextlib.suppress(Exception):
            models[p.stem] = joblib_load(p)

    if not models:
        # Nothing usable found
        return None

    return {"dir": str(mdl_dir), "meta": meta, "scaler": scaler, "models": models}

def build_feature_matrix(food_df: pd.DataFrame, fnd_df: pd.DataFrame | None, required: list[str] | None) -> pd.DataFrame | None:
    """
    Return DataFrame X with shape [n_foods, n_features] aligned to required features.
    We use FNDDS features if present; else try columns in catalog.
    """
    if required is None:
        # auto-discover NUTR_* columns
        src = fnd_df if (fnd_df is not None) else food_df
        cols = [c for c in src.columns if str(c).startswith("NUTR_")]
        if not cols: return None
        required = cols
    if fnd_df is not None and set(required).issubset(fnd_df.columns):
        X = food_df[["FoodCode"]].merge(fnd_df[["FoodCode"] + required], on="FoodCode", how="left")
    else:
        if not set(required).issubset(food_df.columns): return None
        X = food_df[["FoodCode"] + required].copy()
    # fill missing with zeros (or small eps)
    for c in required:
        if c not in X.columns: X[c] = 0.0
    return X.set_index("FoodCode")[required].astype(float)

def predict_targets(bundle, X: pd.DataFrame) -> pd.DataFrame:
    """Apply scaler + each target model; return DF indexed by FoodCode with columns per target."""
    scaler = bundle["scaler"]; models = bundle["models"]
    if X is None or X.empty or not models: return pd.DataFrame(index=X.index)
    Xs = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    preds = {}
    for tgt, mdl in models.items():
        try:
            preds[tgt] = np.asarray(mdl.predict(Xs)).astype(float)
        except Exception:
            continue
    if not preds: return pd.DataFrame(index=X.index)
    P = pd.DataFrame(preds, index=X.index)
    return P

# ---------------- Preference adjustments (diet/exclusions/dislikes) ----------------
_EXCL_PATTERNS = {
    "dairy-free":      re.compile(r"\b(milk|cheese|yogurt|kefir|cream|butter|whey|casein|ghee|custard|ice cream)\b", re.I),
    "gluten-free":     re.compile(r"\b(wheat|barley|rye|farro|couscous|bulgur|seitan)\b", re.I),
    "nut-free":        re.compile(r"\b(almond|walnut|pecan|hazelnut|pistachio|cashew|macadamia)\b", re.I),
    "shellfish-free":  re.compile(r"\b(shrimp|prawn|crab|lobster|oyster|clam|scallop|mussel)\b", re.I),
    "egg-free":        re.compile(r"\b(egg|eggs)\b", re.I),
    "soy-free":        re.compile(r"\b(soy|soya|tofu|tempeh|edamame|soybean)\b", re.I),
    "pork-free":       re.compile(r"\b(pork|ham|bacon)\b", re.I),
}
_DIET_BLOCK = {
    "Vegan":        re.compile(r"\b(beef|pork|chicken|turkey|lamb|veal|fish|salmon|tuna|shrimp|oyster|clam|crab|lobster|egg|milk|cheese|yogurt|kefir)\b", re.I),
    "Vegetarian":   re.compile(r"\b(beef|pork|chicken|turkey|lamb|veal|fish|salmon|tuna|shrimp|oyster|clam|crab|lobster)\b", re.I),
    "Pescatarian":  re.compile(r"\b(beef|pork|chicken|turkey|lamb|veal)\b", re.I),
}

def apply_hard_filters(df: pd.DataFrame, diet_pattern: str, exclusions: list[str], dislikes: str) -> pd.DataFrame:
    R = df.copy()
    desc = R["Desc"].fillna("").astype(str)
    block_re = _DIET_BLOCK.get(diet_pattern)
    if block_re is not None: R = R[~desc.str.contains(block_re)]
    for ex in exclusions:
        ex_re = _EXCL_PATTERNS.get(ex)
        if ex_re is not None: R = R[~desc.str.contains(ex_re)]
    if dislikes.strip():
        bads = [re.escape(x.strip()) for x in dislikes.split(",") if x.strip()]
        if bads:
            bad_re = re.compile(r"(" + "|".join(bads) + r")", re.I)
            R = R[~desc.str.contains(bad_re)]
    return R

# ---------------- UI ----------------
st.title("Blood→Food: BioAge + Marker-Targeted Recs")

with st.sidebar:
    st.markdown("**Data paths**")
    st.text(f"Root: {root}")
    include_tags = st.text_input("Include tags (comma-separated)", value="")
    exclude_tags = st.text_input("Exclude tags (comma-separated)", value="")

    st.markdown("**Preferences**")
    diet_pattern = st.selectbox("Diet pattern", ["Omnivore","Pescatarian","Vegetarian","Vegan","Mediterranean","DASH","Keto-lite"], index=0)
    exclusions   = st.multiselect("Hard exclusions", ["dairy-free","gluten-free","nut-free","shellfish-free","egg-free","soy-free","pork-free"], default=[])
    dislikes     = st.text_input("Avoid ingredients (comma-separated)", value="")
    top_n_show   = st.slider("Rows to show (tables)", 20, 200, 100, step=10)

    st.markdown("**Model blend**")
    w_marker  = st.slider("Marker weighting (↑ = more lab-targeted)", 0.0, 2.0, 0.8, 0.1)
    w_bioage  = st.slider("BioAge weight (z-scaled)", 0.0, 2.0, 1.0, 0.1)

    st.markdown("---")
    pdf_file = st.file_uploader("Upload PDF blood test", type=["pdf"])
    csv_file = st.file_uploader("Or upload CSV", type=["csv"])
    age_input = st.number_input("Your age (years)", min_value=0, max_value=120, value=45)
    run_clicked = st.button("Parse labs & compute recommendations")

schema   = load_lab_schema()
catalog  = load_catalog() if CAT_PARQUET.exists() else None
fnd_nutr = load_fnd_features()  # you’ll pass this into your recommender


parsed_df = None
if run_clicked:
    labs_dict = {}
    try:
        if pdf_file is not None:
            data = pdf_file.read()
            raw_labs = parse_pdf_labs(io.BytesIO(data))
            labs_dict = sanitize_labs(raw_labs)
        elif csv_file is not None:
            raw = pd.read_csv(csv_file)
            norm = normalize_labs(raw, schema)
            labs_dict = sanitize_labs(norm.iloc[0].to_dict())
        else:
            st.error("Please upload a PDF or CSV first.")
    except Exception as e:
        st.error(f"Failed to parse labs: {e}")

    if labs_dict:
        mapped = {
            "age_years": age_input,
            "albumin": labs_dict.get("albumin"),
            "creatinine": labs_dict.get("creatinine"),
            "glucose": labs_dict.get("fasting_glucose") or labs_dict.get("glucose"),
            "crp_mgL": labs_dict.get("crp"),
            "lymphocyte_pct": labs_dict.get("lymphs"),
            "mcv": labs_dict.get("mcv"),
            "rdw": labs_dict.get("rdw"),
            "alk_phosphatase": labs_dict.get("alp"),
            "wbc": labs_dict.get("wbc"),
        }
        parsed_df = pd.DataFrame([mapped])

# ---- Labs + BioAge metrics ----
# -------- 2) Show normalized labs & BioAge --------
st.subheader("2) Parsed labs & BioAge")

def _have_all_needed_cols(df: pd.DataFrame) -> bool:
    need = ["age_years","albumin","creatinine","glucose","crp_mgL",
            "lymphocyte_pct","mcv","rdw","alk_phosphatase","wbc"]
    return all(c in df.columns for c in need)

if isinstance(parsed_df, pd.DataFrame) and not parsed_df.empty and _have_all_needed_cols(parsed_df):
    st.caption("Parsed (normalized) labs used for scoring:")
    st.dataframe(parsed_df, use_container_width=True)

    # ensure empty strings/None become NaN for the model check
    safe_row = parsed_df.iloc[0].apply(lambda v: np.nan if v in (None, "") else v)
    bio, accel = phenoage_from_row(safe_row)

    c1, c2 = st.columns(2)
    with c1: st.metric("BioAge (PhenoAge)", f"{bio:.1f}" if bio is not None else "–")
    with c2: st.metric("BioAgeAccel", f"{accel:+.1f}" if accel is not None else "–")
else:
    st.info("Upload a PDF/CSV, enter age, then click the button to parse and compute BioAge.")


# ---- Recommendations ----
st.subheader("Food recommendations")

if parsed_df is None:
    st.info("Awaiting labs…")
elif catalog is None:
    st.error("Food catalog not found (app_assets/food_catalog.parquet).")
else:
    # 1) Filter + de-dupe + category
    R_all = dedup_rank(catalog, include_tags, exclude_tags).copy()
    R_all["category"] = [coarse_category(d, t) for d, t in zip(R_all["Desc"], R_all["tags"])]

    # 2) Apply user hard filters
    R_all = apply_hard_filters(R_all, diet_pattern, exclusions, dislikes)

    # 3) Per-food predictions (if models available)
    bundle = load_perfood_bundle()
    if bundle is None:
        st.warning("Per-marker ML bundle not found. Skipping per-marker recommendations. "
                   "Ensure models/PerFood (or models/Perfood) contains meta.json, X_scaler.joblib, and *.joblib models.")
    else:
        req_feats = bundle.get("features")
        X = build_feature_matrix(R_all, fnd_feats, req_feats)
        if X is None:
            st.warning("Could not build NUTR_* feature matrix for foods. Ensure FNDDS parquet (processed/FNDDS_MASTER_PER100G.parquet) or catalog has NUTR_* columns.")
        else:
            P = predict_targets(bundle, X)

    # 4) Marker severity
    sev = compute_marker_severity(parsed_df.iloc[0])

    # 5) Marker impact from predictions
    impact_total = pd.Series(0.0, index=R_all["FoodCode"].astype("Int64"))
    per_marker_tables = {}  # for the marker cards

    if P is not None and not P.empty:
        # robust scale per predicted target for goal-oriented benefit
        def goal_benefit(target_code: str, pred: pd.Series, meta: dict) -> pd.Series:
            # robust sigma of predicted distribution
            med = np.nanmedian(pred); iqr = np.nanpercentile(pred, 75) - np.nanpercentile(pred, 25)
            sigma = iqr/1.349 if iqr and np.isfinite(iqr) and iqr>0 else np.nanstd(pred)
            if not (sigma and np.isfinite(sigma) and sigma>0): sigma = 1.0
            lo, hi = meta["goal"]
            if meta["dir"] == "high":
                b = (hi - pred) / sigma  # higher benefit if prediction is below high goal
            else:
                b = (pred - lo) / sigma  # higher benefit if prediction is above low goal
            return b.clip(-3, 3)  # guard extremes

        r2_map = bundle.get("r2_map", {})
        # Build impact per marker
        for mkey, meta in MARKER_MAP.items():
            tgt = meta.get("target")
            if not tgt or tgt not in P.columns:
                continue  # no model for this marker
            sev_w = sev.get(mkey, {}).get("severity", 0.0)
            if sev_w <= 0 and mkey != "glucose":
                # if inside goal, give tiny weight so list still varies; glucose keeps small weight for energy balance
                sev_w = 0.1
            conf = clip01((float(r2_map.get(tgt, 0.0)) - 0.10) / 0.20)  # ~0 at r2<=0.10, ~1 near 0.30+
            if conf <= 0:
                continue

            pred = P[tgt]
            benefit = goal_benefit(tgt, pred, meta)
            imp = sev_w * conf * benefit

            # accumulate
            impact_total = impact_total.add(pd.Series(imp.values, index=P.index), fill_value=0.0)

            # store per-marker top-k for UI cards
            dfm = (R_all.set_index("FoodCode")
                      .assign(pred=pred, impact=imp)
                      .sort_values("impact", ascending=False)
                      [["Desc","impact"]]
                      .rename(columns={"impact":"impact_score"}))
            per_marker_tables[mkey] = dfm.reset_index()

    # 6) Blend with BioAge score
    base = pd.to_numeric(R_all["score"], errors="coerce").astype(float)
    base_z = robust_z(base)
    blended = w_bioage * base_z - w_marker * impact_total.reindex(R_all["FoodCode"].astype("Int64")).fillna(0.0).values
    R_all["score_final"] = blended

    # 7) Show overall table
    top_overall = R_all.sort_values("score_final", ascending=True).head(100).copy()
    st.markdown("**Top 100 overall (lower = better; blended BioAge + Marker)**")
    st.dataframe(
        top_overall[["FoodCode","Desc","kcal_per_100g","score_final","tags","category"]]
                   .rename(columns={"score_final":"score"}),
        use_container_width=True
    )

    # 8) Marker cards (from per-marker impacts, if any)
    st.subheader("Foods by marker (model-targeted)")
    cols = st.columns(2); i = 0
    for mkey, meta in MARKER_MAP.items():
        info = sev.get(mkey, {})
        val = info.get("value"); status = info.get("status"); units = meta["units"]; label = meta["label"]
        if status == "high": glyph = "🔺"
        elif status == "low": glyph = "🔻"
        else: glyph = "✅"
        with cols[i % 2]:
            with st.container():
                st.markdown(f"**{label}** — {glyph} {('%.2f' % val) if val is not None else '—'} {units}")
                dfk = per_marker_tables.get(mkey)
                if dfk is None or dfk.empty:
                    st.caption("No model or no strong matches for this marker.")
                else:
                    show = (dfk.sort_values("impact_score", ascending=False)
                              .head(10)[["FoodCode","Desc","impact_score"]]
                              .rename(columns={"impact_score":"impact"}))
                    st.dataframe(show, use_container_width=True, hide_index=True)
        i += 1

    # 9) Category tabs using blended score
    st.subheader("Browse by category")
    tabs = st.tabs(["Protein","Fats","Fruit","Vegetables","Legume/Grains","Other"])
    CAT_ORDER = ["protein","fats","fruit","vegetables","legume_grains","other"]
    QUOTA = {"protein":5, "fats":5, "fruit":7, "vegetables":10, "legume_grains":4, "other":top_n_show}
    for tab, cat in zip(tabs, CAT_ORDER):
        with tab:
            sub = (R_all[R_all["category"]==cat]
                   .sort_values("score_final")
                   .head(QUOTA.get(cat, 10)))
            if sub.empty:
                st.info(f"No foods found for category: {cat.replace('_','/')}")
            else:
                st.write(f"**Top {len(sub)} — {cat.replace('_','/').title()}**")
                st.dataframe(sub[["FoodCode","Desc","kcal_per_100g","score_final","tags"]]
                               .rename(columns={"score_final":"score"}),
                             use_container_width=True)
                st.download_button(
                    f"Download {cat.replace('_','/').title()} (CSV)",
                    sub.to_csv(index=False).encode("utf-8"),
                    file_name=f"top_{cat}.csv",
                    key=f"dl_{cat}"
                )

