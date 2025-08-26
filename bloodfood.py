# bloodfood.py — Hybrid BioAge + Per-Marker ML food recommender
# - Parses labs (PDF/CSV) → computes PhenoAge + per-marker severity
# - Loads per-food LightGBM models (joblib) + scaler → predicts marker deltas per food
# - Blends BioAge score with per-marker impact + dietary preferences

from __future__ import annotations
import os, re, io, json, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# App id + page
# =============================================================================
APP_BUILD = "bf-v2.3"
st.set_page_config(page_title="Blood→Food: BioAge + Marker Recs", layout="wide")
st.sidebar.caption(f"Build: {APP_BUILD}")

# =============================================================================
# Paths
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent
env_root  = os.environ.get("BIOAGE_ROOT")
root      = Path(env_root) if env_root else REPO_ROOT

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

# =============================================================================
# Optional dependency: pdfplumber
# =============================================================================
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

# =============================================================================
# Small helpers
# =============================================================================
def robust_z(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    med = np.nanmedian(x)
    iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
    scale = iqr / 1.349 if iqr and np.isfinite(iqr) and iqr > 0 else np.nanstd(x)
    if not (scale and np.isfinite(scale) and scale > 0):
        scale = 1.0
    return (x - med) / scale

def clip01(v): return float(np.clip(v, 0.0, 1.0))

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

def _dedupe_by_desc(df: pd.DataFrame) -> pd.DataFrame:
    if "Desc" not in df.columns: return df
    df = df.copy()
    df["dedup_key"] = df["Desc"].map(_normalize_desc)
    df = df.drop_duplicates("dedup_key", keep="first").drop(columns="dedup_key")
    return df

# =============================================================================
# Loaders
# =============================================================================
@st.cache_data(show_spinner=False)
def load_lab_schema() -> dict:
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
def load_catalog() -> pd.DataFrame | None:
    if not CAT_PARQUET.exists(): return None
    cat = pd.read_parquet(CAT_PARQUET)
    cat["FoodCode"] = pd.to_numeric(cat["FoodCode"], errors="coerce").astype("Int64")
    if "tags" not in cat.columns: cat["tags"] = np.nan
    return cat

@st.cache_data(show_spinner=False)
def load_fnd_features() -> pd.DataFrame | None:
    """
    Load the FNDDS per-100g parquet and normalize columns so that:
      - FoodCode is Int64
      - Nutrient columns exist as UPPERCASE 'NUTR_*'
    Searches multiple likely locations.
    """
    candidates = [
        FND_PARQUET,
        root / "processed" / "FNDDS_MASTER_PER100G.parquet",
        Path("/mnt/data/FNDDS_MASTER_PER100G.parquet"),
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        st.sidebar.error("FNDDS parquet not found in processed/ or /mnt/data.")
        return None

    fnd = pd.read_parquet(src)

    # normalize FoodCode
    fc_col = None
    for c in fnd.columns:
        if str(c).lower().replace("_","") == "foodcode":
            fc_col = c; break
    if fc_col and fc_col != "FoodCode":
        fnd = fnd.rename(columns={fc_col:"FoodCode"})
    if "FoodCode" not in fnd.columns:
        st.sidebar.error("FNDDS parquet has no FoodCode column.")
        return None
    fnd["FoodCode"] = pd.to_numeric(fnd["FoodCode"], errors="coerce").astype("Int64")

    # ensure NUTR_* columns (uppercased)
    nutr_cols = [c for c in fnd.columns if str(c).upper().startswith("NUTR_")]
    if not nutr_cols:
        lc = [c for c in fnd.columns if str(c).lower().startswith("nutr_")]
        if lc:
            fnd = fnd.rename(columns={c: str(c).upper() for c in lc})
            nutr_cols = [str(c).upper() for c in lc]
        else:
            # fallback: alias numeric columns into NUTR_*
            for c in list(fnd.columns):
                if c == "FoodCode": continue
                if pd.api.types.is_numeric_dtype(fnd[c]):
                    alias = "NUTR_" + re.sub(r"[^A-Za-z0-9]+","_",str(c)).upper()
                    if alias not in fnd.columns:
                        fnd[alias] = fnd[c]
            nutr_cols = [c for c in fnd.columns if str(c).upper().startswith("NUTR_")]

    st.sidebar.caption(f"FNDDS parquet: {src} • nutrients: {len(nutr_cols)}")
    return fnd[["FoodCode"] + nutr_cols].copy()

# CSV normalizer for fallback CSV uploads
def normalize_labs(df_in: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df_in.copy()
    alias_map = {a.lower(): std for std, meta in schema.items() for a in meta.get("aliases", [])}
    rename = {col: alias_map[col.lower()] for col in df.columns if col.lower() in alias_map}
    if rename: df = df.rename(columns=rename)
    keep = list(schema.keys())
    cols_present = [c for c in keep if c in df.columns]
    out = df[cols_present].copy()
    for c in cols_present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# =============================================================================
# PhenoAge (Levine)
# =============================================================================
def phenoage_from_row(row: pd.Series):
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

# =============================================================================
# Food catalog helpers
# =============================================================================
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

# Whole-foods filter (default ON)
WHOLE_FOODS_BAD = re.compile(
    r"\b(margarine|made with|broth|sauce|dressing|gravy|salad|soup|mix|mixes|"
    r"polishings|cereal|bran|baby ?food|formula|powder|bar|chips|cookie|cake|pie|"
    r"pudding|fried|battered|breaded|drink|soft drink|soda|beverage|energy drink)\b", re.I
)
def filter_whole_foods(df: pd.DataFrame) -> pd.DataFrame:
    if "Desc" not in df.columns: return df
    return df[
        ~df["Desc"].astype(str).str.contains(WHOLE_FOODS_BAD, na=False)
        & ~df["tags"].fillna("").str.contains(r"cereal_fortified|oil_fat_sauce|baby", case=False, regex=True)
    ]

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

    def row_has_token(row_text: str, token: str) -> bool:
        return token in (row_text or "").lower()

    def desc_has_keyword(desc: str, token: str) -> bool:
        return re.search(rf"\b{re.escape(token)}(es|s)?\b", str(desc or ""), re.I) is not None

    if include_tags.strip():
        inc = [t.strip().lower() for t in include_tags.split(",") if t.strip()]
        R = R[
            R.apply(lambda r: any(row_has_token(r.get("tags",""), t) or
                                  desc_has_keyword(r.get("Desc",""), t) for t in inc), axis=1)
        ]

    if exclude_tags.strip():
        exc = [t.strip().lower() for t in exclude_tags.split(",") if t.strip()]
        R = R[
            ~R.apply(lambda r: any(row_has_token(r.get("tags",""), t) or
                                   desc_has_keyword(r.get("Desc",""), t) for t in exc), axis=1)
        ]

    R["dedup_key"] = R["Desc"].map(_normalize_desc)
    R = R.sort_values(["score", "kcal_per_100g"], ascending=[True, True])
    R = R.drop_duplicates(subset="dedup_key", keep="first").drop(columns=["dedup_key"])
    return R.reset_index(drop=True)

# =============================================================================
# PDF → labs → sanitize
# =============================================================================
def parse_pdf_labs(file_like) -> dict:
    if not HAS_PDFPLUMBER:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")

    synonyms = [
        ("serum albumin","albumin"), ("albumin, serum","albumin"),
        ("albumin (serum)","albumin"), ("alb","albumin"), ("albumin","albumin"),
        ("alkaline phosphatase (total)","alp"), ("alkaline phosphatase, total","alp"),
        ("alkaline phosphatase (alp)","alp"), ("alkaline phosphatase (alk phos)","alp"),
        ("alk phosphatase","alp"), ("alk. phosphatase","alp"), ("alk phos","alp"),
        ("alkaline phosphatase","alp"), ("alkaline phosphate","alp"), ("alp","alp"),
        ("c-reactive protein, cardiac","crp"), ("c-reactive protein (cardiac)","crp"),
        ("c reactive protein, cardiac","crp"), ("crp, cardiac","crp"),
        ("high sensitivity crp","crp"), ("hs-crp","crp"), ("hscrp","crp"),
        ("c-reactive protein","crp"), ("c reactive protein","crp"), ("crp","crp"),
        ("glucose, fasting","fasting_glucose"), ("glucose (fasting)","fasting_glucose"),
        ("fasting glucose","fasting_glucose"), ("glucose fasting","fasting_glucose"),
        ("glucose","fasting_glucose"),
        ("white blood cell count","wbc"), ("white blood cells","wbc"),
        ("white blood cell","wbc"), ("wbc","wbc"),
        ("lymphocytes absolute","lymphs_abs"), ("absolute lymphocytes","lymphs_abs"),
        ("abs lymphocytes","lymphs_abs"), ("lymphs #","lymphs_abs"),
        ("lymphocyte percent","lymphs_pct"), ("lymphocytes percent","lymphs_pct"),
        ("lymphocyte %","lymphs_pct"), ("lymphocytes %","lymphs_pct"),
        ("lymphocytes","lymphs_pct"), ("lymphs","lymphs_pct"),
        ("mean corpuscular volume","mcv"), ("mcv","mcv"),
        ("red cell distribution width","rdw"), ("rdw","rdw"),
        ("creatinine","creatinine"), ("creat","creatinine"),
        ("blood urea nitrogen","bun"), ("bun","bun"),
    ]

    range_pat   = re.compile(r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\b")
    number_pat  = re.compile(r"(-?\d+(?:\.\d+)?)")
    percent_pat = re.compile(r"%")

    def match_key(lbl: str) -> str | None:
        s = (lbl or "").strip().lower()
        for name, key in sorted(synonyms, key=lambda kv: -len(kv[0])):
            if name in s: return key
        return None

    def first_numeric(txt: str, prefer_percent=False):
        if not txt or range_pat.search(txt): return (None, None)
        m = number_pat.search(txt)
        if not m: return (None, None)
        val = float(m.group(1)); has_pct = bool(percent_pat.search(txt))
        score = (1 if (prefer_percent and has_pct) else 0, -len(txt))
        return val, score

    labs, aux, evidence = {}, {}, {}

    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            for tbl in (page.extract_tables() or []):
                for r_i, row in enumerate(tbl or []):
                    if not row or all(c is None or str(c).strip()=="" for c in row): continue
                    label = str(row[0] or ""); key = match_key(label)
                    if not key: continue

                    prefer_pct = (key == "lymphs_pct")
                    best = None
                    for cell in row[1:]:
                        txt = str(cell or ""); val, score = first_numeric(txt, prefer_percent=prefer_pct)
                        if val is None: continue
                        cand = (score, val, txt)
                        if (best is None) or (cand > best): best = cand

                    if best is None and len(row) > 1 and r_i + 1 < len(tbl):
                        below = str((tbl[r_i+1] or [""])[1] or "")
                        val, score = first_numeric(below, prefer_percent=prefer_pct)
                        if val is not None: best = (score, val, below)
                    if best is None: continue

                    _, value, rawtxt = best
                    lowtxt = (rawtxt or "").lower()
                    if key == "crp" and ("mg/dl" in lowtxt) and ("mg/l" not in lowtxt): value *= 10.0
                    if key == "albumin" and "g/l" in lowtxt: value /= 10.0

                    if key == "lymphs_abs":
                        aux["lymphs_abs"] = value; evidence["lymphs_abs"] = (value, rawtxt)
                    elif key == "lymphs_pct":
                        aux["lymphs_pct"] = value; evidence["lymphs_pct"] = (value, rawtxt)
                    elif key == "wbc":
                        labs["wbc"] = value;    evidence["wbc"] = (value, rawtxt)
                    else:
                        labs[key] = value;      evidence[key] = (value, rawtxt)

    file_like.seek(0)
    full_text = ""
    with pdfplumber.open(file_like) as pdf:
        for p in pdf.pages:
            full_text += (p.extract_text() or "").replace("\n", " ") + " "

    def from_text(label, std_key):
        if std_key in labs or std_key in aux: return
        pat = re.compile(rf"(?i)\b{re.escape(label)}\b" + r".{0,120}?" + r"(-?\d+(?:\.\d+)?)")
        m = pat.search(full_text)
        if not m: return
        val = float(m.group(1))
        window = full_text[max(0, m.start()-40): m.end()+40].lower()
        if std_key == "crp" and ("mg/dl" in window) and ("mg/l" not in window): val *= 10.0
        if std_key == "albumin" and "g/l" in window: val /= 10.0
        if std_key in ("lymphs_pct","lymphs_abs"):
            aux[std_key] = val;  evidence[std_key] = (val, window.strip())
        else:
            labs[std_key] = val; evidence[std_key] = (val, window.strip())

    for label, key in synonyms: from_text(label, key)

    # extra ALP fallbacks (expanded onto multiple lines to avoid syntax error)
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

    if "lymphs_pct" in aux and "lymphs" not in labs:
        labs["lymphs"] = aux["lymphs_pct"]; evidence["lymphs"] = evidence.get("lymphs_pct",(labs["lymphs"],"% cell"))
    if ("lymphs" not in labs) and ("lymphs_abs" in aux) and ("wbc" in labs):
        try:
            abs_cells = float(aux["lymphs_abs"]); wbc_k = float(labs["wbc"])
            pct = (abs_cells / (wbc_k * 1000.0)) * 100.0
            labs["lymphs"] = pct; evidence["lymphs"] = (pct, f"derived from abs {aux['lymphs_abs']} and wbc {labs['wbc']}")
        except Exception: pass

    labs["_evidence"] = evidence
    return labs

def sanitize_labs(labs: dict) -> dict:
    x = dict(labs) if labs else {}

    if "wbc" in x and x["wbc"] is not None:
        try:
            w = float(x["wbc"])
            if w > 100: w = w / 1000.0
            x["wbc"] = float(np.clip(w, 0.1, 50.0))
        except Exception: pass

    if x.get("lymphs") is None and x.get("lymphs_abs") is not None and x.get("wbc") not in (None, 0):
        try:
            abs_cells = float(x["lymphs_abs"]); wbc_k = float(x["wbc"])
            x["lymphs"] = (abs_cells / (wbc_k * 1000.0)) * 100.0
        except Exception: pass
    if x.get("lymphs") is not None:
        try:
            l = float(x["lymphs"])
            if l > 1000 and x.get("wbc"): l = (l / (float(x["wbc"]) * 1000.0)) * 100.0
            if l > 100: l = l/10.0 if l <= 300 else l/100.0
            x["lymphs"] = float(np.clip(l, 1.0, 80.0))
        except Exception: pass

    if x.get("crp") is not None:
        try:
            c = float(x["crp"])
            if c > 200: c = c * 10.0
            x["crp"] = float(np.clip(c, 0.0, 500.0))
        except Exception: pass

    def _clip(k, lo, hi):
        if x.get(k) is None: return
        with contextlib.suppress(Exception):
            x[k] = float(np.clip(float(x[k]), lo, hi))
    _clip("albumin", 2.0, 6.5)
    _clip("glucose", 40.0, 500.0)
    _clip("creatinine", 0.2, 12.0)
    _clip("alp", 10.0, 1500.0)
    _clip("mcv", 60.0, 130.0)
    _clip("rdw", 8.0, 30.0)
    return x

# =============================================================================
# Marker map & severity
# =============================================================================
MARKER_MAP = {
    "glucose":         {"label":"Glucose",          "units":"mg/dL",  "dir":"high", "goal":(70, 99),   "target":"LBXSGL"},
    "crp_mgL":         {"label":"CRP (hs)",         "units":"mg/L",   "dir":"high", "goal":(0.0, 3.0), "target":"LBXCRP"},
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

# =============================================================================
# Per-food predictor
# =============================================================================
@st.cache_data(show_spinner=False)
def load_perfood_bundle(base_dir: Path | None = None):
    """
    Load per-marker ML bundle from models/PerFood (or models/Perfood).
    Returns dict with: dir, meta, features (aligned to scaler), scaler, models, r2_map — or None.
    """
    base = Path(base_dir) if base_dir else (root / "models")
    mdl_dir = base / "PerFood"
    if not mdl_dir.exists():
        mdl_dir = base / "Perfood"
    if not mdl_dir.exists():
        st.warning(f"PerFood models folder not found under {base}.")
        return None

    # meta.json (optional)
    meta = {}
    meta_path = mdl_dir / "meta.json"
    if meta_path.exists():
        with contextlib.suppress(Exception):
            meta = json.load(open(meta_path))
    features = meta.get("features") or meta.get("feature_names")

    # joblib
    try:
        from joblib import load as joblib_load
    except Exception:
        st.error("Missing dependency 'joblib'. Add it to requirements.txt and redeploy.")
        return None

    # scaler (optional)
    scaler = None
    scal_path = mdl_dir / "X_scaler.joblib"
    if scal_path.exists():
        with contextlib.suppress(Exception):
            scaler = joblib_load(scal_path)

    # Align features to scaler, if present
    if scaler is not None:
        f_in = getattr(scaler, "feature_names_in_", None)
        if f_in is not None:
            features = [str(x) for x in list(f_in)]
        else:
            n_in = int(getattr(scaler, "n_features_in_", 0) or 0)
            if n_in > 0:
                # If meta features length mismatches scaler, pad/truncate deterministically.
                base_feats = (features or [])
                if len(base_feats) != n_in:
                    st.sidebar.warning(f"Aligning features to scaler: meta={len(base_feats)} vs scaler={n_in}")
                features = (base_feats + [f"__PAD_{i}__" for i in range(n_in)])[:n_in]

    # load all LightGBM models (*.joblib) except the scaler
    models = {}
    for p in mdl_dir.glob("*.joblib"):
        if p.name == "X_scaler.joblib":
            continue
        try:
            models[p.stem] = joblib_load(p)
        except Exception as e:
            st.warning(f"Failed to load {p.name}: {e}")

    if not models:
        st.error(f"No *.joblib models found in {mdl_dir}.")
        return None

    # per-target R² (optional)
    r2_map = {}
    r2_path = mdl_dir / "per_target_r2.csv"
    if r2_path.exists():
        with contextlib.suppress(Exception):
            r2df = pd.read_csv(r2_path)
            if {"target","r2"}.issubset(r2df.columns):
                r2_map = dict(zip(r2df["target"], r2df["r2"]))

    st.sidebar.caption(f"PerFood dir: {mdl_dir}")
    return {"dir": str(mdl_dir), "meta": meta, "features": features,
            "scaler": scaler, "models": models, "r2_map": r2_map}

# map model filename stems to NHANES codes
_MODEL_ALIAS_TO_CODE = {
    "LBXSGL":"LBXSGL", "GLUCOSE":"LBXSGL", "GLU":"LBXSGL",
    "LBXCRP":"LBXCRP", "CRP":"LBXCRP", "HSCRP":"LBXCRP",
    "LBXSAL":"LBXSAL", "ALBUMIN":"LBXSAL", "ALB":"LBXSAL",
    "LBXSCR":"LBXSCR", "CREATININE":"LBXSCR", "CREAT":"LBXSCR", "CREA":"LBXSCR",
    "LBXWBCSI":"LBXWBCSI", "WBC":"LBXWBCSI",
    "LBXRDW":"LBXRDW", "RDW":"LBXRDW",
    "LBXSAPSI":"LBXSAPSI", "ALP":"LBXSAPSI", "ALKPHOS":"LBXSAPSI", "ALKALINEPHOSPHATASE":"LBXSAPSI",
    "LBXMCVSI":"LBXMCVSI", "MCV":"LBXMCVSI",
}
_KNOWN_CODES = ["LBXSGL","LBXCRP","LBXSAL","LBXSCR","LBXWBCSI","LBXRDW","LBXSAPSI","LBXMCVSI"]

def _stem_to_code(stem: str) -> str | None:
    u = re.sub(r"[^A-Z0-9]+","", str(stem).upper())
    for code in _KNOWN_CODES:
        if code in u: return code
    for alias, code in _MODEL_ALIAS_TO_CODE.items():
        if alias in u: return code
    tok = (str(stem).split("_")[-1]).upper()
    return _MODEL_ALIAS_TO_CODE.get(tok)

def build_feature_matrix(food_df: pd.DataFrame,
                         fnd_df: pd.DataFrame | None,
                         required: list[str] | None) -> pd.DataFrame | None:
    """
    Build X aligned to 'required' features (order preserved).
    - Case/underscore-insensitive matching
    - Missing features are added as 0.0 so the scaler/model shapes still match
    """
    if food_df is None or food_df.empty: return None
    base = fnd_df if (fnd_df is not None and not fnd_df.empty) else food_df
    if "FoodCode" not in base.columns: return None
    base = base.copy()
    base["FoodCode"] = pd.to_numeric(base["FoodCode"], errors="coerce").astype("Int64")

    if required is None:
        cols = [c for c in base.columns if str(c).upper().startswith("NUTR_")]
        if not cols: return None
        return (base[["FoodCode"] + cols].set_index("FoodCode")[cols].astype(float))

    def norm(s: str) -> str: return re.sub(r"[^a-z0-9]+","", str(s).lower())
    avail = {norm(c): c for c in base.columns}
    X = pd.DataFrame(index=pd.to_numeric(food_df["FoodCode"], errors="coerce").astype("Int64"))

    for feat in required:
        k = norm(feat)
        col = (avail.get(k) or
               avail.get(k.replace("nutr","")) or
               avail.get("nutr"+k) or
               next((orig for nk, orig in avail.items() if nk.endswith(k.replace("nutr",""))), None))
        if col is None:
            X[feat] = 0.0
        else:
            X[feat] = base.set_index("FoodCode").reindex(X.index)[col].astype(float)
    return X

# robust mapping from model filenames → NHANES codes
_MODEL_ALIAS_TO_CODE = {
    "LBXSGL":"LBXSGL","GLUCOSE":"LBXSGL","GLU":"LBXSGL",
    "LBXCRP":"LBXCRP","CRP":"LBXCRP","HSCRP":"LBXCRP",
    "LBXSAL":"LBXSAL","ALBUMIN":"LBXSAL","ALB":"LBXSAL",
    "LBXSCR":"LBXSCR","CREATININE":"LBXSCR","CREAT":"LBXSCR","CREA":"LBXSCR",
    "LBXWBCSI":"LBXWBCSI","WBC":"LBXWBCSI",
    "LBXRDW":"LBXRDW","RDW":"LBXRDW",
    "LBXSAPSI":"LBXSAPSI","ALP":"LBXSAPSI","ALKPHOS":"LBXSAPSI","ALKALINEPHOSPHATASE":"LBXSAPSI",
    "LBXMCVSI":"LBXMCVSI","MCV":"LBXMCVSI",
}
_KNOWN_CODES = list({v for v in _MODEL_ALIAS_TO_CODE.values()})

def _stem_to_code(stem: str) -> str | None:
    s = re.sub(r"[^A-Z0-9]+","", str(stem).upper())
    # exact code inside the stem
    for code in _KNOWN_CODES:
        if code in s:
            return code
    # alias inside the stem
    for alias, code in _MODEL_ALIAS_TO_CODE.items():
        if alias in s:
            return code
    # last-token alias fallback
    last = (str(stem).split("_")[-1]).upper()
    return _MODEL_ALIAS_TO_CODE.get(last)

def predict_targets(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Apply scaler (features already aligned) + each target model. Columns named by NHANES code."""
    if X is None or X.empty or not bundle or not bundle.get("models"):
        return pd.DataFrame(index=X.index if X is not None else None)

    scaler = bundle.get("scaler")
    if scaler is not None:
        try:
            Xs = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
        except Exception as e:
            st.warning(f"Scaler transform failed ({e}); using raw features.")
            Xs = X
    else:
        Xs = X

    preds = {}
    loaded_codes = []
    for stem, mdl in bundle["models"].items():
        code = _stem_to_code(stem) or stem.upper()  # never drop a model due to naming
        try:
            preds[code] = np.asarray(mdl.predict(Xs)).astype(float)
            loaded_codes.append(code)
        except Exception:
            continue

    if loaded_codes:
        st.sidebar.caption("Models loaded for: " + ", ".join(sorted(set(loaded_codes))))
    return pd.DataFrame(preds, index=X.index) if preds else pd.DataFrame(index=X.index)


# =============================================================================
# Preference filters
# =============================================================================
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

# =============================================================================
# UI
# =============================================================================
st.title("Blood→Food: BioAge + Marker-Targeted Recommendations")

with st.sidebar:
    st.markdown("**Paths**")
    st.text(f"Root: {root}")

    include_tags = st.text_input("Include tags (comma-separated)", value="")
    exclude_tags = st.text_input("Exclude tags (comma-separated)", value="")

    st.markdown("**Preferences**")
    diet_pattern = st.selectbox("Diet pattern", ["Omnivore","Pescatarian","Vegetarian","Vegan","Mediterranean","DASH","Keto-lite"], index=0)
    exclusions   = st.multiselect("Hard exclusions", ["dairy-free","gluten-free","nut-free","shellfish-free","egg-free","soy-free","pork-free"], default=[])
    dislikes     = st.text_input("Avoid ingredients (comma-separated)", value="")
    whole_only   = st.checkbox("Whole foods only", value=True)
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
catalog  = load_catalog()
fnd_nutr = load_fnd_features()

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

# ---- Labs & BioAge
st.subheader("Parsed labs & BioAge")
def _have_all_needed_cols(df: pd.DataFrame) -> bool:
    need = ["age_years","albumin","creatinine","glucose","crp_mgL",
            "lymphocyte_pct","mcv","rdw","alk_phosphatase","wbc"]
    return all(c in df.columns for c in need)

if isinstance(parsed_df, pd.DataFrame) and not parsed_df.empty and _have_all_needed_cols(parsed_df):
    st.dataframe(parsed_df, use_container_width=True)
    safe_row = parsed_df.iloc[0].apply(lambda v: np.nan if v in (None, "") else v)
    bio, accel = phenoage_from_row(safe_row)
    c1, c2 = st.columns(2)
    with c1: st.metric("BioAge (PhenoAge)", f"{bio:.1f}" if bio is not None else "–")
    with c2: st.metric("BioAgeAccel", f"{accel:+.1f}" if accel is not None else "–")
else:
    st.info("Upload & run to parse labs and compute BioAge.")

# ---- Recommendations
st.subheader("Food recommendations")

if parsed_df is None:
    st.info("Awaiting labs…")
elif catalog is None or catalog.empty:
    st.error("Food catalog not found (app_assets/food_catalog.parquet).")
else:
    # 1) filter + de-dupe + category (+ whole-foods toggle)
    base_cat = catalog.copy()
    if whole_only: base_cat = filter_whole_foods(base_cat)
    R_all = dedup_rank(base_cat, include_tags, exclude_tags).copy()
    R_all["category"] = [coarse_category(d, t) for d, t in zip(R_all["Desc"], R_all["tags"])]
    # 2) user hard filters
    R_all = apply_hard_filters(R_all, diet_pattern, exclusions, dislikes)

    # 3) per-food predictions
    P = None
    bundle = load_perfood_bundle()
    if bundle is None:
        st.warning("Per-marker ML bundle not found. Skipping per-marker recommendations.")
    else:
        req_feats = bundle.get("features")
        X = build_feature_matrix(R_all, fnd_nutr, req_feats)
        if X is None:
            st.warning("Could not build NUTR_* feature matrix. Ensure processed/FNDDS_MASTER_PER100G.parquet exists or catalog has NUTR_* columns.")
        else:
            P = predict_targets(bundle, X)
            st.caption(f"Loaded PerFood models from: {bundle['dir']}")

    # 4) marker severity
    sev = compute_marker_severity(parsed_df.iloc[0])

    # 5) marker impact
# Robust index for foods (works even if 'FoodCode' is missing)
if "FoodCode" in R_all.columns:
    food_index = pd.to_numeric(R_all["FoodCode"], errors="coerce").astype("Int64")
else:
    food_index = pd.Series(R_all.index, dtype="Int64", name="FoodCode")

impact_total = pd.Series(0.0, index=food_index)

per_marker_tables = {}

if P is not None and not P.empty:
    def goal_benefit(pred: pd.Series, meta: dict) -> pd.Series:
        iqr = np.nanpercentile(pred, 75) - np.nanpercentile(pred, 25)
        sigma = iqr / 1.349 if iqr and np.isfinite(iqr) and iqr > 0 else np.nanstd(pred)
        sigma = sigma if sigma and np.isfinite(sigma) and sigma > 0 else 1.0
        lo, hi = meta["goal"]
        b = (hi - pred) / sigma if meta["dir"] == "high" else (pred - lo) / sigma
        return b.clip(-3, 3)

    r2_map = bundle.get("r2_map", {})

    for mkey, meta in MARKER_MAP.items():          # <— LOOP
        tgt = meta.get("target")
        if not tgt:
            continue  # this marker has no model

        # find prediction column name in P
        possible_cols = [tgt, tgt.split(".")[-1], tgt.split("_")[-1]]
        col = next((c for c in possible_cols if c in P.columns), None)
        if col is None:
            continue  # no matching model column

        # weights
        sev_w = sev.get(mkey, {}).get("severity", 0.0)
        if sev_w <= 0 and mkey != "glucose":
            sev_w = 0.1
        conf = clip01((float(r2_map.get(tgt, r2_map.get(col, 0.0))) - 0.10) / 0.20) if r2_map else 0.6
        if conf <= 0:
            continue

        # impact
        pred = P[col]
        benefit = goal_benefit(pred, meta)
        imp = sev_w * conf * benefit
        impact_total = impact_total.add(pd.Series(imp.values, index=P.index), fill_value=0.0)

        # table for this marker
        dfm = (
            R_all.set_index("FoodCode")
                 .assign(pred=pred, impact=imp)
                 .sort_values("impact", ascending=False)
                 [["Desc", "impact"]]
                 .rename(columns={"impact": "impact_score"})
                 .reset_index()
        )
        dfm["dedup_key"] = dfm["Desc"].map(_normalize_desc)
        dfm = dfm.drop_duplicates("dedup_key", keep="first").drop(columns="dedup_key")
        per_marker_tables[mkey] = dfm
# <— end of loop/block


        def _norm(s: str) -> str:
            return re.sub(r"[^A-Z0-9]+","", str(s).upper())

        col = None
        if tgt in P.columns:
            col = tgt
        else:
            nt = _norm(tgt)
            for c in P.columns:
                if nt in _norm(c):
                    col = c
                    break

        if col is None:
            continue  # no matching model column

        pred = P[col]


        sev_w = sev.get(mkey, {}).get("severity", 0.0)
        if sev_w <= 0 and mkey != "glucose":
            sev_w = 0.1
        conf = clip01((float(r2_map.get(tgt, 0.0)) - 0.10) / 0.20) if r2_map else 0.6
        if conf <= 0:
            continue

        pred = P[tgt]
        benefit = goal_benefit(pred, meta)
        imp = sev_w * conf * benefit

        impact_total = impact_total.add(pd.Series(imp.values, index=P.index), fill_value=0.0)

        dfm = (
            R_all.set_index("FoodCode")
                 .assign(pred=pred, impact=imp)
                 .sort_values("impact", ascending=False)
                 [["Desc","impact"]]
                 .rename(columns={"impact":"impact_score"})
                 .reset_index()
        )
        dfm = _dedupe_by_desc(dfm)
        per_marker_tables[mkey] = dfm

    # 6) blend with BioAge score
    base = pd.to_numeric(R_all["score"], errors="coerce").astype(float)
    base_z = robust_z(base)
    blended = w_bioage * base_z - w_marker * impact_total.reindex(food_index).fillna(0.0).values

    R_all["score_final"] = blended

    # 7) overall table
    top_overall = R_all.sort_values("score_final", ascending=True).head(100).copy()
    st.markdown("**Top 100 overall (lower = better; blended BioAge + Marker)**")
    st.dataframe(
        top_overall[["FoodCode","Desc","kcal_per_100g","score_final","tags","category"]]
                   .rename(columns={"score_final":"score"}),
        use_container_width=True
    )

    # 8) Marker cards (from per-marker impacts, if any)
    st.subheader("Foods by marker (model-targeted)")
    shown_keys = set()     # global de-dupe across cards
    cols = st.columns(2); i = 0

    for mkey, meta in MARKER_MAP.items():
        info  = sev.get(mkey, {})
        val   = info.get("value"); status = info.get("status")
        units = meta["units"]; label = meta["label"]
        glyph = "🔺" if status == "high" else ("🔻" if status == "low" else "✅")

        with cols[i % 2]:
            st.markdown(f"**{label}** — {glyph} {('%.2f' % val) if val is not None else '—'} {units}")
            dfk = per_marker_tables.get(mkey)
            if dfk is None or dfk.empty:
                st.caption("No model or no strong matches for this marker.")
            else:
                dfk = dfk.copy()
                dfk["dedup_key"] = dfk["Desc"].map(_normalize_desc)
                dfk = dfk[~dfk["dedup_key"].isin(shown_keys)]
                dfk = dfk.drop_duplicates("dedup_key", keep="first").drop(columns="dedup_key")

                show = (dfk.sort_values("impact_score", ascending=False)
                          .head(10)[["FoodCode","Desc","impact_score"]]
                          .rename(columns={"impact_score":"impact"}))
                st.dataframe(show, use_container_width=True, hide_index=True)

                shown_keys.update(show["Desc"].map(_normalize_desc))
        i += 1

    # 9) category tabs
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

