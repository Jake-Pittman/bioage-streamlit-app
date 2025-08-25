# bloodfood.py — Hybrid BioAge + Per-Marker ML food recommender
# - Parses labs (PDF/CSV) → computes PhenoAge + per-marker severity
# - Loads per-food LightGBM models (joblib) + scaler → predicts marker deltas per food
# - Blends BioAge score with per-marker impact + dietary preferences

import os, io, re, json, contextlib, glob
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# third-party model utils
from joblib import load as joblib_load

st.set_page_config(page_title="Blood→Food (BioAge + Marker ML)", layout="wide", initial_sidebar_state="expanded")

# ---------------- Paths ----------------
REPO_ROOT = Path(__file__).resolve().parent
env_root  = os.environ.get("BIOAGE_ROOT")
root      = Path(env_root) if env_root else REPO_ROOT

PROC   = root / "processed"
ASSETS = root / "app_assets"
CORE   = root / "models" / "RewardModel" / "core_scoring"

# Catalog + attribution
CAT_PARQUET    = ASSETS / "food_catalog.parquet"
FND_PARQUET    = PROC / "FNDDS_MASTER_PER100G.parquet"   # expected to hold NUTR_* features
ATTR_CSV       = CORE / "core_food_attribution_top50_compact.csv"  # (optional for "why" & prefs nudges)

# Per-food ML models (you can keep your joblibs here)
PERFOOD_DIR    = root / "models" / "PerFood"
PERFOOD_ALT    = root / "models" / "perfood"  # fallback if you name the folder differently

SCALER_FILE    = "X_scaler.joblib"
R2_FILE        = "per_target_r2.csv"
META_FILE      = "meta.json"

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

# ---------------- Loaders ----------------
@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    cat = pd.read_parquet(CAT_PARQUET)
    cat["FoodCode"] = pd.to_numeric(cat["FoodCode"], errors="coerce").astype("Int64")
    if "tags" not in cat.columns: cat["tags"] = np.nan
    return cat

@st.cache_data(show_spinner=False)
def load_fnd_features() -> pd.DataFrame | None:
    """Return a DF with FoodCode + NUTR_* columns (per 100kcal preferred)."""
    if not FND_PARQUET.exists():
        # Try to use catalog if it already contains NUTR_* columns
        if CAT_PARQUET.exists():
            c = pd.read_parquet(CAT_PARQUET)
            nutr_cols = [c for c in c.columns if str(c).startswith("NUTR_")]
            if nutr_cols:
                out = c[["FoodCode"] + nutr_cols].copy()
                out["FoodCode"] = pd.to_numeric(out["FoodCode"], errors="coerce").astype("Int64")
                return out
        return None

    fnd = pd.read_parquet(FND_PARQUET)
    fnd["FoodCode"] = pd.to_numeric(fnd["FoodCode"], errors="coerce").astype("Int64")
    nutr_cols = [c for c in fnd.columns if str(c).startswith("NUTR_")]
    return fnd[["FoodCode"] + nutr_cols].copy()

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
LAB_SCHEMA_FALLBACK = {
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
def load_lab_schema():
    js = ASSETS / "lab_schema.json"
    if js.exists():
        with open(js, "r") as f: return json.load(f)
    return LAB_SCHEMA_FALLBACK

def normalize_labs(df_in: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df_in.copy()
    alias_map = {a.lower(): std for std, meta in schema.items() for a in meta.get("aliases", [])}
    rename = {col: alias_map[col.lower()] for col in df.columns if col.lower() in alias_map}
    if rename: df = df.rename(columns=rename)
    keep = list(schema.keys())
    cols_present = [c for c in keep if c in df.columns]
    out = df[cols_present].copy()
    for c in cols_present: out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def parse_pdf_labs(file_like) -> dict:
    if not HAS_PDFPLUMBER: raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")
    expected_map = {
        "albumin":"albumin", "serum albumin":"albumin", "albumin, serum":"albumin", "alb":"albumin",
        "alkaline phosphatase":"alp","alk phos":"alp","alk. phos":"alp","alp":"alp",
        "c-reactive protein":"crp","hs-crp":"crp","crp":"crp",
        "glucose, fasting":"fasting_glucose","glucose (fasting)":"fasting_glucose","fasting glucose":"fasting_glucose","glucose":"fasting_glucose",
        "wbc":"wbc","white blood cells":"wbc",
        "absolute lymphocytes":"lymphs_abs","abs lymphocytes":"lymphs_abs",
        "lymphocyte %":"lymphs_pct","lymphocytes %":"lymphs_pct","lymphocytes":"lymphs_pct","lymphs":"lymphs_pct",
        "mcv":"mcv","rdw":"rdw",
        "creatinine":"creatinine","creat":"creatinine","bun":"bun"
    }
    range_pat  = re.compile(r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\b")
    number_pat = re.compile(r"(-?\d+(?:\.\d+)?)")
    def first_number(text):
        if not text or range_pat.search(text): return None
        m = number_pat.search(text); return float(m.group(1)) if m else None
    def match_key(lbl_raw):
        s = (lbl_raw or "").strip().lower()
        for syn, std in sorted(expected_map.items(), key=lambda kv: -len(kv[0])):
            if syn in s: return std
        return None
    labs, aux = {}, {}
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            for tbl in (page.extract_tables() or []):
                for row in (tbl or []):
                    if not row: continue
                    key = match_key(row[0]);
                    if not key: continue
                    prefer_pct = (key == "lymphs_pct"); best = None
                    for cell in row[1:]:
                        txt = (cell or ""); val = first_number(txt)
                        if val is None: continue
                        cand = (1 if (prefer_pct and "%" in txt) else 0, -len(txt), val, txt)
                        if (best is None) or (cand > best): best = cand
                    if best is None: continue
                    _, _, val, txt = best; lowtxt = (txt or "").lower()
                    if key == "crp" and ("mg/dl" in lowtxt) and ("mg/l" not in lowtxt): val *= 10.0
                    if key == "albumin" and "g/l" in lowtxt: val /= 10.0
                    if key == "lymphs_abs": aux["lymphs_abs"] = val
                    elif key == "lymphs_pct": labs["lymphs"] = val
                    else: labs[key] = val
    # Text fallback
    file_like.seek(0); full = ""
    with pdfplumber.open(file_like) as pdf:
        for p in pdf.pages: full += (p.extract_text() or "").replace("\n"," ") + " "
    if "alp" not in labs:
        m = re.search(r"(?i)alk[^\n]{0,80}?phos[^\n]{0,80}?(-?\d+(?:\.\d+)?)", full)
        if m:
            with contextlib.suppress(Exception): labs["alp"] = float(m.group(1))
    if "lymphs" not in labs:
        if aux.get("lymphs_pct") is not None: labs["lymphs"] = aux["lymphs_pct"]
        elif aux.get("lymphs_abs") is not None and labs.get("wbc"):
            labs["lymphs_abs"] = aux["lymphs_abs"]
    return labs

def sanitize_labs(labs: dict) -> dict:
    x = dict(labs) if labs else {}
    if "wbc" in x and x["wbc"] is not None:
        with contextlib.suppress(Exception):
            w = float(x["wbc"]); x["wbc"] = w/1000.0 if w>100 else w
    if x.get("lymphs_abs") not in (None, np.nan) and x.get("wbc") not in (None, 0, np.nan):
        with contextlib.suppress(Exception):
            abs_cells = float(x["lymphs_abs"]); wbc_k = float(x["wbc"])
            x["lymphs"] = (abs_cells / (wbc_k*1000.0)) * 100.0
    if x.get("lymphs") not in (None, np.nan):
        with contextlib.suppress(Exception):
            l = float(x["lymphs"])
            if l > 1000 and x.get("wbc"):
                l = (l / (float(x["wbc"])*1000.0))*100.0
            if l > 100: l = l/10.0 if l<=300 else l/100.0
            x["lymphs"] = float(np.clip(l, 1.0, 80.0))
    if x.get("crp") not in (None, np.nan):
        with contextlib.suppress(Exception):
            c = float(x["crp"]);
            if c > 200: c *= 10.0
            x["crp"] = float(np.clip(c, 0.0, 500.0))
    def _clip(k, lo, hi):
        if x.get(k) not in (None, np.nan):
            with contextlib.suppress(Exception): x[k] = float(np.clip(float(x[k]), lo, hi))
    _clip("albumin", 2.0, 6.5); _clip("glucose", 40.0, 500.0); _clip("creatinine", 0.2, 12.0)
    _clip("alp", 10.0, 1500.0); _clip("mcv", 60.0, 130.0); _clip("rdw", 8.0, 30.0); _clip("wbc", 0.1, 50.0)
    return x

def phenoage_from_row(row: pd.Series):
    need = ["age_years","albumin","creatinine","glucose","crp_mgL","lymphocyte_pct","mcv","rdw","alk_phosphatase","wbc"]
    if any(pd.isna(row.get(k)) for k in need): return (None, None)
    albumin_gL   = float(row["albumin"])*10.0
    creat_umol   = float(row["creatinine"])*88.4
    glucose_mmol = float(row["glucose"])/18.0
    lncrp        = np.log(max(float(row["crp_mgL"]),0.0)+1e-6)
    lymph        = float(row["lymphocyte_pct"])
    mcv          = float(row["mcv"]); rdw = float(row["rdw"])
    alp          = float(row["alk_phosphatase"]); wbc = float(row["wbc"]); age = float(row["age_years"])
    xb = (-19.90667 + (-0.03359355*albumin_gL) + (0.009506491*creat_umol) + (0.1953192*glucose_mmol)
          + (0.09536762*lncrp) + (-0.01199984*lymph) + (0.02676401*mcv) + (0.3306156*rdw)
          + (0.001868778*alp) + (0.05542406*wbc) + (0.08035356*age))
    m = 1.0 - np.exp((-1.51714*np.exp(xb))/0.007692696); m = float(np.clip(m, 1e-15, 1-1e-15))
    pheno = (np.log(-0.0055305*np.log(1.0 - m))/0.09165) + 141.50225; accel = pheno - age
    return float(np.clip(pheno,0.0,140.0)), float(np.clip(accel,-60.0,90.0))

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
@st.cache_resource(show_spinner=False)
def load_perfood_bundle():
    """Locate models, scaler, r2, meta; return dict or None if unavailable."""
    # pick a directory that exists
    mdl_dir = PERFOOD_DIR if PERFOOD_DIR.exists() else (PERFOOD_ALT if PERFOOD_ALT.exists() else None)
    if mdl_dir is None:
        # also allow repo root (user sometimes drops joblibs at root)
        if (root / SCALER_FILE).exists(): mdl_dir = root
        else: return None

    bundle = {"dir": mdl_dir}
    # scaler
    sc_path = mdl_dir / SCALER_FILE
    if not sc_path.exists(): return None
    bundle["scaler"] = joblib_load(sc_path)

    # per-target r2 (optional, but recommended)
    r2_path = mdl_dir / R2_FILE
    if r2_path.exists():
        r2 = pd.read_csv(r2_path)
        if {"target","r2"}.issubset(r2.columns):
            bundle["r2_map"] = {str(t).strip(): float(r) for t, r in zip(r2["target"], r2["r2"])}
        else:
            bundle["r2_map"] = {}
    else:
        bundle["r2_map"] = {}

    # meta.json for feature list (optional)
    meta_path = mdl_dir / META_FILE
    feats = None
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
            for k in ["feature_names","X_cols","features"]:
                if k in meta and isinstance(meta[k], list):
                    feats = [str(c) for c in meta[k]]
                    break
    bundle["features"] = feats

    # load all model joblibs that match lgbm_*.joblib
    models = {}
    for p in sorted(glob.glob(str(mdl_dir / "lgbm_*.joblib"))):
        name = Path(p).stem  # e.g., lgbm_LBXSGL
        tgt = name.split("lgbm_")[-1]
        try:
            models[tgt] = joblib_load(p)
        except Exception:
            continue
    bundle["models"] = models
    return bundle

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
fnd_feats= load_fnd_features()

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
st.subheader("Parsed labs & BioAge")
if parsed_df is not None:
    st.dataframe(parsed_df, use_container_width=True)
    bio, accel = phenoage_from_row(parsed_df.iloc[0])
    c1, c2 = st.columns(2)
    with c1: st.metric("BioAge (PhenoAge)", f"{bio:.1f}" if bio is not None else "–")
    with c2: st.metric("BioAgeAccel", f"{accel:+.1f}" if accel is not None else "–")
else:
    st.info("Upload a PDF/CSV, enter age, and press the button.")

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
    P = None
    if bundle is None:
        st.warning("Per-food ML models not found. Showing BioAge-only ranking. Place joblibs + scaler under models/PerFood/.")
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

