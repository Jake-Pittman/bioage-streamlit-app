# app.py — ΔPhenoAge Food Recommender with PDF parsing + robust labs sanitize + personalized recs

import os, re, io, json, contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ΔPhenoAge Food Recommender", layout="wide", initial_sidebar_state="expanded")

# Optional PDF dependency
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

# ---------------- Paths ----------------
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
ATTR_CSV       = CORE / "core_food_attribution_top50_compact.csv"  # (optional for personalization + "why")

# ---------------- Cached loaders ----------------
@st.cache_data(show_spinner=False)
def load_lab_schema():
    if LAB_SCHEMA_JS.exists():
        with open(LAB_SCHEMA_JS, "r") as f:
            return json.load(f)
    # minimal fallback schema (units for display/CSV aliasing)
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
def load_catalog():
    """Catalog with FoodCode, Desc, kcal_per_100g, score (lower=better), tags, per100kcal (optional)."""
    cat = pd.read_parquet(CAT_PARQUET)
    cat["FoodCode"] = pd.to_numeric(cat["FoodCode"], errors="coerce").astype("Int64")

    src = CONSENSUS_CSV if CONSENSUS_CSV.exists() else GUARDRAILS_CSV
    if src.exists():
        r = pd.read_csv(src)
        r["FoodCode"] = pd.to_numeric(r["FoodCode"], errors="coerce").astype("Int64")
        percol = "guarded_score_per_100kcal" if "guarded_score_per_100kcal" in r.columns else \
                 ("core_score_per_100kcal" if "core_score_per_100kcal" in r.columns else None)
        if percol:
            cat = cat.merge(
                r[["FoodCode", percol]].rename(columns={percol: "per100kcal"}),
                on="FoodCode", how="left"
            )
        else:
            cat["per100kcal"] = np.nan
    else:
        cat["per100kcal"] = np.nan

    if "tags" not in cat.columns:
        cat["tags"] = np.nan
    return cat

# ---------------- CSV normalization (CSV fallback) ----------------
def normalize_labs(df_in: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df_in.copy()
    alias_map = {a.lower(): std for std, meta in schema.items() for a in meta.get("aliases", [])}
    rename = {col: alias_map[col.lower()] for col in df.columns if col.lower() in alias_map}
    if rename:
        df = df.rename(columns=rename)
    keep = list(schema.keys())
    cols_present = [c for c in keep if c in df.columns]
    out = df[cols_present].copy()
    for c in cols_present:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# ---------------- PDF parser (table-first + text fallback) ----------------
def parse_pdf_labs(file_like) -> dict:
    if not HAS_PDFPLUMBER:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")

    expected_map = {
        # Key markers
        "albumin":"albumin","serum albumin":"albumin","albumin, serum":"albumin","alb":"albumin",
        # ALP
        "alkaline phosphatase":"alp","alkaline phosphatase (total)":"alp",
        "alkaline phosphatase, total":"alp","alkaline phosphatase, serum":"alp",
        "alkaline phosphatase (alp)":"alp","alkaline phosphatase (alk phos)":"alp",
        "alk phos":"alp","alk. phos":"alp","alk. phosphatase":"alp","alk-phos":"alp",
        "alk phosphatase":"alp","alk-phosphatase":"alp","alp":"alp",
        # CRP
        "c-reactive protein":"crp","c reactive protein":"crp","hs-crp":"crp","hscrp":"crp","crp":"crp",
        # Glucose
        "glucose, fasting":"fasting_glucose","glucose (fasting)":"fasting_glucose",
        "fasting glucose":"fasting_glucose","glucose":"fasting_glucose",
        # WBC
        "wbc":"wbc","white blood cell":"wbc","white blood cells":"wbc",
        # Lymphs (prefer absolute keys before percent)
        "absolute lymphocytes":"lymphs_abs","abs lymphocytes":"lymphs_abs",
        "lymphocytes absolute":"lymphs_abs","absolute lymphocyte":"lymphs_abs","abs lymphs":"lymphs_abs",
        "lymphocyte %":"lymphs_pct","lymphocytes %":"lymphs_pct","lymphocyte percent":"lymphs_pct",
        "lymphocytes":"lymphs_pct","lymphs":"lymphs_pct",
        # RBC indices
        "mcv":"mcv","mean corpuscular volume":"mcv",
        "rdw":"rdw","red cell distribution width":"rdw",
        # extras
        "creatinine":"creatinine","creat":"creatinine",
        "bun":"bun","blood urea nitrogen":"bun",
    }

    range_pat  = re.compile(r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\b")
    number_pat = re.compile(r"(-?\d+(?:\.\d+)?)")

    def cell_has_range(text): return bool(range_pat.search(text or ""))

    def first_number(text):
        if not text: return None
        if cell_has_range(text): return None
        m = number_pat.search(text)
        return float(m.group(1)) if m else None

    def match_key(lbl_raw):
        s = (lbl_raw or "").strip().lower()
        for syn, std in sorted(expected_map.items(), key=lambda kv: -len(kv[0])):  # longest first
            if syn in s:
                return std
        return None

    labs, aux = {}, {}

    # Table-first
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            for tbl in (page.extract_tables() or []):
                for row in (tbl or []):
                    if not row:
                        continue
                    key = match_key(row[0])
                    if not key:
                        continue
                    # choose a numeric from remaining cells; for lymphs % prefer a '%' cell
                    prefer_pct = (key == "lymphs_pct")
                    best = None
                    for cell in row[1:]:
                        txt = (cell or "")
                        val = first_number(txt)
                        if val is None:
                            continue
                        cand = (1 if (prefer_pct and "%" in txt) else 0, -len(txt), val, txt)
                        if (best is None) or (cand > best):
                            best = cand
                    if best is None:
                        continue
                    _, _, val, txt = best
                    lowtxt = (txt or "").lower()
                    if key == "crp" and ("mg/dl" in lowtxt) and ("mg/l" not in lowtxt):
                        val *= 10.0
                    if key == "albumin" and "g/l" in lowtxt:
                        val /= 10.0
                    if key == "lymphs_abs":
                        aux["lymphs_abs"] = val
                    elif key == "lymphs_pct":
                        labs["lymphs"] = val
                    else:
                        labs[key] = val

    # Text fallback
    file_like.seek(0)
    full = ""
    with pdfplumber.open(file_like) as pdf:
        for p in pdf.pages:
            full += (p.extract_text() or "").replace("\n", " ") + " "

    BIG = 800
    def from_text(label, std_key):
        if std_key in labs:
            return
        pat = re.compile(rf"(?i)\b{re.escape(label)}\b" + rf".{{0,{BIG}}}?" + r"(-?\d+(?:\.\d+)?)")
        m = pat.search(full)
        if not m:
            return
        val = float(m.group(1))
        window = full[max(0, m.start()-40): m.end()+40].lower()
        if std_key == "crp" and ("mg/dl" in window) and ("mg/l" not in window):
            val *= 10.0
        if std_key == "albumin" and "g/l" in window:
            val /= 10.0
        if std_key in ("lymphs_pct", "lymphs_abs"):
            aux[std_key] = val
        else:
            labs[std_key] = val

    for label, key in expected_map.items():
        from_text(label, key)

    # Extra ALP safety net
    if "alp" not in labs:
        m = re.search(r"(?i)alk[^\n]{0,80}?phos[^\n]{0,80}?(-?\d+(?:\.\d+)?)", full)
        if m:
            with contextlib.suppress(Exception):
                labs["alp"] = float(m.group(1))

    # If needed, derive lymphs % from absolute (cells/µL) and WBC (10^3/µL)
    if "lymphs" not in labs:
        if aux.get("lymphs_pct") is not None:
            labs["lymphs"] = aux["lymphs_pct"]
        elif aux.get("lymphs_abs") is not None and labs.get("wbc"):
            # WBC might be reported as cells/µL or 10^3/µL — normalize later in sanitize()
            labs["lymphs_abs"] = aux["lymphs_abs"]  # keep; we'll convert in sanitize
    return labs

# ---------------- Labs sanitize (fix units / impossible values) ----------------
def sanitize_labs(labs: dict) -> dict:
    """Coerce labs to PhenoAge units and plausible ranges. Fix common PDF quirks."""
    x = dict(labs) if labs else {}

    # WBC: if reported as cells/µL (e.g., 5400), convert to 10^3/µL → 5.4
    if "wbc" in x and x["wbc"] is not None:
        try:
            w = float(x["wbc"])
            if w > 100:   # e.g., 5400 cells/µL
                w = w / 1000.0
            x["wbc"] = w
        except Exception:
            pass

    # Lymphocytes:
    # 1) If we have absolute + WBC, compute percent
    if x.get("lymphs_abs") not in (None, np.nan) and x.get("wbc") not in (None, 0, np.nan):
        try:
            abs_cells = float(x["lymphs_abs"])        # cells/µL
            wbc_k    = float(x["wbc"])               # 10^3/µL
            pct = (abs_cells / (wbc_k * 1000.0)) * 100.0
            x["lymphs"] = pct
        except Exception:
            pass

    # 2) If lymphs looks like absolute or an off-by-10/100 percent, fix it
    if x.get("lymphs") not in (None, np.nan):
        try:
            l = float(x["lymphs"])
            # absolute (cells/µL) mistakenly captured
            if l > 1000 and x.get("wbc"):
                l = (l / (float(x["wbc"]) * 1000.0)) * 100.0
            # bad percent (e.g., 175.3) → maybe 17.53
            if l > 100:
                l = l / 10.0 if l <= 300 else l / 100.0
            # clamp to plausible range
            l = float(np.clip(l, 1.0, 80.0))
            x["lymphs"] = l
        except Exception:
            pass

    # CRP: mg/L; if implausibly large (>200), assume mg/dL supplied and convert
    if x.get("crp") not in (None, np.nan):
        try:
            c = float(x["crp"])
            if c > 200:
                c = c * 10.0  # if mg/dL mistakenly parsed without unit, bring to mg/L
            x["crp"] = float(np.clip(c, 0.0, 500.0))
        except Exception:
            pass

    # Albumin g/dL typical 3–6; Glucose mg/dL typical 50–400; Creatinine mg/dL 0.2–10; ALP U/L 10–1000; MCV fL 60–130; RDW % 8–25
    def _clip(k, lo, hi):
        if x.get(k) not in (None, np.nan):
            with contextlib.suppress(Exception):
                x[k] = float(np.clip(float(x[k]), lo, hi))
    _clip("albumin", 2.0, 6.5)
    _clip("glucose", 40.0, 500.0)
    _clip("creatinine", 0.2, 12.0)
    _clip("alp", 10.0, 1500.0)
    _clip("mcv", 60.0, 130.0)
    _clip("rdw", 8.0, 30.0)
    _clip("wbc", 0.1, 50.0)

    return x

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

# ---------------- Dedup + categories ----------------
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

    def row_has_token(text: str, token: str) -> bool:
        return token in (text or "").lower()

    def desc_has_token(desc: str, token: str) -> bool:
        return token in str(desc or "").lower()

    if include_tags.strip():
        raw = [t.strip().lower() for t in include_tags.split(",") if t.strip()]
        inc = []
        for t in raw:
            inc.extend(_DISLIKE_SYNONYMS.get(t, [t]))
        R = R[R.apply(lambda r: any(row_has_token(r.get("tags",""), t) or
                                    desc_has_token(r.get("Desc",""), t)
                                    for t in inc), axis=1)]
    if exclude_tags.strip():
        raw = [t.strip().lower() for t in exclude_tags.split(",") if t.strip()]
        exc = []
        for t in raw:
            exc.extend(_DISLIKE_SYNONYMS.get(t, [t]))
        R = R[~R.apply(lambda r: any(row_has_token(r.get("tags",""), t) or
                                     desc_has_token(r.get("Desc",""), t)
                                     for t in exc), axis=1)]
    R["dedup_key"] = R["Desc"].map(_normalize_desc)
    R = R.sort_values(["score", "kcal_per_100g"], ascending=[True, True])
    R = R.drop_duplicates(subset="dedup_key", keep="first")
    return R.reset_index(drop=True)

# ---------------- Attribution + personalization ----------------
_PRETTY = {
    "KCAL":"Calories","CARB":"Carbohydrates","PROT":"Protein","TFAT":"Total fat","SFAT":"Saturated fat",
    "PFAT":"Polyunsat. fat","MFAT":"Monounsat. fat","SUGR":"Sugars","FIBE":"Fiber","CALC":"Calcium",
    "PHOS":"Phosphorus","MAGN":"Magnesium","POTA":"Potassium","ZINC":"Zinc","SELE":"Selenium","COPP":"Copper",
    "ATOC":"Vitamin E","VARA":"Vitamin A (RAE)","VK":"Vitamin K","VC":"Vitamin C","VB1":"B1","VB2":"B2",
    "VB6":"B6","VB12":"B12","NIAC":"Niacin","CAFF":"Caffeine","FDFE":"Iron",
}

def _attr_tokens(s: str) -> set[str]:
    if not isinstance(s, str) or not s.strip():
        return set()
    toks = set()
    for chunk in s.split(";"):
        m = re.search(r"(?:NUTR_)?([A-Z0-9]+)\s*\(([-+]?[\d\.]+)", chunk.strip())
        if m:
            toks.add(m.group(1))
    return toks
    
def _attr_weights(s: str) -> dict[str, float]:
    """
    Parse 'NUTR_CODE(value);...' into {code: abs(value)} so we can weight strengths.
    """
    out = {}
    if not isinstance(s, str) or not s.strip():
        return out
    for chunk in s.split(";"):
        m = re.search(r"(?:NUTR_)?([A-Z0-9]+)\s*\(([-+]?[\d\.]+)", chunk.strip())
        if m:
            out[m.group(1)] = abs(float(m.group(2)))
    return out


def _lab_needs_from_row(row: pd.Series) -> dict[str, set[str]]:
    """Rough, conservative mapping from labs → helpful nutrients / watch-outs."""
    need = {"benefit": set(), "avoid": set()}
    try:
        if float(row.get("glucose", np.nan)) >= 100:
            need["benefit"] |= {"FIBE","PROT","PFAT","MFAT","VK","VC"}
            need["avoid"]   |= {"SUGR","CARB"}
        if float(row.get("crp_mgL", np.nan)) > 3:
            need["benefit"] |= {"VC","ATOC","SELE","ZINC","FIBE","PFAT"}
            need["avoid"]   |= {"SFAT"}
        if float(row.get("lymphocyte_pct", np.nan)) < 20:
            need["benefit"] |= {"VC","ZINC","SELE"}
        if float(row.get("albumin", np.nan)) < 3.8:
            need["benefit"] |= {"PROT","ZINC","SELE"}
        if float(row.get("rdw", np.nan)) > 14.5:
            need["benefit"] |= {"FDFE","VB12","VB6"}
        if float(row.get("alk_phosphatase", np.nan)) > 120:
            need["benefit"] |= {"VK","VARA","VC"}
        if float(row.get("wbc", np.nan)) > 10:
            need["benefit"] |= {"VC","ATOC","SELE","ZINC"}
    except Exception:
        pass
    return need

def personalize_scores(df: pd.DataFrame, labs_row: pd.Series | None, attr_csv: Path,
                       base_col="score", out_col="score_labs"):
    """Adjust scores by labs-driven needs using attribution tokens."""
    df = df.copy()
    df[out_col] = df[base_col].astype(float)
    if labs_row is None or not attr_csv.exists():
        return df

    need = _lab_needs_from_row(labs_row)
    try:
        A = pd.read_csv(attr_csv, usecols=["FoodCode","top_negative_terms","top_positive_terms"])
    except Exception:
        return df

    A["FoodCode"] = pd.to_numeric(A["FoodCode"], errors="coerce").astype("Int64")
    A["_neg"] = A["top_negative_terms"].apply(_attr_tokens)  # helpful nutrients
    A["_pos"] = A["top_positive_terms"].apply(_attr_tokens)  # watch-outs

    M = df.merge(A[["FoodCode","_neg","_pos"]], on="FoodCode", how="left")

    # Scale adjustments to the spread of the base scores (so it actually moves ranks)
    base = pd.to_numeric(M[base_col], errors="coerce").astype(float)
    span = np.nanpercentile(base, 95) - np.nanpercentile(base, 5)
    STEP = 0.05 * (span if np.isfinite(span) and span > 0 else 100.0)  # ~5% of spread

    def _as_set(x):
        return x if isinstance(x, set) else set()

    adj = []
    for _, r in M.iterrows():
        neg = _as_set(r.get("_neg"))
        pos = _as_set(r.get("_pos"))
        b = -1.0 * STEP * len(neg & need["benefit"])   # helpful nutrients → better
        p = +1.0 * STEP * len(pos & need["avoid"])     # watch-outs → worse
        adj.append(b + p)

    M[out_col] = base + pd.Series(adj, index=M.index)
    return M.drop(columns=["_neg","_pos"], errors="ignore")

def _parse_terms(term_str: str, top=3):
    if not isinstance(term_str, str) or not term_str.strip():
        return []
    items = []
    for chunk in term_str.split(";"):
        m = re.search(r"(?:NUTR_)?([A-Z0-9]+)\(([-+]?[\d\.]+)", chunk.strip())
        if not m: continue
        code = m.group(1); val = abs(float(m.group(2)))
        items.append((code, val))
    items.sort(key=lambda x: x[1], reverse=True)
    return [_PRETTY.get(code, code) for code, _ in items[:top]]

def build_why_table(ranked_df: pd.DataFrame, attr_csv: Path, top_n=15):
    if not attr_csv.exists():
        return None
    try:
        attr = pd.read_csv(attr_csv)
    except Exception:
        return None
    need_cols = {"FoodCode","Desc","core_score_per_100kcal","top_negative_terms","top_positive_terms"}
    if not need_cols.issubset(attr.columns):
        return None
    J = ranked_df[["FoodCode","Desc","kcal_per_100g","score_use"]].merge(
        attr[list(need_cols)], on=["FoodCode","Desc"], how="left"
    ).head(top_n)
    rows = []
    for _, r in J.iterrows():
        helps = ", ".join(_parse_terms(r.get("top_negative_terms",""), top=3)) or "—"
        watch = ", ".join(_parse_terms(r.get("top_positive_terms",""), top=2)) or "—"
        rows.append({
            "Food": r["Desc"],
            "Why it helps (top nutrients)": helps,
            "Potential watch-outs": watch,
            "Personalized score (↓ better)": (None if pd.isna(r.get("score_use")) else round(float(r["score_use"]), 3)),
        })
    return pd.DataFrame(rows)

# ---------- Preference adjustments (diet/exclusions/dislikes/soft-goals) ----------
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
    # Mediterranean/DASH/Keto-lite handled as soft-goal nudges only
}

_DISLIKE_SYNONYMS = {
    "fish": [
        "fish","salmon","tuna","sardine","sardines","anchovy","anchovies",
        "mackerel","herring","trout","cod","halibut","tilapia","seafood"
    ]
}

def apply_preferences(df: pd.DataFrame,
                      diet_pattern: str,
                      exclusions: list[str],
                      dislikes: str,
                      attr_csv: Path,
                      base_col="score_labs",
                      out_col="score_use") -> pd.DataFrame:
    """
    - Hard filters: diet_pattern (if Vegan/Vegetarian/Pescatarian) + explicit exclusions + dislikes text.
    - Soft nudges: sugar/sodium/protein via attribution tokens (added outside via weights).
    """
    R = df.copy()
    # Hard filters
    desc = R["Desc"].fillna("").astype(str)

    # Diet pattern blocks (only if pattern has hard rules)
    block_re = _DIET_BLOCK.get(diet_pattern)
    if block_re is not None:
        R = R[~desc.str.contains(block_re)]

    # Exclusions
    for ex in exclusions:
        ex_re = _EXCL_PATTERNS.get(ex)
        if ex_re is not None:
            R = R[~desc.str.contains(ex_re)]

    # Dislikes
    if dislikes.strip():
        raw = [x.strip().lower() for x in dislikes.split(",") if x.strip()]
        tokens = []
        for t in raw:
            tokens.extend(_DISLIKE_SYNONYMS.get(t, [t]))
        if tokens:
            bads = [re.escape(x) for x in tokens]
            bad_re = re.compile(r"(?:" + "|".join(bads) + r")", re.I)
            R = R[~desc.str.contains(bad_re)]

    # Initialize the output score
    R[out_col] = R[base_col].astype(float)

    # Soft goals via attribution tokens (if available)
    if attr_csv.exists():
        with contextlib.suppress(Exception):
            A = pd.read_csv(attr_csv, usecols=["FoodCode","top_negative_terms","top_positive_terms"])
            A["FoodCode"] = pd.to_numeric(A["FoodCode"], errors="coerce").astype("Int64")
            A["_neg"] = A["top_negative_terms"].map(_attr_tokens)   # helpful nutrients present
            A["_pos"] = A["top_positive_terms"].map(_attr_tokens)   # potential watch-outs
            R = R.merge(A[["FoodCode","_neg","_pos"]], on="FoodCode", how="left")
            R["_neg"] = R["_neg"].apply(lambda x: x if isinstance(x, set) else set())
            R["_pos"] = R["_pos"].apply(lambda x: x if isinstance(x, set) else set())

            # weights (tiny nudges; tune freely)
            # - sugar penalty if SUGR appears in potential watch-outs
            R[out_col] += 0.25 * g_min_sugar  * R["_pos"].apply(lambda s: 1 if "SUGR" in s else 0)
            # - sodium penalty if SODI appears in potential watch-outs (if present in your tokens)
            R[out_col] += 0.20 * g_min_sodium * R["_pos"].apply(lambda s: 1 if "SODI" in s else 0)
            # + protein bonus if PROT appears in helpful terms
            R[out_col] += (-0.20) * g_pref_protein * R["_neg"].apply(lambda s: 1 if "PROT" in s else 0)

            R.drop(columns=["_neg","_pos"], inplace=True, errors="ignore")

    return R

# ---------- Marker targets & severity ----------
MARKER_MAP = {
    "glucose": {
        "label": "Glucose", "units": "mg/dL", "dir": "high", "goal": (70, 99),
        "helps": {"FIBE", "PROT", "PFAT", "MFAT", "VK", "VC"},
        "avoid": {"SUGR"}
    },
    "crp_mgL": {
        "label": "CRP (hs)", "units": "mg/L", "dir": "high", "goal": (0.0, 3.0),
        "helps": {"VC", "ATOC", "SELE", "ZINC", "FIBE", "PFAT"},
        "avoid": {"SFAT"}
    },
    "albumin": {
        "label": "Albumin", "units": "g/dL", "dir": "low", "goal": (3.8, 5.0),
        "helps": {"PROT", "ZINC", "SELE"},
        "avoid": set()
    },
    "lymphocyte_pct": {
        "label": "Lymphocytes", "units": "%", "dir": "low", "goal": (20, 40),
        "helps": {"VC", "ZINC", "SELE"},
        "avoid": set()
    },
    "rdw": {
        "label": "RDW", "units": "%", "dir": "high", "goal": (11.5, 14.5),
        "helps": {"FDFE", "VB12", "VB6"},
        "avoid": set()
    },
    "alk_phosphatase": {
        "label": "Alk Phosphatase", "units": "U/L", "dir": "high", "goal": (44, 120),
        "helps": {"VK", "VARA", "VC"},
        "avoid": set()
    },
    "wbc": {
        "label": "WBC", "units": "10^3/µL", "dir": "high", "goal": (4.0, 10.5),
        "helps": {"VC", "ATOC", "SELE", "ZINC"},
        "avoid": set()
    },
    "creatinine": {
        "label": "Creatinine", "units": "mg/dL", "dir": "high", "goal": (0.6, 1.3),
        "helps": {"FIBE", "PFAT", "MFAT"},
        "avoid": set()
    },
    "mcv": {
        "label": "MCV", "units": "fL", "dir": "high", "goal": (80, 100),
        "helps": {"VB12", "FDFE"},
        "avoid": set()
    },
}

def _status_and_severity(value: float | None, goal: tuple[float, float] | None, direc: str) -> tuple[str, float]:
    """
    Returns (status, severity in [0,1]).
    status: 'high', 'low', or 'ok' relative to goal and direction of risk.
    Severity ramps from 0 at the edge of goal to 1 at a far-out value.
    """
    if value is None or goal is None:
        return ("ok", 0.0)
    lo, hi = goal
    if direc == "high":
        if value <= hi: return ("ok", 0.0)
        sev = (value - hi) / max(1.0, hi)
        return ("high", float(np.clip(sev, 0.0, 1.0)))
    elif direc == "low":
        if value >= lo: return ("ok", 0.0)
        sev = (lo - value) / max(1.0, lo if lo else 1.0)
        return ("low", float(np.clip(sev, 0.0, 1.0)))
    else:
        return ("ok", 0.0)

def compute_marker_severity(labs_row: pd.Series) -> dict:
    """
    For each marker in MARKER_MAP, compute:
      - value, status ('high'/'low'/'ok'), severity [0,1].
    """
    out = {}
    for key, meta in MARKER_MAP.items():
        val = labs_row.get(key)
        try:
            val = None if pd.isna(val) else float(val)
        except Exception:
            val = None
        status, sev = _status_and_severity(val, meta.get("goal"), meta.get("dir", "ok"))
        out[key] = {
            "value": val,
            "status": status,
            "severity": sev,
            "label": meta["label"],
            "units": meta["units"]
        }
    return out
# Marker-specific nutrient priorities (positive helps, negative = avoid)
MARKER_WEIGHTS = {
    "glucose":       {"FIBE": 1.0, "PROT": 0.8, "PFAT": 0.4, "MFAT": 0.4, "VK": 0.2, "VC": 0.2, "SUGR": -1.0},
    "crp_mgL":       {"VC": 0.8, "SELE": 0.7, "ZINC": 0.7, "ATOC": 0.6, "FIBE": 0.3, "PFAT": 0.3, "SFAT": -0.8},
    "albumin":       {"PROT": 1.0, "ZINC": 0.4, "SELE": 0.4},
    "lymphocyte_pct":{"VC": 0.9, "ZINC": 0.7, "SELE": 0.7},
    "rdw":           {"FDFE": 0.9, "VB12": 0.7, "VB6": 0.4},
    "alk_phosphatase":{"VK": 0.8, "VARA": 0.5, "VC": 0.3},
    "wbc":           {"VC": 0.8, "SELE": 0.6, "ZINC": 0.6, "ATOC": 0.5},
    "creatinine":    {"FIBE": 0.5, "PFAT": 0.4, "MFAT": 0.4},
    "mcv":           {"VB12": 0.9, "FDFE": 0.7},
}

def foods_by_marker(R: pd.DataFrame,
                    attr_csv: Path,
                    marker_sev: dict,
                    top_k: int = 10) -> dict[str, pd.DataFrame]:
    """
    Use weighted nutrient attribution per marker:
      impact = severity * ( Σ w_marker[n]*neg_weight[n]  -  0.6*Σ |w_marker[m]|*pos_weight[m] )
    where:
      - neg_weight = weight from 'top_negative_terms' (nutrients that help the core score)
      - pos_weight = weight from 'top_positive_terms' (watch-outs)
      - w_marker   = marker-specific priority (MARKER_WEIGHTS)
    Then add a small global desirability bonus from score_use.
    """
    out = {}
    if R is None or R.empty or not attr_csv.exists():
        return out

    try:
        A = pd.read_csv(attr_csv, usecols=["FoodCode","Desc","top_negative_terms","top_positive_terms"])
    except Exception:
        return out

    A["FoodCode"] = pd.to_numeric(A["FoodCode"], errors="coerce").astype("Int64")

    # Parse to dicts of weights instead of sets
    A["_negw"] = A["top_negative_terms"].map(_attr_weights)   # helpful nutrients w/ strength
    A["_posw"] = A["top_positive_terms"].map(_attr_weights)   # watch-outs w/ strength

    J = R.merge(A[["FoodCode","_negw","_posw"]], on="FoodCode", how="left")
    J["_negw"] = J["_negw"].apply(lambda d: d if isinstance(d, dict) else {})
    J["_posw"] = J["_posw"].apply(lambda d: d if isinstance(d, dict) else {})

    # Precompute small global bonus from score_use (keep it small so marker dominates)
    su = J["score_use"].astype(float)
    su_rank = su.rank(method="average", ascending=True)
    inv_bonus = (su_rank.max() - su_rank + 1) / su_rank.max()  # ~1 best, ~0 worst
    global_bonus = 0.15 * inv_bonus  # was 0.3 before; make it smaller

    rows_by_marker = {}

    for key, meta in MARKER_MAP.items():
        sev = marker_sev.get(key, {}).get("severity", 0.0)
        if sev <= 0:
            sev = 0.10  # tiny list even if marker okay

        # choose nutrient priorities
        wmark = MARKER_WEIGHTS.get(key, {})
        helps = meta.get("helps", set())
        avoid = meta.get("avoid", set())

        # If MARKER_WEIGHTS not provided for some tokens, fall back to +0.3 for helps / -0.5 for avoids
        def w_for(code: str) -> float:
            if code in wmark:
                return float(wmark[code])
            if code in helps:
                return 0.30
            if code in avoid:
                return -0.50
            return 0.0

        help_scores = []
        avoid_scores = []

        for negw, posw in zip(J["_negw"], J["_posw"]):
            # weighted sum of matching tokens
            h = sum(w_for(tok) * float(negw.get(tok, 0.0)) for tok in set(list(negw.keys()) + list(wmark.keys())))
            a = sum(abs(w_for(tok)) * float(posw.get(tok, 0.0)) for tok in set(list(posw.keys()) + list(wmark.keys())))
            help_scores.append(h)
            avoid_scores.append(a)

        help_scores = pd.Series(help_scores, index=J.index).astype(float)
        avoid_scores = pd.Series(avoid_scores, index=J.index).astype(float)

        impact = sev * (help_scores - 0.6 * avoid_scores)

        total = 0.85 * impact + global_bonus  # marker dominates

        dfk = J.assign(
            impact_score=total,
            why_help=J["_negw"].apply(
                lambda d: ", ".join(
                    _PRETTY.get(k, k)
                    for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if w_for(k) > 0
                )[:100] if d else "—"
            ),
            why_watch=J["_posw"].apply(
                lambda d: ", ".join(
                    _PRETTY.get(k, k)
                    for k, v in sorted(d.items(), key=lambda kv: -kv[1]) if w_for(k) < 0
                )[:100] if d else "—"
            )
        ).sort_values("impact_score", ascending=False)

        rows_by_marker[key] = dfk[["FoodCode","Desc","score_use","impact_score","why_help","why_watch"]].head(top_k)

    return rows_by_marker


def render_category_tabs(R_all: pd.DataFrame, top_n_show: int):
    """Render the 6 category tabs using the personalized 'score_use' column."""
    tabs = st.tabs(["Protein", "Fats", "Fruit", "Vegetables", "Legume/Grains", "Other"])
    CAT_ORDER = ["protein","fats","fruit","vegetables","legume_grains","other"]
    QUOTA = {"protein": 5, "fats": 5, "fruit": 7, "vegetables": 10, "legume_grains": 4, "other": top_n_show}

    for tab, cat in zip(tabs, CAT_ORDER):
        with tab:
            sub = (
                R_all[R_all["category"] == cat]
                .sort_values("score_use")
                .head(QUOTA.get(cat, top_n_show))
            )
            if sub.empty:
                st.info(f"No foods found for category: {cat.replace('_','/')}")
            else:
                st.write(f"**Top {len(sub)} — {cat.replace('_','/').title()}**")
                st.dataframe(
                    sub[["FoodCode","Desc","kcal_per_100g","score_use","tags"]]
                       .rename(columns={"score_use":"score"}),
                    use_container_width=True
                )
                st.download_button(
                    f"Download {cat.replace('_','/').title()} (CSV)",
                    sub.to_csv(index=False).encode("utf-8"),
                    file_name=f"top_{cat}.csv",
                    key=f"dl_{cat}"
                )

def render_marker_cards(R_all: pd.DataFrame, labs_row: pd.Series):
    """Marker cards section."""
    st.subheader("Foods by marker (personalized)")
    marker_sev = compute_marker_severity(labs_row)
    by_marker = foods_by_marker(R_all, ATTR_CSV, marker_sev, top_k=10)

    cols = st.columns(2)
    i = 0
    for key, meta in MARKER_MAP.items():
        info = marker_sev.get(key, {})
        val = info.get("value")
        status = info.get("status")
        units = meta["units"]
        label = meta["label"]

        # simple status glyph
        color = "✅"
        if status == "high": color = "🔺"
        if status == "low":  color = "🔻"

        with cols[i % 2]:
            # keep simple container for wide compatibility
            with st.container():
                st.markdown(f"**{label}** — {color} {('%.2f' % val) if val is not None else '—'} {units}")
                dfk = by_marker.get(key)
                if dfk is None or dfk.empty:
                    st.caption("No strong matches — try relaxing filters.")
                else:
                    show = dfk[["FoodCode","Desc","score_use","impact_score","why_help","why_watch"]].copy()
                    show = show.rename(columns={"score_use":"score", "impact_score":"impact"})
                    st.dataframe(show, use_container_width=True, hide_index=True)
        i += 1

# ============================ UI ============================
st.title("ΔPhenoAge Food Recommender")

# ===== Sidebar =====
with st.sidebar:
    st.markdown("**Paths**")
    st.text(f"Root: {root}")
    if not CAT_PARQUET.exists():
        st.error("Missing app assets. Please generate:\n- app_assets/food_catalog.parquet")

    # Filters
    include_tags = st.text_input("Include tags (comma-separated)", value="")
    exclude_tags = st.text_input("Exclude tags (comma-separated)", value="")

    # Preferences
    st.markdown("**Preferences**")
    diet_pattern = st.selectbox(
        "Diet pattern",
        ["Omnivore", "Pescatarian", "Vegetarian", "Vegan", "Mediterranean", "DASH", "Keto-lite"],
        index=0
    )
    exclusions = st.multiselect(
        "Hard exclusions",
        ["dairy-free", "gluten-free", "nut-free", "shellfish-free", "egg-free", "soy-free", "pork-free"],
        default=[]
    )
    dislikes = st.text_input("Avoid ingredients (comma-separated)", value="")

    st.caption("Soft goals (tune ranking, not a hard filter)")
    g_min_sugar    = st.slider("Minimize added sugar",        0.0, 1.0, 0.3, 0.1)
    g_min_sodium   = st.slider("Minimize sodium",             0.0, 1.0, 0.0, 0.1)
    g_pref_protein = st.slider("Prefer higher-protein foods", 0.0, 1.0, 0.3, 0.1)

    top_n_show = st.slider("How many foods to show in table views", 20, 200, 100, step=10)

    st.markdown("---")
    if TEMPLATE_CSV.exists():
        st.download_button(
            "Download labs CSV template",
            TEMPLATE_CSV.read_bytes(),
            file_name="labs_upload_template.csv"
        )
    if not HAS_PDFPLUMBER:
        st.info("PDF parsing needs `pdfplumber` → `pip install pdfplumber`")

    schema  = load_lab_schema()
    catalog = load_catalog() if CAT_PARQUET.exists() else None

    # -------- 1) Upload labs (PDF or CSV) + age --------
    st.subheader("1) Upload your blood test (PDF preferred) and enter your age")
    c_up1, c_up2 = st.columns(2)
    with c_up1:
        pdf_file = st.file_uploader("Upload PDF blood test", type=["pdf"])
    with c_up2:
        csv_file = st.file_uploader("Or upload CSV (headers may use aliases)", type=["csv"])
    age_input = st.number_input("Your age (years)", min_value=0, max_value=120, value=45)

    run_clicked = st.button("Run model on uploaded labs → compute BioAge & recommend foods")

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

# -------- 2) Show normalized labs & BioAge --------
st.subheader("2) Parsed labs & BioAge")
if parsed_df is not None:
    st.caption("Parsed (normalized) labs used for scoring:")
    st.dataframe(parsed_df, use_container_width=True)
    bio, accel = phenoage_from_row(parsed_df.iloc[0])
    c1, c2 = st.columns(2)
    with c1: st.metric("BioAge (PhenoAge)", f"{bio:.1f}" if bio is not None else "–")
    with c2: st.metric("BioAgeAccel", f"{accel:+.1f}" if accel is not None else "–")
else:
    st.info("Upload & click the button to parse and compute BioAge.")

# -------- 3) Food recommendations (personalized) --------
st.subheader("3) Food recommendations")

if parsed_df is None:
    st.info("Upload a PDF/CSV, enter age, then click the button to get recommendations.")
elif catalog is None:
    st.error("Food catalog not found. Re-run your data prep to create app_assets/food_catalog.parquet.")
else:
    # 1) Filter + de-duplicate
    R_all = dedup_rank(catalog, include_tags, exclude_tags)
    R_all["category"] = [coarse_category(d, t) for d, t in zip(R_all["Desc"], R_all["tags"])]

    # 2) Personalize by labs -> score_labs
    R_all = personalize_scores(
        R_all, parsed_df.iloc[0], ATTR_CSV,
        base_col="score", out_col="score_labs"
    )

    # 3) Apply preferences -> score_use
    R_all = apply_preferences(
        R_all, diet_pattern, exclusions, dislikes, ATTR_CSV,
        base_col="score_labs", out_col="score_use"
    )

    # 4) Sort and show
    R_all = R_all.sort_values("score_use", ascending=True).reset_index(drop=True)

    top_overall = R_all.head(100).copy()
    st.markdown("**Top 100 overall (personalized — lower is better)**")
    st.dataframe(
        top_overall[["FoodCode","Desc","kcal_per_100g","score_use","tags","category"]]
                   .rename(columns={"score_use":"score"}),
        use_container_width=True
    )

    # Category tabs
    render_category_tabs(R_all, top_n_show)

    # Marker cards
    render_marker_cards(R_all, parsed_df.iloc[0])

    # Why these foods?
    why = build_why_table(top_overall, ATTR_CSV, top_n=15)
    if why is not None and len(why):
        st.subheader("Why these foods?")
        st.caption("Top nutrients driving each pick — helpful vs potential watch-outs, matched to your labs.")
        st.dataframe(why, use_container_width=True)

