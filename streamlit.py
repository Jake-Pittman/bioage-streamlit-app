# app.py — ΔPhenoAge Food Recommender with PDF parsing
import streamlit as st
st.set_page_config(page_title="ΔPhenoAge Food Recommender", layout="wide", initial_sidebar_state="expanded")

import os, re, io, json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Optional dependency: pdfplumber ----------
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

# ---------- Roots / paths ----------
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parent  # folder that contains app.py
env_root = os.environ.get("BIOAGE_ROOT")     # optional override via env/secrets
root = Path(env_root) if env_root else REPO_ROOT


PROC   = root / "processed"
ASSETS = root / "app_assets"
CORE   = root / "models" / "RewardModel" / "core_scoring"

FND_PARQUET   = PROC / "FNDDS_MASTER_PER100G.parquet"
CAT_PARQUET   = ASSETS / "food_catalog.parquet"                 # built by earlier "app-kernel" cell
LAB_SCHEMA_JS = ASSETS / "lab_schema.json"                       # aliases + expected units (optional but nice)
TEMPLATE_CSV  = ASSETS / "labs_upload_template.csv"
CONSENSUS_CSV = CORE / "consensus_food_scores.csv"
GUARDRAILS_CSV= CORE / "core_food_scores_guardrails.csv"
ATTR_CSV      = CORE / "core_food_attribution_top50_compact.csv" # optional

# ---------- Cached loaders ----------
@st.cache_data(show_spinner=False)
def load_lab_schema():
    if LAB_SCHEMA_JS.exists():
        with open(LAB_SCHEMA_JS, "r") as f:
            return json.load(f)
    # fallback minimal schema
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
    """Catalog with FoodCode, Desc, kcal_per_100g, score (lower is better), tags, per100kcal."""
    cat = pd.read_parquet(CAT_PARQUET)
    cat["FoodCode"] = pd.to_numeric(cat["FoodCode"], errors="coerce").astype("Int64")

    # join per-100kcal score (guarded/core)
    src = CONSENSUS_CSV if CONSENSUS_CSV.exists() else GUARDRAILS_CSV
    if src.exists():
        r = pd.read_csv(src)
        r["FoodCode"] = pd.to_numeric(r["FoodCode"], errors="coerce").astype("Int64")
        percol = "guarded_score_per_100kcal" if "guarded_score_per_100kcal" in r.columns else (
                 "core_score_per_100kcal" if "core_score_per_100kcal" in r.columns else None)
        if percol:
            cat = cat.merge(r[["FoodCode", percol]].rename(columns={percol:"per100kcal"}),
                            on="FoodCode", how="left")
        else:
            cat["per100kcal"] = np.nan
    else:
        cat["per100kcal"] = np.nan

    if "tags" not in cat.columns:
        cat["tags"] = np.nan
    return cat

# ---------- CSV normalization (for CSV fallback) ----------
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

# ---------- PDF parser (table-first, then text fallback; unit fixes) ----------
def parse_pdf_labs(file_like) -> dict:
    if not HAS_PDFPLUMBER:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")

    expected_map = {
    # --- PhenoAge-critical ---
    "albumin":"albumin","serum albumin":"albumin","albumin, serum":"albumin","alb":"albumin",

    # ALP / alkaline phosphatase (many vendors/variants)
    "alkaline phosphatase":"alp","alkaline phosphatase (total)":"alp",
    "alkaline phosphatase, total":"alp","alkaline phosphatase, serum":"alp",
    "alkaline phosphatase (alp)":"alp","alkaline phosphatase (alk phos)":"alp",
    "alk phos":"alp","alk. phos":"alp","alk. phosphatase":"alp","alk-phos":"alp",
    "alk phosphatase":"alp","alk-phosphatase":"alp","alp":"alp",

    "c-reactive protein":"crp","c reactive protein":"crp","hs-crp":"crp","hscrp":"crp","crp":"crp",

    # glucose / fasting glucose
    "glucose, fasting":"fasting_glucose","glucose (fasting)":"fasting_glucose",
    "fasting glucose":"fasting_glucose","glucose":"fasting_glucose",

    "wbc":"wbc","white blood cell":"wbc","white blood cells":"wbc",

    "lymphocyte %":"lymphs_pct","lymphocytes %":"lymphs_pct","lymphocytes":"lymphs_pct","lymphs":"lymphs_pct",
    "lymphocytes absolute":"lymphs_abs","lymphs absolute":"lymphs_abs","abs lymphs":"lymphs_abs",

    "mcv":"mcv","mean corpuscular volume":"mcv",
    "rdw":"rdw","red cell distribution width":"rdw",

    # extras (safe if present)
    "creatinine":"creatinine","creat":"creatinine",
    "bun":"bun","blood urea nitrogen":"bun",
}

    range_pat  = re.compile(r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\b")
    number_pat = re.compile(r"(-?\d+(?:\.\d+)?)")

    def cell_has_range(text): return bool(range_pat.search(text or ""))
    def first_number(text):
        if not text: return None
        if cell_has_range(text): return None
        m = number_pat.search(text); return float(m.group(1)) if m else None
    def match_key(lbl_raw):
        s = (lbl_raw or "").strip().lower()
        for syn, std in sorted(expected_map.items(), key=lambda kv: -len(kv[0])):
            if syn in s: return std
        return None

    labs, aux = {}, {}

    # --- Table extraction ---
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            for tbl in (page.extract_tables() or []):
                for row in tbl or []:
                    if not row: continue
                    key = match_key(row[0])
                    if not key: continue
                    chosen_val, chosen_txt = None, None
                    for cell in row[1:]:
                        val = first_number(cell)
                        if val is not None:
                            chosen_val, chosen_txt = val, (cell or ""); break
                    if chosen_val is None: continue
                    lowtxt = (chosen_txt or "").lower()
                    # unit fixups
                    if key == "crp" and ("mg/dl" in lowtxt) and ("mg/l" not in lowtxt):
                        chosen_val *= 10.0
                    if key == "albumin" and "g/l" in lowtxt:
                        chosen_val /= 10.0
                    # stash
                    if key == "lymphs_abs": aux["lymphs_abs"] = chosen_val
                    elif key == "lymphs_pct": labs["lymphs"] = chosen_val
                    else: labs[key] = chosen_val

    # --- Text fallback (scan if anything missing) ---
    file_like.seek(0)
    full = ""
    with pdfplumber.open(file_like) as pdf:
        for p in pdf.pages:
            full += (p.extract_text() or "").replace("\n"," ") + " "
    BIG = 800
    def from_text(label, std_key):
        if std_key in labs: return
        pat = re.compile(rf"(?i)\b{re.escape(label)}\b" + rf".{{0,{BIG}}}?" + r"(-?\d+(?:\.\d+)?)")
        m = pat.search(full)
        if not m: return
        val = float(m.group(1))
        window = full[max(0,m.start()-40): m.end()+40].lower()
        if std_key == "crp" and ("mg/dl" in window) and ("mg/l" not in window):
            val *= 10.0
        if std_key == "albumin" and "g/l" in window:
            val /= 10.0
        if std_key in ("lymphs_pct","lymphs_abs"): aux[std_key] = val
        else: labs[std_key] = val

    for label, key in expected_map.items(): from_text(label, key)

# Final ALP safety net: look for "alk ... phos" pattern anywhere and grab the nearby number
    if "alp" not in labs:
        m = re.search(r"(?i)alk[^\n]{0,80}?phos[^\n]{0,80}?(-?\d+(?:\.\d+)?)", full)
        if m:
            try:
                labs["alp"] = float(m.group(1))
            except Exception:
                pass

    # derive lymphs% if needed (abs in 10^3/µL; WBC same unit)
    if "lymphs" not in labs:
        if aux.get("lymphs_pct") is not None:
            labs["lymphs"] = aux["lymphs_pct"]
        elif aux.get("lymphs_abs") is not None and labs.get("wbc"):
            labs["lymphs"] = (aux["lymphs_abs"] / labs["wbc"]) * 100.0

    # optional ratio, may be useful elsewhere
    if labs.get("bun") and labs.get("creatinine"):
        labs["bun_to_cr_ratio"] = labs["bun"] / labs["creatinine"]

    return labs

# ---------- PhenoAge (BioAge) ----------
# ==== PATCH: PhenoAge fix + dedup + readable attribution ====
import re
import numpy as np
import pandas as pd
import streamlit as st

# ---- 1) PhenoAge (Levine) — robust & unit-correct ----
# --- Replace your phenoage_from_row with this ---
def phenoage_from_row(row: pd.Series):
    """
    Original Levine PhenoAge ("orig") using the official coefficients and
    Gompertz mapping. Required inputs (typical clinical units shown):
      albumin [g/dL], creatinine [mg/dL], glucose [mg/dL], crp_mgL [mg/L],
      lymphocyte_pct [%], mcv [fL], rdw [%], alk_phosphatase [U/L],
      wbc [10^3/µL], age_years [years]
    Returns (PhenoAge, PhenoAgeAccel) or (None, None) if any inputs missing.
    """

    need = ["age_years","albumin","creatinine","glucose","crp_mgL",
            "lymphocyte_pct","mcv","rdw","alk_phosphatase","wbc"]
    if any(pd.isna(row.get(k)) for k in need):
        return (None, None)

    # ---- Convert to SI that the published model uses ----
    albumin_gL   = float(row["albumin"]) * 10.0          # g/dL → g/L
    creat_umol   = float(row["creatinine"]) * 88.4       # mg/dL → µmol/L
    glucose_mmol = float(row["glucose"]) / 18.0          # mg/dL → mmol/L
    lncrp        = np.log(max(float(row["crp_mgL"]), 0) + 1e-6)  # mg/L → ln(mg/L)
    lymph        = float(row["lymphocyte_pct"])
    mcv          = float(row["mcv"])
    rdw          = float(row["rdw"])
    alp          = float(row["alk_phosphatase"])
    wbc          = float(row["wbc"])
    age          = float(row["age_years"])

    # ---- Linear predictor (xb_orig) with age term ----
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

    # ---- 10-year mortality risk under Gompertz ----
    # m_orig = 1 - exp( (-1.51714 * exp(xb)) / 0.007692696 )
    m = 1.0 - np.exp( (-1.51714 * np.exp(xb)) / 0.007692696 )
    m = float(np.clip(m, 1e-15, 1 - 1e-15))

    # ---- Map mortality risk to PhenoAge ----
    pheno = (np.log(-0.0055305 * np.log(1.0 - m)) / 0.09165) + 141.50225
    accel = pheno - age

    # Clamp for UI
    pheno = float(np.clip(pheno, 0.0, 140.0))
    accel = float(np.clip(accel, -60.0, 90.0))
    return pheno, accel


# ---------- Categorization for Top-100 splits ----------
# ---------- Dedup & categorization helpers (single source of truth) ----------
_STOPWORDS = re.compile(
    r"\b(ns as to form|nsf|nfs|assume.*?|fat not added in cooking|"
    r"no added fat|from (?:fresh|frozen|canned)|fresh|frozen|canned|raw|cooked|"
    r"reconstituted|for use on a sandwich|reduced sodium|low(?:fat| sodium)|diet)\b",
    flags=re.I
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

_VEG_RE   = re.compile(r"\b(kale|chard|lettuce|greens?|watercress|parsley|basil|cilantro|spinach|broccoli|cabbage|cauliflower|asparagus|zucchini|squash|okra|tomato|mushroom|onion|pepper|beet|cucumber|pickle|radish|artichoke|brussels|celery)\b", re.I)
_FRUIT_RE = re.compile(r"\b(apple|banana|orange|berry|berries|strawberry|blueberry|raspberry|blackberry|grape|pear|peach|plum|cherry|pineapple|mango|papaya|melon|watermelon|cantaloupe|honeydew|kiwi|lemon|lime|grapefruit|pomegranate|date|fig|raisin|prune|avocado)\b", re.I)
_LEG_RE   = re.compile(r"\b(bean|lentil|chickpea|pea|soy|tofu|tempeh|edamame|peanut|hummus|bread|rice|pasta|noodle|oat|oatmeal|barley|quinoa|corn|tortilla|wheat|bran|cereal|bulgur|couscous|polenta)\b", re.I)
_PROT_RE  = re.compile(r"\b(beef|pork|chicken|turkey|lamb|veal|bacon|sausage|ham|steak|meat|fish|salmon|tuna|cod|tilapia|shrimp|oyster|clam|scallop|crab|lobster|egg|cheese|yogurt|milk|dairy|cottage cheese)\b", re.I)
_FAT_RE   = re.compile(
    r"\b(oil|olive oil|canola|avocado oil|sunflower oil|safflower|sesame oil|peanut oil|"
    r"butter|ghee|margarine|shortening|lard|mayonnaise|mayo|aioli|tahini|nut butter|"
    r"peanut butter|almond butter|cashew butter|seed butter|tallow)\b"
    r"|\b(almond|walnut|pecan|cashew|pistachio|hazelnut|macadamia|sunflower seed|"
    r"pumpkin seed|flaxseed|chia|sesame seed)s?\b",
    re.I
)

def coarse_category(desc: str, tags: str) -> str:
    t = (tags or "")
    s = str(desc or "")
    if "cereal_fortified" in t: return "legume_grains"
    if "oil_fat_sauce" in t:    return "fats"
    if "leafy_green" in t or "pickled_veg" in t or "seaweed_algae" in t: return "vegetables"
    if _FAT_RE.search(s):  return "fats"
    if _PROT_RE.search(s): return "protein"
    if _VEG_RE.search(s):  return "vegetables"
    if _FRUIT_RE.search(s):return "fruit"
    if _LEG_RE.search(s):  return "legume_grains"
    return "other"

def dedup_rank(df: pd.DataFrame, include_tags: str = "", exclude_tags: str = "") -> pd.DataFrame:
    R = df.copy()
    if include_tags.strip():
        inc = [t.strip() for t in include_tags.split(",") if t.strip()]
        if inc:
            R = R[R["tags"].fillna("").apply(lambda s: any(t in s for t in inc))]
    if exclude_tags.strip():
        exc = [t.strip() for t in exclude_tags.split(",") if t.strip()]
        if exc:
            R = R[~R["tags"].fillna("").apply(lambda s: any(t in s for t in exc))]
    R["dedup_key"] = R["Desc"].map(_normalize_desc)
    R = R.sort_values(["score", "kcal_per_100g"], ascending=[True, True])
    R = R.drop_duplicates(subset="dedup_key", keep="first")
    return R.reset_index(drop=True)

# ---------- Why these foods? (pretty, robust) ----------
_PRETTY = {
    "KCAL":"Calories","CARB":"Carbohydrates","PROT":"Protein","TFAT":"Total fat",
    "SFAT":"Saturated fat","PFAT":"Polyunsat. fat","MFAT":"Monounsat. fat",
    "SUGR":"Sugars","FIBE":"Fiber","CALC":"Calcium","PHOS":"Phosphorus",
    "MAGN":"Magnesium","POTA":"Potassium","ZINC":"Zinc","SELE":"Selenium",
    "COPP":"Copper","ATOC":"Vitamin E","VARA":"Vitamin A (RAE)","VK":"Vitamin K",
    "VC":"Vitamin C","VB1":"Thiamin (B1)","VB2":"Riboflavin (B2)","VB6":"Vitamin B6",
    "VB12":"Vitamin B12","NIAC":"Niacin (B3)","CAFF":"Caffeine","FDFE":"Iron",
}

def _parse_terms(term_str: str, top=3):
    if not isinstance(term_str, str) or not term_str.strip():
        return []
    items = []
    for chunk in term_str.split(";"):
        m = re.search(r"(?:NUTR_)?([A-Z0-9]+)\(([-+]?[\d\.]+)", chunk.strip())
        if not m:
            continue
        code = m.group(1)
        val  = abs(float(m.group(2)))
        items.append((code, val))
    items.sort(key=lambda x: x[1], reverse=True)
    return [_PRETTY.get(code, code) for code, _ in items[:top]]

def build_why_table(ranked_df: pd.DataFrame, attr_csv: Path, top_n=15):
    try:
        if not attr_csv.exists():
            return None
        attr = pd.read_csv(attr_csv)
        need = {"FoodCode","Desc","core_score_per_100kcal","top_negative_terms","top_positive_terms"}
        if not need.issubset(attr.columns):
            return None

        J = ranked_df[["FoodCode","Desc","kcal_per_100g","score"]].merge(
            attr[list(need)], on=["FoodCode","Desc"], how="left"
        ).head(top_n)

        rows = []
        for _, r in J.iterrows():
            helps = ", ".join(_parse_terms(r.get("top_negative_terms",""), top=3)) or "—"
            watch = ", ".join(_parse_terms(r.get("top_positive_terms",""), top=2)) or "—"
            rows.append({
                "Food": r["Desc"],
                "Why it helps (top nutrients)": helps,
                "Potential watch-outs": watch,
                "Score (per 100 kcal)": (None if pd.isna(r.get("score")) else round(float(r["score"]), 3)),
            })
        return pd.DataFrame(rows)
    except Exception:
        return None

# ---------- UI ----------
st.title("ΔPhenoAge Food Recommender")

with st.sidebar:
    st.markdown("**Paths**")
    st.text(f"Root: {root}")
    ok = all(p.exists() for p in [CAT_PARQUET])
    if not ok:
        st.error("Missing app assets. Please generate:\n- food_catalog.parquet")
    tag_help = "Optional: filter shown foods by tags (e.g., leafy_green, tea_coffee)"
    include_tags = st.text_input("Include tags (comma-separated)", value="")
    # keep fats visible by default
    exclude_tags = st.text_input("Exclude tags (comma-separated)",)
    top_n_show = st.slider("How many foods to show in table views", 20, 200, 100, step=10)
    st.markdown("---")
    if TEMPLATE_CSV.exists():
        st.download_button("Download labs CSV template", TEMPLATE_CSV.read_bytes(), file_name="labs_upload_template.csv")
    if not HAS_PDFPLUMBER:
        st.info("PDF parsing needs `pdfplumber`. Install with: `pip install pdfplumber`")

schema  = load_lab_schema()
catalog = load_catalog() if CAT_PARQUET.exists() else None

# ---------- 1) Upload labs (PDF or CSV) + age ----------
st.subheader("1) Upload your blood test (PDF preferred) and enter your age")
c_up1, c_up2 = st.columns(2)
with c_up1:
    pdf_file = st.file_uploader("Upload PDF blood test", type=["pdf"])
with c_up2:
    csv_file = st.file_uploader("Or upload CSV (headers may use aliases)", type=["csv"])
age_input = st.number_input("Your age (years)", min_value=0, max_value=120, value=45)

# Parse on button
run_clicked = st.button("Run model on uploaded labs → compute BioAge & recommend foods")

parsed_df = None
if run_clicked:
    labs_dict = {}
    try:
        if pdf_file is not None:
            data = pdf_file.read()
            labs_dict = parse_pdf_labs(io.BytesIO(data))
        elif csv_file is not None:
            raw = pd.read_csv(csv_file)
            norm = normalize_labs(raw, schema)
            labs_dict = norm.iloc[0].to_dict()
        else:
            st.error("Please upload a PDF or CSV first.")
    except Exception as e:
        st.error(f"Failed to parse labs: {e}")

    # Map parser keys → app’s canonical keys
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

# ---------- 2) Show normalized labs & BioAge ----------
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


# ---------- Dedup, categorization, and “why” helpers ----------
# ---------- Dedup & categorization helpers ----------
# ---------- Dedup & categorization helpers ----------
_STOPWORDS = re.compile(
    r"\b("
    r"ns as to form|nsf|nfs|assume.*?|fat not added in cooking|no added fat|"
    r"from (?:fresh|frozen|canned)|fresh|frozen|canned|raw|cooked|reconstituted|"
    r"for use on a sandwich|reduced sodium|low(?:fat| sodium)|diet|"
    r"made with (?:margarine|butter|oil|ghee)"
    r")\b",
    re.I
)

def _normalize_desc(desc: str) -> str:
    s = str(desc or "").lower()
    s = re.sub(r"\(.*?\)", " ", s)          # drop parentheticals
    s = _STOPWORDS.sub(" ", s)              # drop prep notes
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # unify common names
    s = s.replace("water cress", "watercress")
    s = s.replace("beet green", "beet greens")
    s = s.replace("turnip green", "turnip greens")
    return s

# Broad detectors
_VEG_RE   = re.compile(r"\b(kale|chard|lettuce|greens?|watercress|parsley|basil|cilantro|spinach|broccoli|cabbage|cauliflower|asparagus|zucchini|squash|okra|tomato|mushroom|onion|pepper|beet|cucumber|pickle|radish|artichoke|brussels|celery|collards|mustard greens|turnip greens)\b", re.I)
_FRUIT_RE = re.compile(r"\b(apple|banana|orange|berry|berries|strawberry|blueberry|raspberry|blackberry|grape|pear|peach|plum|cherry|pineapple|mango|papaya|melon|watermelon|cantaloupe|honeydew|kiwi|lemon|lime|grapefruit|pomegranate|date|fig|raisin|prune|avocado)\b", re.I)
_LEG_RE   = re.compile(r"\b(bean|lentil|chickpea|pea|soy|tofu|tempeh|edamame|peanut|hummus|bread|rice|pasta|noodle|oat|oatmeal|barley|quinoa|corn|tortilla|wheat|bran|cereal|bulgur|couscous|polenta)\b", re.I)

# Heavier protein coverage (favor meat/fish)
_PROT_RE  = re.compile(
    r"\b("
    r"beef|steak|veal|lamb|pork|bacon|sausage|ham|chicken|turkey|duck|"
    r"fish|seafood|salmon|tuna|sardine|sardines|anchovy|anchovies|mackerel|herring|trout|cod|halibut|tilapia|"
    r"shrimp|prawn|oyster|clam|scallop|crab|lobster|"
    r"egg|eggs|cheese|yogurt|kefir|milk|cottage cheese"
    r")\b", re.I
)

# “Pure fats” (oils, spreads, nuts/seeds, nut butters)
_FAT_RE   = re.compile(
    r"\b("
    r"oil|olive oil|canola|avocado oil|sunflower oil|safflower|sesame oil|peanut oil|coconut oil|"
    r"butter|ghee|margarine|shortening|lard|mayonnaise|mayo|aioli|tahini|"
    r"nut butter|peanut butter|almond butter|cashew butter|seed butter|tallow|"
    r"walnut|walnuts|almond|almonds|pecan|pecans|cashew|cashews|pistachio|pistachios|hazelnut|hazelnuts|"
    r"macadamia|sunflower seed|sunflower seeds|pumpkin seed|pumpkin seeds|flaxseed|chia|sesame seed|sesame seeds"
    r")\b", re.I
)

# Plant milks -> don't treat as protein/fats; bucket as 'other'
_BEV_RE   = re.compile(r"\b(almond milk|soy milk|oat milk|rice milk|coconut milk|hemp milk|cashew milk)\b", re.I)

def coarse_category(desc: str, tags: str) -> str:
    t = (tags or "")
    s = str(desc or "").lower()

    # Tag-guided priorities
    if "cereal_fortified" in t:
        return "legume_grains"
    if "tea_coffee" in t or "zero_kcal_beverage" in t:
        return "other"

    veg_hit = ("leafy_green" in t) or bool(_VEG_RE.search(s))
    fruit_hit = bool(_FRUIT_RE.search(s))
    leg_hit = bool(_LEG_RE.search(s))
    prot_hit = bool(_PROT_RE.search(s)) and not _BEV_RE.search(s)  # exclude plant milks
    fat_hit  = ("oil_fat_sauce" in t) or bool(_FAT_RE.search(s))
    bev_hit  = bool(_BEV_RE.search(s))

    # Resolve overlaps:
    # If it's a veg dish "made with butter/oil", still call it VEGETABLES (not fats)
    if veg_hit and fat_hit:
        return "vegetables"

    if prot_hit:
        return "protein"
    if fat_hit:
        return "fats"
    if veg_hit:
        return "vegetables"
    if fruit_hit:
        return "fruit"
    if leg_hit:
        return "legume_grains"
    if bev_hit:
        return "other"
    return "other"


def dedup_rank(df: pd.DataFrame, include_tags: str = "", exclude_tags: str = "") -> pd.DataFrame:
    """Filter by tags, then keep best (lowest score) per normalized description."""
    R = df.copy()

    if include_tags.strip():
        inc = [t.strip() for t in include_tags.split(",") if t.strip()]
        if inc:
            R = R[R["tags"].fillna("").apply(lambda s: any(t in s for t in inc))]

    if exclude_tags.strip():
        exc = [t.strip() for t in exclude_tags.split(",") if t.strip()]
        if exc:
            R = R[~R["tags"].fillna("").apply(lambda s: any(t in s for t in exc))]

    R["dedup_key"] = R["Desc"].map(_normalize_desc)
    R = R.sort_values(["score", "kcal_per_100g"], ascending=[True, True])
    R = R.drop_duplicates(subset="dedup_key", keep="first")
    return R.reset_index(drop=True)


def build_category_buckets(R_sorted: pd.DataFrame, quotas: dict, top_other: int = 50):
    """
    Iterate a deduped, score-sorted DataFrame and fill each category to its quota.
    Ensures a food never appears in two tabs.
    """
    used = set()
    buckets = {k: [] for k in ["protein","fruit","vegetables","legume_grains","other"]}

    for _, r in R_sorted.iterrows():
        key = _normalize_desc(r["Desc"])
        if key in used:
            continue
        cat = coarse_category(r["Desc"], r.get("tags", ""))
        # enforce quotas for non-'other'
        if cat != "other":
            if len(buckets[cat]) >= quotas.get(cat, 0):
                continue
        buckets[cat].append(r)
        used.add(key)

        # early exit: if all non-'other' filled and we have enough 'other'
        if all(len(buckets[c]) >= quotas.get(c, 0) for c in ["protein","fruit","vegetables","legume_grains"]) \
           and len(buckets["other"]) >= top_other:
            break

    # Convert lists to DataFrames
    for k in buckets:
        buckets[k] = (pd.DataFrame(buckets[k])
                      if len(buckets[k]) else
                      pd.DataFrame(columns=R_sorted.columns))
    return buckets


# ---------- 3) Food recommendations ----------
# ---------- 3) Food recommendations ----------
st.subheader("3) Food recommendations")

if (parsed_df is not None) and (catalog is not None) and len(catalog):

    # 1) Filter + de-duplicate
    R_all = dedup_rank(catalog, include_tags, exclude_tags)
    R_all["category"] = [coarse_category(d, t) for d, t in zip(R_all["Desc"], R_all["tags"])]
    R_all = R_all.sort_values("score", ascending=True).reset_index(drop=True)

    # 2) Top 100 overall
    top_overall = R_all.head(100).copy()
    st.markdown("**Top 100 overall (lower = better)**")
    st.dataframe(
        top_overall[["FoodCode","Desc","kcal_per_100g","score","tags","category"]]
                   .head(top_n_show),
        use_container_width=True
    )

    # 3) Category tabs with quotas (Protein + Fats + Fruit + Vegetables + Legume/Grains + Other)
    tabs = st.tabs(["Protein", "Fats", "Fruit", "Vegetables", "Legume/Grains", "Other"])
    QUOTA = {"protein": 5, "fats": 5, "fruit": 7, "vegetables": 10, "legume_grains": 4, "other": 10}
    CAT_ORDER = ["protein","fats","fruit","vegetables","legume_grains","other"]

    for tab, cat in zip(tabs, CAT_ORDER):
        with tab:
            sub = R_all[R_all["category"] == cat].head(QUOTA.get(cat, 10))
            if sub.empty:
                st.info(f"No foods found for category: {cat.replace('_','/')}")
            else:
                st.write(f"**Top {len(sub)} — {cat.replace('_','/').title()}**")
                st.dataframe(sub[["FoodCode","Desc","kcal_per_100g","score","tags"]], use_container_width=True)
                st.download_button(
                    f"Download {cat.replace('_','/').title()} (CSV)",
                    sub.to_csv(index=False).encode("utf-8"),
                    file_name=f"top_{cat}.csv",
                    key=f"dl_{cat}"
                )

    # 4) Why these foods? (safe, optional)
    why = build_why_table(top_overall, ATTR_CSV, top_n=15)
    if why is not None and len(why):
        st.subheader("Why these foods?")
        st.caption("Top nutrients driving each pick — helpful vs. potential watch-outs.")
        st.dataframe(why, use_container_width=True)



elif parsed_df is None:
    st.info("Upload a PDF/CSV, enter age, then click the button to get recommendations.")
elif catalog is None:
    st.error("Food catalog not found. Re-run your data prep to create app_assets/food_catalog.parquet.")
