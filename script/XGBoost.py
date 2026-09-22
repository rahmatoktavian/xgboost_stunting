from __future__ import annotations

import json
import platform
import re
import sys
import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
import xgboost
from docx import Document
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TOP_FEATURE_COUNT = 8
MODEL_NAME = "XGBoost"
MODEL_SLUG = "xgboost"
OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 3
N_SEARCH_ITER = 20
RUN_RISK_FACTOR_SCENARIO = True  # nested CV tambahan tanpa WAZ/BB (metrik saja)

PROJECT_DIR = Path(__file__).resolve().parent
if not (PROJECT_DIR / "Data600.xlsx").exists():
    PROJECT_DIR = Path("//xgboost_stunting")

EXCEL_FILE = PROJECT_DIR / "Data600.xlsx"
CODEBOOK_FILE = PROJECT_DIR / "Workbook.docx"
OUTPUT_DIR = PROJECT_DIR / "output_data600_fixed_xgboost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_TOTAL_DATASET = 696

def excel_column_to_index(column_letters: str) -> int:
    index = 0
    for char in column_letters.strip().upper():
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def index_to_excel_column(index: int) -> str:
    index += 1
    letters = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def clean_column_name(name, fallback: str) -> str:
    if pd.isna(name) or str(name).strip() == "":
        name = fallback
    name = str(name).replace("\n", " ").replace("\r", " ").replace("\t", " ")
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"[^\w\s.,;:/()\-+%]", "", name)
    return name or fallback


def make_unique(names: list[str]) -> list[str]:
    counts = {}
    unique_names = []
    for name in names:
        if name not in counts:
            counts[name] = 0
            unique_names.append(name)
        else:
            counts[name] += 1
            unique_names.append(f"{name}__dup{counts[name]}")
    return unique_names


def normalize_codebook_code(value) -> str:
    text = str(value).strip().upper()
    text = re.sub(r"\s+", "", text)
    text = text.replace("-", "")
    text = re.sub(r"[^A-Z0-9.]", "", text)
    return text


def normalize_category_value(value) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return text


def parse_codebook_values(text: str) -> dict[str, str]:
    text = str(text).replace("\n", " | ")
    matches = re.findall(r"(?:^|\|)\s*(\d+(?:\.\d+)?)\s*[:.]\s*([^|]+)", text)
    values = {}
    for code, label in matches:
        cleaned_label = re.sub(r"\s+", " ", str(label)).strip()
        if cleaned_label:
            values[normalize_category_value(code)] = cleaned_label
    return values


def build_codebook_maps(path: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    if not path.exists():
        return {}, {}
    document = Document(path)
    label_map = {}
    value_label_map = {}
    valid_code_pattern = re.compile(r"^[A-Z]{1,3}\d+[A-Z]?(?:\.\d+)?$")
    for table in document.tables:
        for row in table.rows[1:]:
            cells = [cell.text.strip() for cell in row.cells]
            if len(cells) < 2:
                continue
            code = normalize_codebook_code(cells[0])
            label = re.sub(r"\s+", " ", str(cells[1])).strip()
            if not code or not label or not valid_code_pattern.match(code):
                continue
            label_map[code] = label
            if len(cells) >= 3:
                parsed_values = parse_codebook_values(cells[2])
                if parsed_values:
                    value_label_map[code] = parsed_values
    return label_map, value_label_map


def extract_feature_code(raw_name: str) -> str | None:
    text = str(raw_name).replace("__dup", " __dup").strip().upper()
    patterns = [r"\b([A-Z]{1,3}\d+[A-Z]?(?:\.\d+)?)\b", r"\b(I)[\-\s]?(\d+[A-Z]?)"]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            if len(match.groups()) == 2 and match.group(1) == "I":
                return normalize_codebook_code("I" + match.group(2))
            return normalize_codebook_code(match.group(1))
    return None


def code_candidates(code: str | None) -> list[str]:
    if not code:
        return []
    candidates = [code]
    if re.match(r"^I\d+[A-Z]$", code):
        candidates.append(code[:-1])
    return candidates


def strip_feature_code(raw_name: str) -> str:
    text = str(raw_name)
    text = re.sub(r"__dup\d+$", "", text)
    text = re.sub(r"^\s*[A-Z]{1,3}[-\s]?\d+[A-Za-z]?(?:\.\d+)?[.\s:-]*", "", text).strip()
    return re.sub(r"\s+", " ", text) or str(raw_name)


CODEBOOK_LABEL_MAP, CODEBOOK_VALUE_LABEL_MAP = build_codebook_maps(CODEBOOK_FILE)
MANUAL_FEATURE_LABELS_BY_LETTER = {
    "D": "Usia baduta",
    "E": "Jenis kelamin baduta",
    "F": "Kondisi saat pengukuran",
    "G": "Berat badan baduta saat pengukuran",
    "H": "Panjang/tinggi badan baduta saat pengukuran",
    "JM": "WHZ wasting",
    "JO": "BAZ overweight",
    "JQ": "WAZ underweight",
}
MANUAL_FEATURE_LABELS_BY_CODE = {
    "C4.3": "PMT pangan lokal (Apakah dihabiskan)",
    "D16U": "Recall mengenai Makanan padat, setengah padat, makana lumat lainnya termasuk kue-kue seperti kue pisang, cucur, pancong, permen",
}
MANUAL_FEATURE_VALUE_LABELS_BY_LETTER = {
    "E": {"1": "Laki-laki", "2": "Perempuan"},
    "F": {"1": "Sehat", "2": "Sakit ringan", "3": "Sakit berat/kronis", "4": "Diare", "5": "Oedema"},
}
MANUAL_FEATURE_VALUE_LABELS_BY_CODE = {
    "C4.3": {"1": "Ya", "2": "Tidak", "888": "Empty", "MISSING_CODE": "Empty"},
    "D16U": {"1": "Ya", "2": "Tidak", "888": "Empty", "MISSING_CODE": "Empty"},
}


def feature_display_label(raw_name: str) -> str:
    excel_letter = FEATURE_SOURCE_MAP.get(raw_name, "")
    if excel_letter in MANUAL_FEATURE_LABELS_BY_LETTER:
        return MANUAL_FEATURE_LABELS_BY_LETTER[excel_letter]
    code = extract_feature_code(raw_name)
    for candidate in code_candidates(code):
        if candidate in MANUAL_FEATURE_LABELS_BY_CODE:
            return MANUAL_FEATURE_LABELS_BY_CODE[candidate]
        label = CODEBOOK_LABEL_MAP.get(candidate)
        if label:
            return label
    return strip_feature_code(raw_name)


def feature_value_display_label(raw_name: str, value) -> str:
    special_display = {"NOT_APPLICABLE": "Tidak berlaku", "DONT_KNOW": "Tidak tahu", "MISSING_CODE": "Empty"}
    value_text = str(value).strip()
    code = extract_feature_code(raw_name)
    normalized_value = normalize_category_value(value_text)
    for candidate in code_candidates(code):
        manual_code_label = MANUAL_FEATURE_VALUE_LABELS_BY_CODE.get(candidate, {}).get(normalized_value)
        if manual_code_label:
            return manual_code_label
    if value_text in special_display:
        return special_display[value_text]
    excel_letter = FEATURE_SOURCE_MAP.get(raw_name, "")
    manual_label = MANUAL_FEATURE_VALUE_LABELS_BY_LETTER.get(excel_letter, {}).get(normalize_category_value(value_text))
    if manual_label:
        return manual_label
    for candidate in code_candidates(code):
        label = CODEBOOK_VALUE_LABEL_MAP.get(candidate, {}).get(normalized_value)
        if label:
            return label
    return value_text


def subgroup_value_display_label(raw_name: str, value, available_values) -> str:
    """Return a non-empty subgroup option label, preferring Workbook choices."""
    normalized_value = normalize_category_value(value)
    if normalized_value in {"888", "MISSING_CODE"}:
        return "Empty"

    workbook_or_manual_label = str(feature_value_display_label(raw_name, normalized_value)).strip()
    if workbook_or_manual_label and workbook_or_manual_label != normalized_value:
        return workbook_or_manual_label

    special_option_codes = {"", "77", "88", "888", "NOT_APPLICABLE", "DONT_KNOW", "MISSING_CODE"}
    normalized_options = {
        normalize_category_value(option)
        for option in available_values
        if normalize_category_value(option) not in special_option_codes
    }
    if normalized_options and normalized_options.issubset({"1", "2"}):
        return {"1": "Ya", "2": "Tidak"}.get(normalized_value, f"Pilihan {normalized_value}")
    return f"Pilihan {normalized_value}" if normalized_value else "Empty"


def build_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_special_code_map() -> dict[str, str]:
    return {
        "77": "NOT_APPLICABLE",
        "77.0": "NOT_APPLICABLE",
        "88": "DONT_KNOW",
        "88.0": "DONT_KNOW",
        "888": "MISSING_CODE",
        "888.0": "MISSING_CODE",
    }




class SurveyFeatureCleaner(BaseEstimator, TransformerMixin):
    """Pembersih fitur survei - versi FIXED.

    Perbedaan vs versi lama:
    - Kolom pengukuran kontinu (>= ``continuous_min_unique`` nilai unik numerik dan
      rasio numerik >= 0.8 pada data latih) TIDAK terkena pemetaan sentinel 77/88,
      karena 77/88 bisa merupakan nilai pengukuran asli (mis. panjang badan 77 cm).
      Untuk kolom tersebut hanya 888/8888 yang diperlakukan sebagai missing (NaN).
    - Kolom kontinu selalu dipaksa numerik (tidak pernah di-one-hot).
    """

    def __init__(
        self,
        forced_drop_reasons=None,
        missing_threshold=0.90,
        special_code_map=None,
        multiresponse_max_categories=30,
        continuous_min_unique=15,
    ):
        self.forced_drop_reasons = forced_drop_reasons
        self.missing_threshold = missing_threshold
        self.special_code_map = special_code_map
        self.multiresponse_max_categories = multiresponse_max_categories
        self.continuous_min_unique = continuous_min_unique

    def _normalize_special_value(self, value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            token = value.strip()
        else:
            try:
                numeric = float(value)
                token = str(int(numeric)) if numeric.is_integer() else str(value)
            except Exception:
                token = str(value).strip()
        return self.special_code_map_.get(token, value)

    def _detect_continuous_columns(self, X: pd.DataFrame) -> set:
        continuous = set()
        for column in X.columns:
            converted = pd.to_numeric(X[column], errors="coerce")
            non_missing = X[column].notna() & ~X[column].astype(str).str.strip().eq("")
            if non_missing.sum() == 0:
                continue
            numeric_ratio = converted.notna().sum() / non_missing.sum()
            if numeric_ratio >= 0.8 and converted.dropna().nunique() >= self.continuous_min_unique:
                continuous.add(column)
        return continuous

    def _prepare(self, X):
        X = pd.DataFrame(X).copy()
        X = X.reindex(columns=self.original_columns_, fill_value=np.nan) if hasattr(self, "original_columns_") else X
        X = X.replace(r"^\s*$", np.nan, regex=True)
        continuous_columns = getattr(self, "continuous_columns_", set())
        for column in X.columns:
            if column in continuous_columns:
                values = pd.to_numeric(X[column], errors="coerce")
                values = values.mask(values.isin([888.0, 8888.0]))
                X[column] = values
                continue
            if X[column].dtype == "object":
                X[column] = X[column].map(lambda value: re.sub(r"\s+", " ", value.strip()) if isinstance(value, str) else value)
            X[column] = X[column].map(self._normalize_special_value)
        return X

    def _is_multiresponse(self, series: pd.Series) -> tuple[bool, list[str]]:
        as_text = series.dropna().astype(str).str.strip()
        if as_text.empty:
            return False, []
        mask = as_text.str.match(r"^\d+(\s*,\s*\d+)+$")
        if mask.mean() < 0.02:
            return False, []
        categories = sorted({token.strip() for value in as_text[mask] for token in value.split(",")}, key=lambda x: int(x))
        if 1 < len(categories) <= self.multiresponse_max_categories:
            return True, categories
        return False, []

    def fit(self, X, y=None):
        self.original_columns_ = list(pd.DataFrame(X).columns)
        self.forced_drop_reasons_ = dict(self.forced_drop_reasons or {})
        self.special_code_map_ = dict(self.special_code_map or {})
        X_raw = pd.DataFrame(X).replace(r"^\s*$", np.nan, regex=True)
        self.continuous_columns_ = self._detect_continuous_columns(
            X_raw[[column for column in X_raw.columns if column not in self.forced_drop_reasons_]]
        )
        X_prepared = self._prepare(X)
        self.drop_reasons_ = {}
        for column, reason in self.forced_drop_reasons_.items():
            if column in X_prepared.columns:
                self.drop_reasons_[column] = reason
        for column in [col for col in X_prepared.columns if col not in self.drop_reasons_]:
            missing_rate = X_prepared[column].isna().mean()
            non_missing_unique = X_prepared[column].dropna().nunique()
            if missing_rate >= 1.0:
                self.drop_reasons_[column] = "100% missing in training data"
            elif missing_rate > self.missing_threshold:
                self.drop_reasons_[column] = f"missing rate > {self.missing_threshold:.0%} in training data"
            elif non_missing_unique <= 1:
                self.drop_reasons_[column] = "constant feature in training data"
        self.multiresponse_categories_ = {}
        self.numeric_columns_ = []
        self.selected_raw_features_ = []
        for column in X_prepared.columns:
            if column in self.drop_reasons_:
                continue
            if column in self.continuous_columns_:
                self.numeric_columns_.append(column)
                self.selected_raw_features_.append(column)
                continue
            is_multi, categories = self._is_multiresponse(X_prepared[column])
            if is_multi:
                self.multiresponse_categories_[column] = categories
            else:
                non_missing = X_prepared[column].dropna()
                has_special_category = non_missing.astype(str).isin(set(self.special_code_map_.values())).any()
                converted = pd.to_numeric(non_missing, errors="coerce")
                numeric_ratio = converted.notna().mean() if len(non_missing) else 0.0
                if not has_special_category and numeric_ratio >= 0.95:
                    self.numeric_columns_.append(column)
            self.selected_raw_features_.append(column)
        self.feature_names_out_ = []
        for column in self.selected_raw_features_:
            if column in self.multiresponse_categories_:
                self.feature_names_out_.extend([f"{column}__option_{category}" for category in self.multiresponse_categories_[column]])
            else:
                self.feature_names_out_.append(column)
        return self

    def transform(self, X):
        X_prepared = self._prepare(X)
        transformed = pd.DataFrame(index=X_prepared.index)
        for column in self.selected_raw_features_:
            if column in self.multiresponse_categories_:
                text_series = X_prepared[column].fillna("").astype(str)
                for category in self.multiresponse_categories_[column]:
                    transformed[f"{column}__option_{category}"] = text_series.map(
                        lambda value, category=category: int(category in {token.strip() for token in value.split(",")})
                    )
            else:
                if column in self.numeric_columns_:
                    transformed[column] = pd.to_numeric(X_prepared[column], errors="coerce")
                else:
                    transformed[column] = X_prepared[column].map(lambda value: np.nan if pd.isna(value) else str(value))
        return transformed

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)

    def get_drop_audit(self) -> pd.DataFrame:
        return pd.DataFrame([{"feature": column, "reason": reason} for column, reason in self.drop_reasons_.items()])

    def get_selected_features(self) -> pd.DataFrame:
        rows = []
        for column in self.selected_raw_features_:
            rows.append(
                {
                    "raw_feature": column,
                    "display_label": feature_display_label(column),
                    "excel_column": FEATURE_SOURCE_MAP.get(column, ""),
                    "encoding": "multi-hot" if column in self.multiresponse_categories_ else "model input",
                    "generated_feature_count": len(self.multiresponse_categories_.get(column, [column])),
                }
            )
        return pd.DataFrame(rows)


def clean_transformed_name(name: str, raw_features: list[str]) -> str:
    transformed_name = re.sub(r"^(num|cat)__", "", str(name)).strip()
    if "__option_" in transformed_name:
        raw_name, option_value = transformed_name.split("__option_", 1)
        return f"{feature_display_label(raw_name)} = {feature_value_display_label(raw_name, option_value)}"
    for raw_name in sorted(raw_features, key=len, reverse=True):
        if transformed_name == raw_name:
            return feature_display_label(raw_name)
        if transformed_name.startswith(f"{raw_name}_"):
            category_value = transformed_name[len(raw_name) + 1 :]
            return f"{feature_display_label(raw_name)} = {feature_value_display_label(raw_name, category_value)}"
    return feature_display_label(transformed_name)


def transformed_to_raw_feature(name: str, raw_features: list[str]) -> str:
    """Map transformed feature names back to the raw DataFrame column expected by the pipeline."""
    transformed_name = re.sub(r"^(num|cat)__", "", str(name)).strip()
    if "__option_" in transformed_name:
        return transformed_name.split("__option_", 1)[0]
    for raw_name in sorted(raw_features, key=len, reverse=True):
        if transformed_name == raw_name or transformed_name.startswith(f"{raw_name}_"):
            return raw_name
    return transformed_name


def json_ready(value):
    """Convert NumPy/pandas scalar values to JSON-safe Python objects."""
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    return value


def api_label(value) -> str:
    return re.sub(r"\s+", " ", str(value).replace("\n", " ")).strip()


def build_form_predict_parameters(feature_schema: list[dict]) -> list[dict]:
    schema_by_key = {str(item.get("key")): item for item in feature_schema if item.get("key")}
    parameters = []
    for rank, spec in enumerate(FORM_PREDICT_PARAMETER_SPECS, start=1):
        key = spec["key"]
        schema = schema_by_key.get(key, {})
        used_in_model = bool(spec.get("used_in_model", True) and schema)
        parameter = {
            "rank": rank,
            "key": key,
            "raw_feature": schema.get("raw_feature"),
            "excel_column": schema.get("excel_column", key if used_in_model else ""),
            "label": spec["label"],
            "label_id": spec["label"],
            "label_en": spec.get("label_en", spec["label"]),
            "raw_label": schema.get("label", spec["label"]),
            "input_type": spec.get("input_type", schema.get("input_type", "text")),
            "default_value": schema.get("default_value", spec.get("default_value")),
            "options": spec.get("options", schema.get("options", [])),
            "used_in_model": used_in_model,
            "required": bool(spec.get("required", used_in_model)),
        }
        parameters.append(parameter)
    return parameters


def make_unique_display_names(names: list[str]) -> list[str]:
    counts = {}
    output = []
    for name in names:
        if name not in counts:
            counts[name] = 0
            output.append(name)
        else:
            counts[name] += 1
            output.append(f"{name} ({counts[name] + 1})")
    return output


def chart_label(name: str, width: int = 34, max_chars: int = 96) -> str:
    """Wrap long description labels so chart axes stay readable."""
    clean_name = re.sub(r"\s+", " ", str(name)).strip()
    if len(clean_name) > max_chars:
        clean_name = clean_name[: max_chars - 1].rstrip() + "…"
    return "\n".join(textwrap.wrap(clean_name, width=width, break_long_words=False))


LABEL_TRANSLATION_RULES = [
    {
        "patterns": [r"pmt pangan lokal.*apakah dihabiskan"],
        "id": "PMT pangan lokal (Apakah dihabiskan)",
        "en": "Local Food Supplementary Feeding (Consumed completely?)",
    },
    {
        "patterns": [r"recall mengenai makanan padat.*kue pisang"],
        "id": "Recall mengenai Makanan padat, setengah padat, makana lumat lainnya termasuk kue-kue seperti kue pisang, cucur, pancong, permen",
        "en": "Recall of solid, semi-solid, or soft foods, including snacks such as banana cake, cucur, pancong, and candies.",
    },
    {
        "patterns": [r"\bwaz\b", r"waz underweight"],
        "id": "Berat Badan Menurut Umur (BB/U)",
        "en": "Weight-for-Age Z-score (WAZ)",
    },
    {
        "patterns": [r"\bbaz\b", r"baz overweight"],
        "id": "Indeks Massa Tubuh menurut Umur (IMT/U)",
        "en": "Body Mass Index-for-Age Z-score (BAZ)",
    },
    {
        "patterns": [r"\bwhz\b", r"whz wasting"],
        "id": "Berat Badan menurut Panjang Badan (BB/PB)",
        "en": "Weight-for-Height Z-score (WHZ)",
    },
    {
        "patterns": [r"diare", r"bab lebih"],
        "id": "Riwayat Penyakit Baduta seperti gejala Diare (BAB lebih cair dan >3x/hari tidak bercampur darah)",
        "en": "History of symptoms like diarrheal in children under two years (passing loose stools more than three times per day without blood)",
    },
    {
        "patterns": [r"\bttd\b", r"tablet tambah darah"],
        "id": "Kepatuhan mengonsumsi Tablet Tambah Darah (TTD) selama kehamilan",
        "en": "Adherence to iron and folic acid (IFA) supplementation during pregnancy",
    },
    {
        "patterns": [r"lokasi edukasi"],
        "id": "Lokasi edukasi",
        "en": "Education location",
    },
    {
        "patterns": [r"lokasi.*ukur.*lila", r"ukur lila.*lokasi"],
        "id": "Lokasi Ukur LiLA",
        "en": "Location of Mid-Upper Arm Circumference (MUAC) measurement",
    },
    {
        "patterns": [r"lokasi.*vit", r"pemberian vit"],
        "id": "Lokasi Pemberian Vit A",
        "en": "Location of vitamin A supplementation",
    },
    {
        "patterns": [r"berat badan baduta saat pengukuran", r"\bbb\s*\(kg\)"],
        "id": "Berat Badan Baduta (kg)",
        "en": "Body weight of children (kg)",
    },
    {
        "patterns": [r"\be6b\.2\b", r"anc.*4-6", r"4-6 bulan"],
        "id": "Frekuensi Pemeriksaan Antenatal Care (ANC) pada usia kehamilan 4-6 bulan (minimal 2x)",
        "en": "Frequency of antenatal care (ANC) visits during the second trimester (4-6 months; at least two visits)",
    },
    {
        "patterns": [r"\be6c\.1\b"],
        "id": "Frekuensi ANC pada nakes lain usia kehamilan 0-3 bulan",
        "en": "Frequency of ANC visits with other health workers during 0-3 months of pregnancy",
    },
    {
        "patterns": [r"dimana anda sering melakukan pemeriksaan kehamilan"],
        "id": "Tempat yang sering digunakan untuk pemeriksaan kehamilan",
        "en": "Usual place for antenatal care visits",
    },
    {
        "patterns": [r"selain asi", r"mpasi", r"makanan.*minuman"],
        "id": "Perilaku Pemberian ASI & MPASI seperti makanan/minuman (cairan) yang diberikan selain ASI",
        "en": "Breastfeeding and complementary feeding practices, including foods or liquids provided in addition to breast milk",
    },
    {
        "patterns": [r"susu c5\.1", r"\bc5\.1\b"],
        "id": "Susu - frekuensi, jumlah pemberian, dan alasan tidak dihabiskan",
        "en": "Milk supplementary feeding - frequency, serving quantity, and reason not finished",
    },
    {
        "patterns": [r"\bc3\.2\b"],
        "id": "Biskuit Program - Jumlah/kali pemberian",
        "en": "Supplementary biscuit program - quantity per serving",
    },
    {
        "patterns": [r"\bc6\.4\b"],
        "id": "Kacang Hijau - Alasan tidak dihabiskan",
        "en": "Mung bean program - reason not finished",
    },
    {
        "patterns": [r"\bd16l\b", r"\bd16n\b", r"sayuran hijau", r"bayam", r"kangkung"],
        "id": "Konsumsi baduta terhadap sayuran hijau",
        "en": "Green vegetable consumption of children",
    },
    {
        "patterns": [r"\be8g\b", r"tes urin"],
        "id": "Riwayat Pemeriksaan Kehamilan seperti Tes Urin",
        "en": "History of antenatal screening, including urine testing",
    },
    {
        "patterns": [r"usia baduta", r"usia anak"],
        "id": "Usia Anak (bulan)",
        "en": "Child's age (months)",
    },
    {
        "patterns": [r"tb ibu", r"tinggi badan ibu"],
        "id": "Tinggi Badan Ibu saat Awal Kehamilan (cm)",
        "en": "Maternal height at the beginning of pregnancy (cm)",
    },
    {
        "patterns": [r"\btb\s*\(cm\)"],
        "id": "Tinggi Badan (cm)",
        "en": "Height (cm)",
    },
    {
        "patterns": [r"\blila\s*\(cm\)"],
        "id": "LiLA (cm)",
        "en": "MUAC (cm)",
    },
    {
        "patterns": [r"frekuensi timbang bb"],
        "id": "Frekuensi timbang BB",
        "en": "Frequency of body weight measurement",
    },
    {
        "patterns": [r"pmt pangan lokal"],
        "id": "PMT pangan lokal",
        "en": "Local food supplementary feeding",
    },
    {
        "patterns": [r"total skor"],
        "id": "Total Skor",
        "en": "Total score",
    },
    {
        "patterns": [r"frekuensi ukur lila"],
        "id": "Frekuensi Pengukuran LiLA Anak",
        "en": "Frequency of child MUAC measurement",
    },
    {
        "patterns": [r"ukur lila", r"pengukuran lila"],
        "id": "Pengukuran LiLA Anak (Ya/Tidak)",
        "en": "Mid-Upper Arm Circumference (MUAC) measurement (Yes/No) of Child",
    },
    {
        "patterns": [r"\be9a\.1\b"],
        "id": "Jumlah Konsumsi TTD Selama Masa Kehamilan (butir)",
        "en": "Number of Iron and Folic Acid (IFA) Tablets Consumed During Pregnancy (tablets)",
    },
]

CATEGORY_TRANSLATIONS = {
    "Empty": {"id": "Empty", "en": "Empty"},
    "Kode missing": {"id": "Kode missing", "en": "Missing code"},
    "Tidak berlaku": {"id": "Tidak berlaku", "en": "Not applicable"},
    "Tidak tahu": {"id": "Tidak tahu", "en": "Do not know"},
    "Tidak": {"id": "Tidak", "en": "No"},
    "Ya": {"id": "Ya", "en": "Yes"},
    "Laki-laki": {"id": "Laki-laki", "en": "Male"},
    "Perempuan": {"id": "Perempuan", "en": "Female"},
    "Rumah sakit": {"id": "Rumah sakit", "en": "Hospital"},
    "Klinik/Praktek Dokter/Bidan": {"id": "Klinik/Praktek Dokter/Bidan", "en": "Clinic/doctor/midwife practice"},
    "Puskesmas": {"id": "Puskesmas", "en": "Community health center"},
    "Kunjungan petugas/kader": {"id": "Kunjungan petugas/kader", "en": "Health worker/cadre visit"},
    "Pemantauan mandiri tercatat di buku KIA": {
        "id": "Pemantauan mandiri tercatat di buku KIA",
        "en": "Self-monitoring recorded in the maternal and child health book",
    },
    "Bubur nasi/nasi tim/lauk dihaluskan": {
        "id": "Bubur nasi/nasi tim/lauk dihaluskan",
        "en": "Rice porridge/soft rice/mashed side dishes",
    },
    "Ya, berdasarkan KIA/Dokumen lainnya": {
        "id": "Ya, berdasarkan KIA/Dokumen lainnya",
        "en": "Yes, based on MCH book/other documents",
    },
    "Klinik/Praktik Dokter/Bidan": {"id": "Klinik/Praktik Dokter/Bidan", "en": "Clinic/doctor/midwife practice"},
    "Rasanya tidak enak/kurang bervariasi": {
        "id": "Rasanya tidak enak/kurang bervariasi",
        "en": "Unpleasant taste/lack of variety",
    },
}

CATEGORY_PHRASE_TRANSLATIONS_EN = [
    ("Rasanya tidak enak/kurang bervariasi", "Unpleasant taste/lack of variety"),
    ("Klinik/Praktek Dokter/Bidan", "Clinic/doctor/midwife practice"),
    ("Klinik/Praktik Dokter/Bidan", "Clinic/doctor/midwife practice"),
    ("Puskesmas", "Community health center"),
    ("Kunjungan petugas/kader", "Health worker/cadre visit"),
    ("Rumah sakit", "Hospital"),
    ("Pemantauan mandiri tercatat di buku KIA", "Self-monitoring recorded in the maternal and child health book"),
    ("Bubur nasi/nasi tim/lauk dihaluskan", "Rice porridge/soft rice/mashed side dishes"),
    ("Ya, berdasarkan KIA/Dokumen lainnya", "Yes, based on MCH book/other documents"),
    ("Tidak", "No"),
    ("Ya", "Yes"),
]


def translate_category_label(value: str | None, language: str) -> str | None:
    if value is None:
        return None
    clean_value = re.sub(r"\s+", " ", str(value)).strip()
    translated_value = CATEGORY_TRANSLATIONS.get(clean_value, {}).get(language, clean_value)
    if language == "en":
        for source, target in CATEGORY_PHRASE_TRANSLATIONS_EN:
            translated_value = translated_value.replace(source, target)
    return translated_value


def translated_feature_label(label: str, language: str) -> str:
    """Translate chart labels using the Indonesian/English display names from Hasil 300.docx."""
    clean_label = re.sub(r"\s+", " ", str(label)).strip()
    feature_part, _, _ = clean_label.partition(" = ")
    search_text = feature_part.lower()
    for rule in LABEL_TRANSLATION_RULES:
        if any(re.search(pattern, search_text, flags=re.IGNORECASE) for pattern in rule["patterns"]):
            translated = rule.get(language) or feature_part
            return translated
    return feature_part


def remove_label_value_suffix(label: str) -> str:
    clean_label = re.sub(r"\s+", " ", str(label)).strip()
    feature_part, _, _ = clean_label.partition(" = ")
    return feature_part


def make_language_chart_labels(labels: list[str], language: str, width: int, max_chars: int) -> list[str]:
    translated = make_unique_display_names([translated_feature_label(label, language) for label in labels])
    return [chart_label(label, width=width, max_chars=max_chars) for label in translated]


def make_plain_chart_labels(labels: list[str], width: int, max_chars: int) -> list[str]:
    cleaned = make_unique_display_names([remove_label_value_suffix(label) for label in labels])
    return [chart_label(label, width=width, max_chars=max_chars) for label in cleaned]


def save_cross_validation_metrics_figure(results: pd.DataFrame, output_path: Path):
    """Save cross-validation mean and standard deviation as a publication-style table."""
    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "precision": "Precision",
        "recall": "Recall / sensitivity",
        "f1": "F1-score",
        "roc_auc": "AUC-ROC",
    }
    display_rows = [
        [
            metric_labels.get(str(row.metric), str(row.metric).replace("_", " ").title()),
            f"{float(row.mean):.3f}",
            f"{float(row.std):.3f}",
            f"{float(row.mean):.3f} ± {float(row.std):.3f}",
        ]
        for row in results.itertuples(index=False)
    ]

    fig_height = max(3.8, 1.65 + 0.52 * len(display_rows))
    fig, ax = plt.subplots(figsize=(9.6, fig_height), facecolor="white")
    ax.axis("off")
    ax.text(
        0.0,
        1.02,
        "Cross-validation performance of the XGBoost stunting model",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color="#454650",
    )
    table = ax.table(
        cellText=display_rows,
        colLabels=["Metric", "Mean", "Std. Dev.", "Mean ± Std. Dev."],
        colWidths=[0.37, 0.18, 0.18, 0.27],
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0.0, 0.13, 1.0, 0.79],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.35)

    for column_index in range(4):
        cell = table[(0, column_index)]
        cell.set_facecolor("#686A76")
        cell.set_edgecolor("#686A76")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")

    for row_index in range(1, len(display_rows) + 1):
        for column_index in range(4):
            cell = table[(row_index, column_index)]
            cell.set_edgecolor("#B7B8BE")
            cell.set_linewidth(0.7)
            cell.set_facecolor("#F4F4F6" if row_index % 2 == 0 else "white")
            if column_index == 0:
                cell.get_text().set_horizontalalignment("left")

    ax.text(
        0.0,
        0.035,
        "Nilai merupakan mean dan std dari 5-fold NESTED stratified CV (seleksi fitur + tuning diulang di setiap fold).",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#666872",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_confusion_matrix_figure(cm_values: np.ndarray, output_path: Path, threshold_label: str = "0.50"):
    """Save confusion matrix with count and total-percent annotations."""
    cm_values = np.asarray(cm_values)
    total = cm_values.sum()
    fig = plt.figure(figsize=(7.2, 6.0), facecolor="white")
    ax = fig.add_axes([0.24, 0.24, 0.50, 0.58])
    image = ax.imshow(cm_values, cmap="Blues", vmin=0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")

    ax.set_title(f"Confusion matrix (threshold = {threshold_label})", fontsize=12, fontweight="bold", pad=12, color="#4a4a4a")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nnon-high-risk", "Predicted\nhigh-risk"], fontsize=10)
    ax.set_yticklabels(["Actual\nnon-high-risk", "Actual\nhigh-risk"], fontsize=10)
    ax.tick_params(axis="both", length=0)

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = cm_values.max() / 2 if cm_values.size else 0
    for row in range(cm_values.shape[0]):
        for col in range(cm_values.shape[1]):
            count = int(cm_values[row, col])
            pct = (count / total * 100) if total else 0
            color = "white" if cm_values[row, col] > threshold else "#3d3d3d"
            ax.text(col, row, f"{count:,}\n({pct:.1f}%)", ha="center", va="center", fontsize=10, fontweight="bold", color=color)

    fig.text(0.08, 0.12, "FIGURE 7", fontsize=9, fontweight="bold", color="#5a5a5a")
    fig.text(0.08, 0.08, f"Confusion matrix pada threshold = {threshold_label}.", fontsize=10, color="#7a7a7a")
    fig.add_artist(plt.Rectangle((0.04, 0.04), 0.92, 0.90, fill=False, edgecolor="#9a9a9a", linewidth=0.8, transform=fig.transFigure))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def to_dense_array(matrix):
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def extract_positive_class_shap(shap_values):
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    if isinstance(shap_values, shap.Explanation):
        values = np.asarray(shap_values.values)
    else:
        values = np.asarray(shap_values)
    if values.ndim == 3:
        if values.shape[-1] == 2:
            return values[:, :, 1]
        if values.shape[0] == 2:
            return values[1, :, :]
    return values


def compute_shap_values(explainer, values):
    try:
        return explainer.shap_values(values, check_additivity=False)
    except TypeError:
        try:
            return explainer.shap_values(values)
        except Exception:
            return explainer(values)
    except Exception:
        try:
            return explainer(values, check_additivity=False)
        except TypeError:
            return explainer(values)


def build_shap_subgroup_performance(
    source_df: pd.DataFrame,
    test_index: pd.Index,
    y_true: pd.Series,
    y_prediction: np.ndarray,
    y_probability: np.ndarray,
    shap_features: pd.DataFrame,
    trained_cleaner: SurveyFeatureCleaner,
) -> tuple[pd.DataFrame, list[str]]:
    """Evaluate the model within groups derived from the Indonesian SHAP attributes."""
    evaluation = pd.DataFrame(
        {
            "actual": y_true.loc[test_index].astype(int).to_numpy(),
            "prediction": np.asarray(y_prediction, dtype=int),
            "probability": np.asarray(y_probability, dtype=float),
        },
        index=test_index,
    )
    prepared_source = trained_cleaner._prepare(source_df).loc[test_index]
    rows = []
    subgroup_order = []
    seen_raw_features = set()

    def append_group(subgroup_name: str, excel_column: str, raw_feature: str, rank: int, category_code: str, category_label: str, mask):
        group_evaluation = evaluation.loc[pd.Series(mask, index=test_index).fillna(False).to_numpy()]
        n_rows = len(group_evaluation)
        if n_rows:
            high_risk_percent = float(group_evaluation["actual"].mean() * 100)
            recall = float(recall_score(group_evaluation["actual"], group_evaluation["prediction"], pos_label=1, zero_division=0))
            precision = float(
                precision_score(group_evaluation["actual"], group_evaluation["prediction"], pos_label=1, zero_division=0)
            )
            f1 = float(f1_score(group_evaluation["actual"], group_evaluation["prediction"], pos_label=1, zero_division=0))
            auc_roc = (
                float(roc_auc_score(group_evaluation["actual"], group_evaluation["probability"]))
                if group_evaluation["actual"].nunique() == 2
                else np.nan
            )
        else:
            high_risk_percent = np.nan
            auc_roc = np.nan
            recall = np.nan
            precision = np.nan
            f1 = np.nan
        rows.append(
            {
                "shap_rank": int(rank),
                "subgroup": subgroup_name,
                "raw_feature": raw_feature,
                "excel_column": excel_column,
                "category_code": category_code,
                "category_label": category_label,
                "N": int(n_rows),
                "high_risk_percent": high_risk_percent,
                "auc_roc": auc_roc,
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }
        )

    for shap_rank, shap_row in enumerate(shap_features.itertuples(index=False), start=1):
        raw_feature = getattr(shap_row, "raw_feature", None) or transformed_to_raw_feature(
            shap_row.transformed_feature, trained_cleaner.selected_raw_features_
        )
        if raw_feature in seen_raw_features or raw_feature not in prepared_source.columns:
            continue
        seen_raw_features.add(raw_feature)
        excel_column = FEATURE_SOURCE_MAP.get(raw_feature, "")
        attribute_label = re.sub(
            r"\s+",
            " ",
            str(getattr(shap_row, "chart_label_id", translated_feature_label(feature_display_label(raw_feature), "id"))),
        ).strip()
        subgroup_name = f"{attribute_label} (Kolom {excel_column})" if excel_column else attribute_label
        subgroup_order.append(subgroup_name)
        values = prepared_source[raw_feature]

        if raw_feature in trained_cleaner.multiresponse_categories_:
            text_values = values.fillna("").astype(str)
            available_options = trained_cleaner.multiresponse_categories_[raw_feature]
            for option in available_options:
                mask = text_values.map(lambda value, option=option: option in {token.strip() for token in value.split(",")})
                category_label = subgroup_value_display_label(raw_feature, option, available_options)
                append_group(subgroup_name, excel_column, raw_feature, shap_rank, option, category_label, mask)
            continue

        if raw_feature in trained_cleaner.numeric_columns_:
            numeric_values = pd.to_numeric(values, errors="coerce")
            unique_count = numeric_values.nunique(dropna=True)
            if unique_count > 6:
                quantile_groups = pd.qcut(numeric_values, q=min(4, unique_count), duplicates="drop")
                for interval in quantile_groups.cat.categories:
                    left = f"{interval.left:.2f}".rstrip("0").rstrip(".")
                    right = f"{interval.right:.2f}".rstrip("0").rstrip(".")
                    append_group(
                        subgroup_name,
                        excel_column,
                        raw_feature,
                        shap_rank,
                        str(interval),
                        f"{left} – {right}",
                        quantile_groups.eq(interval),
                    )
            else:
                available_numeric_values = sorted(numeric_values.dropna().unique())
                for value in available_numeric_values:
                    code = f"{float(value):g}"
                    category_label = subgroup_value_display_label(raw_feature, code, available_numeric_values)
                    append_group(subgroup_name, excel_column, raw_feature, shap_rank, code, category_label, numeric_values.eq(value))
            continue

        normalized_values = values.map(lambda value: "" if pd.isna(value) else normalize_category_value(value))
        value_counts = normalized_values[normalized_values.ne("")].value_counts()
        selected_codes = value_counts.index.tolist()[:8]
        for code in selected_codes:
            category_label = subgroup_value_display_label(raw_feature, code, value_counts.index)
            append_group(subgroup_name, excel_column, raw_feature, shap_rank, code, category_label, normalized_values.eq(code))
        if len(value_counts) > len(selected_codes):
            append_group(
                subgroup_name,
                excel_column,
                raw_feature,
                shap_rank,
                "OTHER",
                "Kategori lainnya",
                normalized_values.ne("") & ~normalized_values.isin(selected_codes),
            )

    return pd.DataFrame(rows), subgroup_order


def save_subgroup_performance_figure(results: pd.DataFrame, subgroup_order: list[str], output_path: Path):
    headers = ["Subgroup", "N", "High-risk (%)", "AUC-ROC", "Recall", "Precision", "F1"]
    display_rows = []
    section_row_indices = []
    for subgroup_name in subgroup_order:
        section_row_indices.append(len(display_rows) + 1)
        display_rows.append([chart_label(subgroup_name, width=48, max_chars=150), "", "", "", "", "", ""])
        subgroup_results = results[results["subgroup"].eq(subgroup_name)]
        for row in subgroup_results.itertuples(index=False):
            metric = lambda value: "-" if pd.isna(value) else f"{value:.3f}"
            display_rows.append(
                [
                    f"   {row.category_label}",
                    f"{row.N:,}",
                    "-" if pd.isna(row.high_risk_percent) else f"{row.high_risk_percent:.1f}",
                    metric(row.auc_roc),
                    metric(row.recall),
                    metric(row.precision),
                    metric(row.f1),
                ]
            )

    fig_height = 1.75 + 0.44 * len(display_rows)
    fig, ax = plt.subplots(figsize=(12.8, fig_height), facecolor="white")
    ax.axis("off")
    ax.text(
        0.0,
        1.035,
        "TABLE 2  Subgroup performance of the XGBoost stunting model (atribut SHAP beeswarm Bahasa Indonesia)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#454650",
    )
    table = ax.table(
        cellText=display_rows,
        colLabels=headers,
        colWidths=[0.43, 0.075, 0.135, 0.10, 0.085, 0.10, 0.075],
        cellLoc="center",
        colLoc="center",
        loc="upper left",
        bbox=[0.0, 0.08, 1.0, 0.90],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.35)

    for column_index in range(len(headers)):
        cell = table[(0, column_index)]
        cell.set_facecolor("#686A76")
        cell.set_edgecolor("#686A76")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
        if column_index == 1:
            cell.get_text().set_fontstyle("italic")

    for row_index in range(1, len(display_rows) + 1):
        is_section = row_index in section_row_indices
        for column_index in range(len(headers)):
            cell = table[(row_index, column_index)]
            cell.set_edgecolor("#B7B8BE")
            cell.set_linewidth(0.7)
            cell.set_facecolor("#E7E8ED" if is_section else "white")
            if column_index == 0:
                cell.get_text().set_horizontalalignment("left")
            if is_section:
                cell.get_text().set_fontweight("bold")

    ax.text(
        0.0,
        0.015,
        "High-risk (%) = proporsi aktual stunting pada data uji. AUC-ROC ditampilkan '-' jika subkelompok hanya memiliki satu kelas.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#666872",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)




# ============================== MODEL ADAPTER (XGBOOST) ==============================
# Semua logika spesifik-algoritma dikumpulkan di blok ini. File CatBoost memakai blok
# adapter yang berbeda; sisanya identik.

MODEL_LIB_VERSION = xgboost.__version__

DEFAULT_HYPERPARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [2, 3, 4, 5],
    "model__learning_rate": [0.02, 0.05, 0.1],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 2, 5, 10],
    "model__reg_alpha": [0.0, 0.1, 1.0],
    "model__reg_lambda": [1.0, 5.0, 10.0],
}


def make_model(**hyperparams):
    params = {**DEFAULT_HYPERPARAMS, **hyperparams}
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )


def build_preprocessor():
    """Tanpa imputasi numerik: NaN diteruskan ke XGBoost (ditangani native).
    Kategori kosong menjadi kategori eksplisit 'Missing' sebelum one-hot."""
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", build_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", make_column_selector(dtype_include=np.number)),
            ("cat", categorical_transformer, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def class_balance_params(y_values) -> dict:
    """scale_pos_weight = n_negatif / n_positif, dihitung dari data latih fold ybs."""
    y_values = pd.Series(y_values)
    positives = int((y_values == 1).sum())
    negatives = int((y_values == 0).sum())
    ratio = float(negatives / positives) if positives else 1.0
    return {"model__scale_pos_weight": ratio}


def compute_model_shap(fitted_pipeline, X_raw: pd.DataFrame) -> np.ndarray:
    """SHAP kelas positif pada ruang fitur tertransformasi (one-hot)."""
    cleaner = fitted_pipeline.named_steps["feature_cleaner"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    transformed = to_dense_array(preprocessor.transform(cleaner.transform(X_raw))).astype(np.float32)
    explainer = shap.TreeExplainer(fitted_pipeline.named_steps["model"])
    return extract_positive_class_shap(compute_shap_values(explainer, transformed))


def analysis_float_frame(transformed_df: pd.DataFrame) -> pd.DataFrame:
    """Representasi float untuk korelasi/heatmap/beeswarm (one-hot sudah numerik)."""
    return transformed_df.astype(float)


# ============================ END MODEL ADAPTER (XGBOOST) ============================



# ================================== MAIN WORKFLOW ==================================

if not EXCEL_FILE.exists():
    raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}")

df_raw = pd.read_excel(EXCEL_FILE, sheet_name=0, dtype=object)
total_dataset_all = int(len(df_raw))
df = df_raw.copy()
df.columns = make_unique([clean_column_name(name, f"Column_{index_to_excel_column(i)}") for i, name in enumerate(df_raw.columns)])

A_INDEX = excel_column_to_index("A")
JQ_INDEX = excel_column_to_index("JQ")
JS_INDEX = excel_column_to_index("JS")
JT_INDEX = excel_column_to_index("JT")
if df.shape[1] <= JT_INDEX:
    raise ValueError("Dataset must contain target column JT.")

target_original = pd.to_numeric(df.iloc[:, JT_INDEX], errors="coerce")
valid_target_values = set(target_original.dropna().astype(int).unique())
if valid_target_values != {0, 1}:
    raise ValueError(f"Column JT must contain labels 0 and 1 only. Found: {sorted(valid_target_values)}")

# Pool kandidat = SEMUA kolom A..JS (bukan 8 kolom hard-coded seperti versi lama).
candidate_indices = list(range(A_INDEX, JS_INDEX + 1))
candidate_features = [df.columns[i] for i in candidate_indices]
FEATURE_SOURCE_MAP = {df.columns[i]: index_to_excel_column(i) for i in candidate_indices}
X = df.iloc[:, candidate_indices].copy()
y = target_original.astype(int).map({0: 1, 1: 0}).astype(int)

identity_drop_letters = {
    "A": "identity: baduta name",
    "B": "identity: baduta sequence number",
    "C": "identity: baduta code",
    "K": "identity: mother name",
    "L": "identity: NUIB",
    "M": "identity: mother code",
    "N": "identity: mother EL code",
    "O": "identity: FFQ or identifier code",
    "P": "identity/date: mother birth date; mother age is already available",
    "Y": "identity: father name",
    "Z": "location/identifier: RT",
    "AA": "location/identifier: RW",
    "AB": "location: desa/kelurahan",
    "AC": "location: puskesmas",
    "AD": "location: posyandu",
    "AE": "location: kota/kabupaten",
    "AF": "location: provinsi",
}
excluded_derived_letters = {
    "H": "anti-leakage: child length/height (with age and sex) mathematically determines the HAZ target",
    "JM": "excluded requested anthropometry: WHZ wasting",
    "JN": "excluded derived anthropometry: WHZ category",
    "JO": "excluded requested anthropometry: BAZ overweight",
    "JP": "excluded derived anthropometry: BAZ category",
    "JR": "excluded derived anthropometry: WAZ category",
    "JS": "excluded derived anthropometry: HAZ",
}
forced_drop_reasons = {}
for letter, reason in {**identity_drop_letters, **excluded_derived_letters}.items():
    idx = excel_column_to_index(letter)
    if idx in candidate_indices:
        forced_drop_reasons[df.columns[idx]] = reason

EXCLUDED_ATTRIBUTE_PATTERNS = {
    r"\bnama\b": "excluded identity attribute: name",
    r"\btanggal\b": "excluded date attribute",
    r"\bno\.?\b|\bnomor\b": "excluded identifier attribute: number",
    r"\bkode\b": "excluded identifier attribute: code",
    r"\bnik\b|\bnuib\b|\bid\b": "excluded identifier attribute",
    r"\bffq\b": "excluded identifier/code attribute",
    r"\brt\b|\brw\b": "excluded location attribute: RT/RW",
    r"\bdesa\b|\bkelurahan\b": "excluded location attribute: village",
    r"\bpuskesmas\b": "excluded location attribute: puskesmas",
    r"\bposyandu\b": "excluded location attribute: posyandu",
    r"\bkota\b|\bkabupaten\b": "excluded location attribute: city/regency",
    r"\bprovinsi\b": "excluded location attribute: province",
    r"\brand\b": "excluded sampling artifact: RAND column",
}
for column in candidate_features:
    normalized_column = re.sub(r"[_\n\r\t./()-]+", " ", str(column).lower())
    normalized_column = re.sub(r"\s+", " ", normalized_column).strip()
    for pattern, reason in EXCLUDED_ATTRIBUTE_PATTERNS.items():
        if re.search(pattern, normalized_column):
            forced_drop_reasons.setdefault(column, reason)
            break

included_anthropometry_columns = {"JQ": df.columns[JQ_INDEX]}
for letter, feature_name in included_anthropometry_columns.items():
    if feature_name in forced_drop_reasons:
        raise ValueError(f"{letter} should be included in this model, but it is marked for drop: {feature_name}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

METRIC_COLUMNS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]


def make_full_pipeline(drop_reasons: dict) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "feature_cleaner",
                SurveyFeatureCleaner(
                    forced_drop_reasons=drop_reasons,
                    missing_threshold=0.90,
                    special_code_map=infer_special_code_map(),
                ),
            ),
            ("preprocessor", build_preprocessor()),
            ("model", make_model()),
        ]
    )


def transformed_frame(fitted_pipeline: Pipeline, X_raw: pd.DataFrame) -> pd.DataFrame:
    cleaner = fitted_pipeline.named_steps["feature_cleaner"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    data = preprocessor.transform(cleaner.transform(X_raw))
    names = list(preprocessor.get_feature_names_out())
    if isinstance(data, pd.DataFrame):
        data = data.copy()
        data.columns = names
        return data
    return pd.DataFrame(to_dense_array(data), columns=names, index=X_raw.index)


def select_top_raw_features(X_subset: pd.DataFrame, y_subset: pd.Series, drop_reasons: dict, top_count: int = TOP_FEATURE_COUNT) -> list[str]:
    """Seleksi fitur HANYA dari data yang diberikan (anti-leakage): fit model default
    pada seluruh pool, agregasi mean|SHAP| per fitur mentah, ambil top-N."""
    selection_pipeline = make_full_pipeline(drop_reasons)
    selection_pipeline.set_params(**class_balance_params(y_subset))
    selection_pipeline.fit(X_subset, y_subset)
    cleaner = selection_pipeline.named_steps["feature_cleaner"]
    names = list(selection_pipeline.named_steps["preprocessor"].get_feature_names_out())
    shap_matrix = compute_model_shap(selection_pipeline, X_subset)
    mean_abs = np.abs(np.asarray(shap_matrix)).mean(axis=0)
    aggregated: dict[str, float] = {}
    for name, value in zip(names, mean_abs):
        raw_name = transformed_to_raw_feature(name, cleaner.selected_raw_features_)
        aggregated[raw_name] = aggregated.get(raw_name, 0.0) + float(value)
    ranked = sorted(aggregated.items(), key=lambda item: -item[1])
    return [raw_name for raw_name, _ in ranked[:top_count]]


def build_search(y_subset: pd.Series, drop_reasons: dict) -> RandomizedSearchCV:
    pipeline_for_search = make_full_pipeline(drop_reasons)
    pipeline_for_search.set_params(**class_balance_params(y_subset))
    return RandomizedSearchCV(
        pipeline_for_search,
        PARAM_DISTRIBUTIONS,
        n_iter=N_SEARCH_ITER,
        cv=StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
    )


def evaluate_predictions(y_true, y_prediction, y_probability) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_prediction)),
        "precision": float(precision_score(y_true, y_prediction, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_prediction, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_prediction, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_probability)) if pd.Series(y_true).nunique() == 2 else float("nan"),
    }


def run_nested_cv(X_all: pd.DataFrame, y_all: pd.Series, drop_reasons: dict, scenario_label: str):
    """Nested CV yang jujur: seleksi fitur + tuning diulang di DALAM setiap outer fold."""
    outer_cv = StratifiedKFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    fold_details = []
    for fold_number, (train_idx, test_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
        X_fold_train, y_fold_train = X_all.iloc[train_idx], y_all.iloc[train_idx]
        X_fold_test, y_fold_test = X_all.iloc[test_idx], y_all.iloc[test_idx]
        top_features = select_top_raw_features(X_fold_train, y_fold_train, drop_reasons)
        search = build_search(y_fold_train, drop_reasons)
        search.fit(X_fold_train[top_features], y_fold_train)
        fold_proba = search.predict_proba(X_fold_test[top_features])[:, 1]
        fold_pred = (fold_proba >= 0.5).astype(int)
        metrics = evaluate_predictions(y_fold_test, fold_pred, fold_proba)
        fold_rows.append({"fold": fold_number, **metrics})
        fold_details.append(
            {
                "fold": fold_number,
                "scenario": scenario_label,
                "selected_features": [
                    {"excel_column": FEATURE_SOURCE_MAP.get(feature, ""), "raw_feature": feature, "display_label": feature_display_label(feature)}
                    for feature in top_features
                ],
                "best_params": {key.replace("model__", ""): json_ready(value) for key, value in search.best_params_.items()},
                "inner_cv_best_roc_auc": float(search.best_score_),
            }
        )
        print(f"[{scenario_label}] outer fold {fold_number}/{OUTER_CV_SPLITS} selesai: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    metrics_df = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(
        [{"metric": metric, "mean": float(metrics_df[metric].mean()), "std": float(metrics_df[metric].std(ddof=0))} for metric in METRIC_COLUMNS]
    )
    return summary, metrics_df, fold_details


print(f"=== {MODEL_NAME} Data600 fixed pipeline ===")
print(f"Pool kandidat: {len(candidate_features)} kolom, forced drop: {len(forced_drop_reasons)} kolom")
print(f"Nested CV: {OUTER_CV_SPLITS} outer x {INNER_CV_SPLITS} inner, RandomizedSearch n_iter={N_SEARCH_ITER}")

# --- Nested CV (skenario utama: skrining, WAZ disertakan) --------------------------
cv_summary, cv_fold_metrics, cv_fold_details = run_nested_cv(X_train, y_train, forced_drop_reasons, "screening_with_waz")

# --- Skenario pembanding: faktor risiko murni (tanpa WAZ & BB anak) ----------------
risk_factor_summary = None
risk_factor_details = None
if RUN_RISK_FACTOR_SCENARIO:
    risk_factor_drops = dict(forced_drop_reasons)
    for letter, reason in {
        "JQ": "risk-factor scenario: WAZ excluded (weight-derived anthropometry)",
        "G": "risk-factor scenario: child body weight excluded (component of WAZ)",
    }.items():
        idx = excel_column_to_index(letter)
        if idx in candidate_indices:
            risk_factor_drops[df.columns[idx]] = reason
    risk_factor_summary, risk_factor_fold_metrics, risk_factor_details = run_nested_cv(
        X_train, y_train, risk_factor_drops, "risk_factor_no_weight_anthropometry"
    )
    risk_factor_summary.to_csv(OUTPUT_DIR / "risk_factor_scenario_cv_metrics_data600.csv", index=False)
    risk_factor_fold_metrics.to_csv(OUTPUT_DIR / "risk_factor_scenario_cv_folds_data600.csv", index=False)

# --- Model final: seleksi + tuning pada train split, evaluasi pada test split ------
top_features_final = select_top_raw_features(X_train, y_train, forced_drop_reasons)
print("Fitur terpilih (train split): " + ", ".join(FEATURE_SOURCE_MAP.get(f, f) for f in top_features_final))

final_search = build_search(y_train, forced_drop_reasons)
training_started_at = datetime.now().isoformat()
final_search.fit(X_train[top_features_final], y_train)
training_finished_at = datetime.now().isoformat()
pipeline = final_search.best_estimator_
best_hyperparams = {key.replace("model__", ""): json_ready(value) for key, value in final_search.best_params_.items()}
print(f"Hyperparameter terbaik: {best_hyperparams}")

# Threshold tuning dari prediksi out-of-fold pada data latih (test set tidak disentuh)
oof_proba = cross_val_predict(
    clone(pipeline),
    X_train[top_features_final],
    y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    method="predict_proba",
    n_jobs=1,
)[:, 1]
threshold_grid = np.round(np.arange(0.10, 0.91, 0.01), 2)
threshold_scores = [
    {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y_train, (oof_proba >= threshold).astype(int))),
        "f1": float(f1_score(y_train, (oof_proba >= threshold).astype(int), pos_label=1, zero_division=0)),
    }
    for threshold in threshold_grid
]
best_balanced_threshold = max(threshold_scores, key=lambda row: row["balanced_accuracy"])["threshold"]
best_f1_threshold = max(threshold_scores, key=lambda row: row["f1"])["threshold"]

y_proba = pipeline.predict_proba(X_test[top_features_final])[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
y_pred_tuned = (y_proba >= best_balanced_threshold).astype(int)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cm_tuned = confusion_matrix(y_test, y_pred_tuned, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) else 0.0
tn_t, fp_t, fn_t, tp_t = cm_tuned.ravel()
specificity_tuned = tn_t / (tn_t + fp_t) if (tn_t + fp_t) else 0.0

test_metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    "precision_stunting": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
    "recall_sensitivity_stunting": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
    "specificity": float(specificity),
    "f1_stunting": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "pr_auc": float(average_precision_score(y_test, y_proba)),
}
test_metrics_tuned_threshold = {
    "threshold": float(best_balanced_threshold),
    "accuracy": float(accuracy_score(y_test, y_pred_tuned)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_tuned)),
    "precision_stunting": float(precision_score(y_test, y_pred_tuned, pos_label=1, zero_division=0)),
    "recall_sensitivity_stunting": float(recall_score(y_test, y_pred_tuned, pos_label=1, zero_division=0)),
    "specificity": float(specificity_tuned),
    "f1_stunting": float(f1_score(y_test, y_pred_tuned, pos_label=1, zero_division=0)),
}

trained_cleaner = pipeline.named_steps["feature_cleaner"]
trained_preprocessor = pipeline.named_steps["preprocessor"]
selected_features = trained_cleaner.get_selected_features()
dropped_columns = trained_cleaner.get_drop_audit()
dropped_columns["display_label"] = dropped_columns["feature"].map(feature_display_label) if not dropped_columns.empty else []
dropped_columns["excel_column"] = dropped_columns["feature"].map(FEATURE_SOURCE_MAP).fillna("") if not dropped_columns.empty else []

selected_features.to_csv(OUTPUT_DIR / "selected_features_data600.csv", index=False)
dropped_columns.to_csv(OUTPUT_DIR / "dropped_columns_data600.csv", index=False)
cv_summary.to_csv(OUTPUT_DIR / "cross_validation_metrics_data600.csv", index=False)
cv_fold_metrics.to_csv(OUTPUT_DIR / "cross_validation_fold_metrics_data600.csv", index=False)
save_cross_validation_metrics_figure(cv_summary, OUTPUT_DIR / "cross_validation_metrics_data600.png")

target_distribution = y.map({1: "Stunting", 0: "Normal"}).value_counts().reindex(["Normal", "Stunting"]).rename_axis("label").reset_index(name="count")
plt.figure(figsize=(6, 4))
sns.barplot(data=target_distribution, x="label", y="count", palette=["#4C78A8", "#F58518"])
plt.title("Distribusi Target Risiko Stunting")
plt.xlabel("")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution_data600.png", dpi=300, bbox_inches="tight")
plt.close()

save_confusion_matrix_figure(
    cm_tuned,
    OUTPUT_DIR / "confusion_matrix_data600.png",
    threshold_label=f"{best_balanced_threshold:.2f} (balanced-accuracy-optimal pada CV train)",
)

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {test_metrics['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {MODEL_NAME} Fixed Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve_data600.png", dpi=300, bbox_inches="tight")
plt.close()

precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, label=f"PR-AUC = {test_metrics['pr_auc']:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve - {MODEL_NAME} Fixed Model")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "precision_recall_curve_data600.png", dpi=300, bbox_inches="tight")
plt.close()

X_train_transformed_df = transformed_frame(pipeline, X_train)
transformed_feature_names = list(X_train_transformed_df.columns)
display_names = make_unique_display_names([clean_transformed_name(name, trained_cleaner.selected_raw_features_) for name in transformed_feature_names])
X_train_analysis_df = analysis_float_frame(X_train_transformed_df)
y_train_float = y_train.loc[X_train_analysis_df.index].astype(float)
corr_values = {}
for column in X_train_analysis_df.columns:
    values = X_train_analysis_df[column].astype(float)
    corr = values.corr(y_train_float) if values.nunique(dropna=True) > 1 else 0.0
    corr_values[column] = 0.0 if pd.isna(corr) else float(corr)
corr_df = (
    pd.Series(corr_values, name="correlation_with_stunting")
    .sort_values(key=lambda series: series.abs(), ascending=False)
    .head(TOP_FEATURE_COUNT)
    .reset_index()
    .rename(columns={"index": "feature"})
)
name_lookup = dict(zip(transformed_feature_names, display_names))
corr_df["display_feature"] = corr_df["feature"].map(name_lookup)
corr_df["chart_label"] = make_plain_chart_labels(corr_df["display_feature"].tolist(), width=26, max_chars=78)
corr_df["chart_label_id"] = make_language_chart_labels(corr_df["display_feature"].tolist(), "id", width=28, max_chars=92)
corr_df["chart_label_en"] = make_language_chart_labels(corr_df["display_feature"].tolist(), "en", width=30, max_chars=104)
corr_df.insert(0, "plot_id", [f"F{i}" for i in range(1, len(corr_df) + 1)])
corr_df.to_csv(OUTPUT_DIR / "top_8_correlation_features_data600.csv", index=False)
# Legacy filename retained so existing reports do not break; its content now has eight features.
corr_df.to_csv(OUTPUT_DIR / "top_10_correlation_features_data600.csv", index=False)

heatmap_df = X_train_analysis_df[corr_df["feature"].tolist()].copy()
heatmap_df["Target Stunting"] = y_train.loc[heatmap_df.index].values
heatmap_corr = heatmap_df.corr()


def save_heatmap(axis_labels: list[str], title: str, filename: str, left: float = 0.30, bottom: float = 0.32):
    heatmap_axis_labels = axis_labels + ["Target\nStunting"]
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        heatmap_corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        xticklabels=heatmap_axis_labels,
        yticklabels=heatmap_axis_labels,
        square=True,
        ax=ax,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, pad=14)
    ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("right")
    fig.subplots_adjust(left=left, right=0.94, top=0.90, bottom=bottom)
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


save_heatmap(
    corr_df["chart_label"].tolist(),
    f"Heatmap Korelasi 8 Fitur - {MODEL_NAME} Fixed Model",
    "heatmap_correlation_data600.png",
)
save_heatmap(
    corr_df["chart_label_id"].tolist(),
    f"Heatmap Korelasi 8 Fitur - {MODEL_NAME} Fixed Model (Indonesia)",
    "heatmap_correlation_data600_id.png",
    left=0.34,
    bottom=0.36,
)
save_heatmap(
    corr_df["chart_label_en"].tolist(),
    f"8-Feature Correlation Heatmap - {MODEL_NAME} Fixed Model (English)",
    "heatmap_correlation_data600_en.png",
    left=0.36,
    bottom=0.40,
)

X_test_transformed_df = transformed_frame(pipeline, X_test)
X_test_analysis = analysis_float_frame(X_test_transformed_df).to_numpy(dtype=np.float32)
positive_class_shap_values = np.asarray(compute_model_shap(pipeline, X_test))
mean_abs_shap = np.abs(positive_class_shap_values).mean(axis=0)
shap_importance = (
    pd.DataFrame(
        {
            "feature_index": np.arange(len(display_names)),
            "transformed_feature": transformed_feature_names,
            "feature": display_names,
            "mean_absolute_shap": mean_abs_shap,
        }
    )
    .sort_values("mean_absolute_shap", ascending=False)
    .reset_index(drop=True)
)
shap_importance["rank"] = np.arange(1, len(shap_importance) + 1)
shap_importance["plot_id"] = shap_importance["rank"].map(lambda rank: f"S{rank}")
shap_importance["chart_label"] = make_plain_chart_labels(shap_importance["feature"].tolist(), width=42, max_chars=110)
shap_importance["chart_label_id"] = make_language_chart_labels(shap_importance["feature"].tolist(), "id", width=46, max_chars=130)
shap_importance["chart_label_en"] = make_language_chart_labels(shap_importance["feature"].tolist(), "en", width=48, max_chars=145)
shap_importance["raw_feature"] = shap_importance["transformed_feature"].map(
    lambda value: transformed_to_raw_feature(value, trained_cleaner.selected_raw_features_)
)
shap_importance["excel_column"] = shap_importance["raw_feature"].map(FEATURE_SOURCE_MAP).fillna("")
shap_importance.to_csv(OUTPUT_DIR / "shap_feature_importance_data600.csv", index=False)

# SHAP_DISPLAY_SPECS dibangun DINAMIS dari fitur terpilih (bukan hard-coded),
# diurutkan berdasarkan agregat mean|SHAP| per fitur mentah.
raw_shap_order = (
    shap_importance.groupby("raw_feature")["mean_absolute_shap"].sum().sort_values(ascending=False).index.tolist()
)
SHAP_DISPLAY_SPECS = []
for raw_feature in raw_shap_order:
    display = feature_display_label(raw_feature)
    SHAP_DISPLAY_SPECS.append(
        {
            "excel_column": FEATURE_SOURCE_MAP.get(raw_feature, ""),
            "id": translated_feature_label(display, "id"),
            "en": translated_feature_label(display, "en"),
        }
    )

selected_shap_rows = []
missing_shap_columns = []
for display_rank, feature_spec in enumerate(SHAP_DISPLAY_SPECS, start=1):
    matches = shap_importance[shap_importance["excel_column"].eq(feature_spec["excel_column"])]
    if matches.empty:
        missing_shap_columns.append(feature_spec["excel_column"])
        continue
    selected_row = matches.iloc[0].copy()
    selected_row["importance_rank"] = int(selected_row["rank"])
    selected_row["rank"] = display_rank
    selected_row["plot_id"] = f"S{display_rank}"
    selected_row["chart_label"] = chart_label(feature_spec["id"], width=42, max_chars=110)
    selected_row["chart_label_id"] = chart_label(feature_spec["id"], width=46, max_chars=130)
    selected_row["chart_label_en"] = chart_label(feature_spec["en"], width=48, max_chars=145)
    selected_shap_rows.append(selected_row)

if missing_shap_columns:
    raise ValueError(f"Requested SHAP display columns were not found after preprocessing: {missing_shap_columns}")

top_shap = pd.DataFrame(selected_shap_rows).sort_values("rank").reset_index(drop=True)
top_shap.to_csv(OUTPUT_DIR / "shap_selected_features_data600.csv", index=False)

raw_shap_importance = (
    shap_importance.groupby("raw_feature", as_index=False)
    .agg(
        mean_absolute_shap=("mean_absolute_shap", "sum"),
        transformed_feature_count=("transformed_feature", "count"),
        excel_column=("excel_column", "first"),
    )
    .sort_values("mean_absolute_shap", ascending=False)
    .reset_index(drop=True)
)
raw_shap_importance["rank"] = np.arange(1, len(raw_shap_importance) + 1)
raw_shap_importance["display_label"] = raw_shap_importance["raw_feature"].map(feature_display_label)

transformed_corr_df = pd.DataFrame(
    {
        "transformed_feature": list(corr_values.keys()),
        "correlation_with_stunting": list(corr_values.values()),
    }
)
transformed_corr_df["raw_feature"] = transformed_corr_df["transformed_feature"].map(
    lambda value: transformed_to_raw_feature(value, trained_cleaner.selected_raw_features_)
)
raw_corr_rows = []
for raw_feature, group in transformed_corr_df.groupby("raw_feature"):
    best_idx = group["correlation_with_stunting"].abs().idxmax()
    raw_corr_rows.append(
        {
            "raw_feature": raw_feature,
            "correlation_with_stunting": float(group.loc[best_idx, "correlation_with_stunting"]),
        }
    )
raw_corr_df = pd.DataFrame(raw_corr_rows)
raw_shap_importance = raw_shap_importance.merge(raw_corr_df, on="raw_feature", how="left")
raw_shap_importance.to_csv(OUTPUT_DIR / "top_8_frontend_features_data600.csv", index=False)
# Legacy filename retained so existing frontend/report readers keep working.
raw_shap_importance.to_csv(OUTPUT_DIR / "top_10_frontend_features_data600.csv", index=False)

X_train_prepared_for_defaults = trained_cleaner._prepare(X_train)
default_input = {}
feature_schema = []
for raw_feature in trained_cleaner.selected_raw_features_:
    excel_column = FEATURE_SOURCE_MAP.get(raw_feature, "")
    series = X_train_prepared_for_defaults[raw_feature] if raw_feature in X_train_prepared_for_defaults.columns else pd.Series(dtype=object)
    non_missing = series.dropna()
    is_numeric = raw_feature in trained_cleaner.numeric_columns_
    if is_numeric:
        default_value = float(pd.to_numeric(non_missing, errors="coerce").median()) if len(non_missing) else None
        input_type = "number"
        options = []
    else:
        mode_values = non_missing.mode()
        default_value = json_ready(mode_values.iloc[0]) if len(mode_values) else None
        input_type = "select"
        value_counts = non_missing.astype(str).value_counts().head(25)
        options = [
            {
                "value": json_ready(value),
                "label": feature_value_display_label(raw_feature, value),
                "count": int(count),
            }
            for value, count in value_counts.items()
        ]
    default_input[raw_feature] = default_value
    default_input[excel_column] = default_value
    feature_schema.append(
        {
            "key": excel_column,
            "raw_feature": raw_feature,
            "excel_column": excel_column,
            "label": feature_display_label(raw_feature),
            "input_type": input_type,
            "default_value": default_value,
            "options": options,
        }
    )

# FORM_PREDICT_PARAMETER_SPECS dibangun dinamis mengikuti fitur terpilih.
schema_by_key_for_form = {item["key"]: item for item in feature_schema}
FORM_PREDICT_PARAMETER_SPECS = [
    {
        "key": spec["excel_column"],
        "label": spec["id"],
        "label_en": spec["en"],
        "input_type": schema_by_key_for_form.get(spec["excel_column"], {}).get("input_type", "text"),
        "required": True,
    }
    for spec in SHAP_DISPLAY_SPECS
]

shap_frontend_features = []
schema_by_raw = {item["raw_feature"]: item for item in feature_schema}
raw_corr_lookup = raw_shap_importance.set_index("raw_feature")["correlation_with_stunting"].to_dict()
for row in top_shap.to_dict(orient="records"):
    schema = schema_by_raw.get(row["raw_feature"], {})
    correlation_value = raw_corr_lookup.get(row["raw_feature"])
    shap_frontend_features.append(
        {
            "rank": int(row["rank"]),
            "key": schema.get("key", row.get("excel_column", "")),
            "raw_feature": row["raw_feature"],
            "excel_column": row.get("excel_column", ""),
            "transformed_feature": row["transformed_feature"],
            "shap_feature": row["feature"],
            "label": api_label(row["chart_label_id"]),
            "label_id": api_label(row["chart_label_id"]),
            "label_en": api_label(row["chart_label_en"]),
            "raw_label": schema.get("label", row["raw_feature"]),
            "input_type": schema.get("input_type", "text"),
            "default_value": schema.get("default_value"),
            "options": schema.get("options", []),
            "mean_absolute_shap": float(row["mean_absolute_shap"]),
            "correlation_with_stunting": None if pd.isna(correlation_value) else float(correlation_value),
        }
    )
top_10_frontend_features = shap_frontend_features
predict_parameters = build_form_predict_parameters(feature_schema)
api_feature_schema = [dict(parameter) for parameter in predict_parameters if parameter.get("used_in_model")]
expected_api_keys = [spec["excel_column"] for spec in SHAP_DISPLAY_SPECS]
actual_api_keys = [str(item.get("key")) for item in api_feature_schema]
if actual_api_keys != expected_api_keys:
    raise ValueError(f"Selected-feature API contract mismatch. Expected {expected_api_keys}, found {actual_api_keys}")

api_default_input = {}
for item in api_feature_schema:
    key = item["key"]
    api_default_input[key] = item.get("default_value")

with open(OUTPUT_DIR / "frontend_feature_schema_data600.json", "w", encoding="utf-8") as file:
    json.dump(
        {
            "target": {"excel_column": "JT", "name": str(df.columns[JT_INDEX])},
            "total_dataset": FRONTEND_TOTAL_DATASET,
            "top_10_features": top_10_frontend_features,
            "top_8_features": top_10_frontend_features,
            "shap_features": shap_frontend_features,
            "predict_parameters": predict_parameters,
            "required_features": api_feature_schema,
            "all_features": api_feature_schema,
            "all_model_features": feature_schema,
            "default_input": api_default_input,
        },
        file,
        indent=2,
        ensure_ascii=False,
    )

top_indices = top_shap["feature_index"].astype(int).to_numpy()
top_values = positive_class_shap_values[:, top_indices]
top_data = X_test_analysis[:, top_indices]

subgroup_performance, subgroup_order = build_shap_subgroup_performance(
    df,
    X_test.index,
    y_test,
    y_pred,
    y_proba,
    top_shap,
    trained_cleaner,
)
subgroup_performance.to_csv(OUTPUT_DIR / "subgroup_performance_shap_data600.csv", index=False)
subgroup_performance.to_csv(OUTPUT_DIR / "table_2_performance_xgboost_shap_id_data600.csv", index=False)
save_subgroup_performance_figure(
    subgroup_performance,
    subgroup_order,
    OUTPUT_DIR / "subgroup_performance_shap_data600.png",
)
save_subgroup_performance_figure(
    subgroup_performance,
    subgroup_order,
    OUTPUT_DIR / "table_2_performance_xgboost_shap_id_data600.png",
)


def save_shap_beeswarm(feature_names: list[str], title: str, filename: str):
    plt.figure()
    try:
        shap_explanation = shap.Explanation(values=top_values, data=top_data, feature_names=feature_names)
        shap.plots.beeswarm(shap_explanation, max_display=len(feature_names), show=False)
    except Exception:
        shap.summary_plot(
            top_values,
            pd.DataFrame(top_data, columns=feature_names, index=X_test.index),
            max_display=len(feature_names),
            show=False,
            plot_type="dot",
        )
    plt.title(title)
    plt.gcf().set_size_inches(13, max(7, 0.55 * len(feature_names) + 2))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


save_shap_beeswarm(
    top_shap["chart_label"].tolist(),
    f"SHAP Beeswarm Plot - {MODEL_NAME} Fixed Model",
    "shap_beeswarm_data600.png",
)
save_shap_beeswarm(
    top_shap["chart_label_id"].tolist(),
    f"SHAP Beeswarm Plot - {MODEL_NAME} Fixed Model (Indonesia)",
    "shap_beeswarm_data600_id.png",
)
save_shap_beeswarm(
    top_shap["chart_label_en"].tolist(),
    f"SHAP Beeswarm Plot - {MODEL_NAME} Fixed Model (English)",
    "shap_beeswarm_data600_en.png",
)

feature_importance = pd.DataFrame(
    {
        "feature": display_names,
        "importance": np.asarray(pipeline.named_steps["model"].feature_importances_, dtype=float),
    }
).sort_values("importance", ascending=False)
feature_importance.to_csv(OUTPUT_DIR / f"{MODEL_SLUG}_feature_importance_data600.csv", index=False)
# Legacy filename retained for report compatibility.
feature_importance.to_csv(OUTPUT_DIR / "xgboost_feature_importance_data600.csv", index=False)

pipeline_fixes = {
    "sentinel_fix": "Kode 77/88 tidak diterapkan pada kolom kontinu (>=15 nilai unik numerik); hanya 888 menjadi NaN",
    "no_imputation": "Numerik: NaN diteruskan ke model (native); kategorikal: kategori 'Missing' eksplisit",
    "class_balance": "scale_pos_weight/class weight dihitung dari data latih tiap fold",
    "honest_selection": f"Top-{TOP_FEATURE_COUNT} fitur dipilih dari pool {len(candidate_features)} kolom hanya dengan data latih (agregat mean|SHAP| per fitur mentah)",
    "nested_cv": f"{OUTER_CV_SPLITS}-fold outer, seleksi fitur + RandomizedSearchCV({N_SEARCH_ITER} kandidat, {INNER_CV_SPLITS}-fold inner) diulang per fold",
    "anti_leakage": "Kolom H (panjang/tinggi badan anak) dibuang karena menentukan HAZ target; identitas/lokasi/antropometri turunan juga dibuang",
    "threshold": "Ambang klasifikasi di-tuning dari prediksi out-of-fold data latih; confusion matrix memakai ambang balanced-accuracy-optimal",
}

metrics_payload = {
    "model_name": f"{MODEL_NAME} Stunting Data600 Fixed-Pipeline Model (honest nested CV, WAZ included)",
    "pipeline_fixes": pipeline_fixes,
    "target_column": {"excel_column": "JT", "name": str(df.columns[JT_INDEX])},
    "included_anthropometry_columns": {
        letter: {"excel_column": letter, "name": str(feature_name)}
        for letter, feature_name in included_anthropometry_columns.items()
    },
    "excluded_columns": excluded_derived_letters,
    "excluded_attribute_patterns": EXCLUDED_ATTRIBUTE_PATTERNS,
    "label_mapping": {"original": {"0": "Stunting", "1": "Normal"}, "model": {"1": "Stunting", "0": "Normal"}},
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "initial_feature_count": int(len(candidate_features)),
    "selected_feature_count": int(len(top_features_final)),
    "selected_features": [
        {"excel_column": FEATURE_SOURCE_MAP.get(feature, ""), "raw_feature": feature, "display_label": feature_display_label(feature)}
        for feature in top_features_final
    ],
    "final_raw_feature_count": int(len(trained_cleaner.selected_raw_features_)),
    "final_transformed_feature_count": int(len(transformed_feature_names)),
    "best_hyperparameters": best_hyperparams,
    "test_metrics": test_metrics,
    "test_metrics_tuned_threshold": test_metrics_tuned_threshold,
    "threshold_analysis": {
        "balanced_accuracy_optimal_threshold": float(best_balanced_threshold),
        "f1_optimal_threshold": float(best_f1_threshold),
        "note": "Ambang dihitung dari prediksi out-of-fold pada data latih; test set tidak digunakan untuk memilih ambang.",
    },
    "classification_report": classification_report(y_test, y_pred, labels=[0, 1], target_names=["Normal", "Stunting"], zero_division=0, output_dict=True),
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_tuned_threshold": cm_tuned.tolist(),
    "cross_validation": cv_summary.to_dict(orient="records"),
    "cross_validation_folds": cv_fold_metrics.to_dict(orient="records"),
    "nested_cv_details": cv_fold_details,
    "risk_factor_scenario": {
        "description": "Nested CV tanpa WAZ (JQ) dan BB anak (G): estimasi kemampuan prediksi murni dari kuesioner.",
        "cross_validation": risk_factor_summary.to_dict(orient="records") if risk_factor_summary is not None else None,
        "details": risk_factor_details,
    },
    "model_parameters": {key: json_ready(value) for key, value in pipeline.named_steps["model"].get_params().items()},
    "frontend": {
        "total_dataset": FRONTEND_TOTAL_DATASET,
        "model_input_rows": total_dataset_all,
        "total_features": int(len(top_10_frontend_features)),
        "model_name": MODEL_NAME,
        "accuracy_percent": round(float(test_metrics["accuracy"]) * 100, 2),
        "cv_accuracy_percent": round(float(cv_summary.loc[cv_summary["metric"] == "accuracy", "mean"].iloc[0]) * 100, 2)
        if "accuracy" in set(cv_summary["metric"])
        else None,
        "top_10_features": top_10_frontend_features,
    },
}
with open(OUTPUT_DIR / "metrics_data600.json", "w", encoding="utf-8") as file:
    json.dump(metrics_payload, file, indent=2, ensure_ascii=False, default=json_ready)

metadata = {
    "modeling_mode": f"{MODEL_NAME} fixed pipeline: honest nested CV, dynamic top-{TOP_FEATURE_COUNT} selection, WAZ (JQ) included, child length (H) excluded",
    "pipeline_fixes": pipeline_fixes,
    "target_name": str(df.columns[JT_INDEX]),
    "target_source": "Excel column JT",
    "included_anthropometry_columns": {
        letter: {"excel_column": letter, "name": str(feature_name)}
        for letter, feature_name in included_anthropometry_columns.items()
    },
    "excluded_derived_columns": excluded_derived_letters,
    "excluded_attribute_patterns": EXCLUDED_ATTRIBUTE_PATTERNS,
    "raw_input_features": [{"feature": feature, "display_label": feature_display_label(feature)} for feature in trained_cleaner.selected_raw_features_],
    "top_10_frontend_features": top_10_frontend_features,
    "frontend_feature_schema_path": str(OUTPUT_DIR / "frontend_feature_schema_data600.json"),
    "transformed_features": display_names,
    "python_version": sys.version,
    "platform": platform.platform(),
    "scikit_learn_version": sklearn.__version__,
    "model_library": MODEL_NAME,
    "model_library_version": MODEL_LIB_VERSION,
    "training_started_at": training_started_at,
    "training_finished_at": training_finished_at,
    "evaluation_metrics": test_metrics,
}
with open(OUTPUT_DIR / "model_metadata_data600.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2, ensure_ascii=False, default=json_ready)

joblib.dump(pipeline, OUTPUT_DIR / "stunting_prediction_pipeline_data600.pkl")

frontend_bundle = {
    "pipeline": pipeline,
    "model_name": MODEL_NAME,
    "modeling_mode": f"{MODEL_NAME} fixed-pipeline Data600 frontend bundle (dynamic selected-feature SHAP API contract)",
    "feature_contract": {
        "name": f"data600-fixed-{MODEL_SLUG}-shap{len(expected_api_keys)}-v1",
        "strict": True,
        "feature_count": len(expected_api_keys),
        "required_keys": expected_api_keys,
    },
    "target": {"excel_column": "JT", "name": str(df.columns[JT_INDEX])},
    "label_mapping": {"original": {"0": "Stunting", "1": "Normal"}, "model": {"1": "Stunting", "0": "Normal"}},
    "classification_threshold": {"default": 0.5, "balanced_accuracy_optimal": float(best_balanced_threshold), "f1_optimal": float(best_f1_threshold)},
    "default_input": api_default_input,
    "all_model_default_input": default_input,
    "feature_schema": api_feature_schema,
    "all_model_feature_schema": feature_schema,
    "required_features": api_feature_schema,
    "top_10_features": top_10_frontend_features,
    "top_8_features": top_10_frontend_features,
    "shap_features": shap_frontend_features,
    "predict_parameters": predict_parameters,
    "dashboard": {
        "total_dataset": FRONTEND_TOTAL_DATASET,
        "model_input_rows": total_dataset_all,
        "total_features": int(len(top_10_frontend_features)),
        "model_name": MODEL_NAME,
        "accuracy": float(test_metrics["accuracy"]),
        "accuracy_percent": round(float(test_metrics["accuracy"]) * 100, 2),
        "cv_accuracy": float(cv_summary.loc[cv_summary["metric"] == "accuracy", "mean"].iloc[0])
        if "accuracy" in set(cv_summary["metric"])
        else None,
        "cv_accuracy_percent": round(float(cv_summary.loc[cv_summary["metric"] == "accuracy", "mean"].iloc[0]) * 100, 2)
        if "accuracy" in set(cv_summary["metric"])
        else None,
        "top_10_feature_bar_chart": [
            {
                "rank": item["rank"],
                "key": item["key"],
                "label": item["label"],
                "score": item["mean_absolute_shap"],
                "correlation_with_stunting": item["correlation_with_stunting"],
            }
            for item in top_10_frontend_features
        ],
        "top_8_feature_bar_chart": [
            {
                "rank": item["rank"],
                "key": item["key"],
                "label": item["label"],
                "score": item["mean_absolute_shap"],
                "correlation_with_stunting": item["correlation_with_stunting"],
            }
            for item in top_10_frontend_features
        ],
    },
    "metrics": test_metrics,
    "created_at": datetime.now().isoformat(),
}
joblib.dump(frontend_bundle, OUTPUT_DIR / "stunting_frontend_bundle_data600.pkl")
joblib.dump(frontend_bundle, OUTPUT_DIR / "stunting_model_8_features_data600.pkl")

summary = pd.DataFrame(
    [
        {"output": path.name, "status": "OK" if path.exists() else "BELUM ADA"}
        for path in sorted(OUTPUT_DIR.glob("*_data600*.*"))
    ]
)
summary.to_csv(OUTPUT_DIR / "output_summary_data600.csv", index=False)

print(f"{MODEL_NAME} Data600 fixed-pipeline model selesai.")
print(f"Project directory: {PROJECT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print("Fitur terpilih:", ", ".join(FEATURE_SOURCE_MAP.get(f, f) for f in top_features_final))
print("Nested CV (mean±std):")
for row in cv_summary.itertuples(index=False):
    print(f"  {row.metric:20s} {row.mean:.3f} ± {row.std:.3f}")
if risk_factor_summary is not None:
    print("Skenario faktor risiko tanpa WAZ/BB (mean±std):")
    for row in risk_factor_summary.itertuples(index=False):
        print(f"  {row.metric:20s} {row.mean:.3f} ± {row.std:.3f}")
print("Test metrics (threshold 0.5):")
print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
print(f"Test metrics (threshold {best_balanced_threshold:.2f}, balanced-accuracy-optimal):")
print(json.dumps(test_metrics_tuned_threshold, indent=2, ensure_ascii=False))
