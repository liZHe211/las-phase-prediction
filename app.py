from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, hamming_loss
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
except ImportError:  # pragma: no cover
    ExtraTreesClassifier = None
    SimpleImputer = None
    GroupShuffleSplit = None
    accuracy_score = f1_score = hamming_loss = None
    Pipeline = None


st.set_page_config(
    page_title="基于真实数据驱动的 LAS 微晶玻璃晶相智能预测与目标晶相设计系统",
    page_icon="L",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
REAL_DATA_PATH = DATA_DIR / "data_merged.csv"
DB_PATH = DATA_DIR / "las_lab.db"

COMPOSITION_COLUMNS = [
    "Al2O3", "SiO2", "Li2O", "ZrO2", "P2O5", "Na2O", "TiO2", "K2O",
    "MgO", "ZnO", "BaO", "CaO", "SnO2", "Fe2O3", "As2O3", "Sb2O3",
    "B2O3", "SrO", "Ta2O5", "Nd2O3", "V2O5", "CeO2", "CoO", "Cl",
    "MnO2", "MoO3", "F", "La2O3", "Cr2O3",
]
PROCESS_COLUMNS = ["T_nucleation", "t1", "T_crystal", "t2"]
PROCESS_LABELS = {
    "T_nucleation": "成核温度",
    "t1": "成核时间",
    "T_crystal": "晶化温度",
    "t2": "晶化时间",
}
PHASE_COLUMNS = [
    "β-quartz s.s.", "β-spodumene", "Li2Si2O5(LD)", "Petalite",
    "Li2SiO3(LM)", "Keatite", "LiAlSi3O8", "LiAlSi2O6", "KMK",
    "Cristobalite", "Li4SiO4", "Mullite", "Tridymite",
]
PHASE_LABELS = {
    "β-quartz s.s.": "β-石英固溶体",
    "β-spodumene": "β-锂辉石",
    "Li2Si2O5(LD)": "二硅酸锂（LD）",
    "Petalite": "透锂长石",
    "Li2SiO3(LM)": "偏硅酸锂（LM）",
    "Keatite": "假硅灰石（Keatite）",
    "LiAlSi3O8": "锂铝硅酸盐（LiAlSi3O8）",
    "LiAlSi2O6": "锂辉石型 LiAlSi2O6",
    "KMK": "KMK 晶相",
    "Cristobalite": "方石英",
    "Li4SiO4": "正硅酸锂",
    "Mullite": "莫来石",
    "Tridymite": "鳞石英",
}
PHASE_COLUMN_ALIASES = {
    "β?quartz s.s": "β-quartz s.s.",
    "β-quartz s.s": "β-quartz s.s.",
    "β?spodumene": "β-spodumene",
}
MODEL_VERSION = "phase-v1.0"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root{--page:#e9eff0;--surface:#fbfcfa;--surface2:#edf4f2;--ink:#17272b;--muted:#60757a;--line:rgba(25,56,63,.13);--accent:#0f918b;--amber:#d8943e;--blue:#4a7180;--dark:#17343a;--shadow:0 14px 34px rgba(33,66,70,.08)}
        html,body{overflow-x:clip}.stApp,[data-testid="stAppViewContainer"]{background:var(--page);color:var(--ink)}
        [data-testid="stHeader"]{background:transparent}.block-container{max-width:1480px;padding:2rem 2.8rem 4rem}
        [data-testid="stSidebar"]{background:var(--dark);border-right:1px solid rgba(255,255,255,.1)}[data-testid="stSidebar"] *{color:#e8f3f2}[data-testid="stSidebar"] .stCaption{color:#a8c5c5!important}
        [data-testid="stSidebar"] [data-testid="stAlert"]{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14)}
        h1{font-size:clamp(1.85rem,3vw,2.7rem);line-height:1.08;letter-spacing:0;overflow-wrap:anywhere;color:var(--ink)}h2{line-height:1.16;overflow-wrap:anywhere;color:var(--ink)}h3{color:#39545a;line-height:1.25}p,label,[data-testid="stMarkdownContainer"]{color:var(--muted)}
        [data-testid="stMetric"]{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:1rem 1.1rem;box-shadow:var(--shadow);min-height:106px}[data-testid="stMetricLabel"]{color:var(--muted);font-size:.78rem}[data-testid="stMetricValue"]{color:var(--ink);font-size:1.72rem;font-variant-numeric:tabular-nums}
        [data-testid="stTabs"] [role="tablist"]{gap:.7rem;border-bottom:1px solid var(--line);margin-bottom:1.2rem;overflow-x:auto}[data-testid="stTabs"] button[role="tab"]{min-height:44px;padding:.55rem 1rem;border-radius:10px 10px 0 0;color:var(--muted);white-space:nowrap}[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent);background:rgba(15,145,139,.11);font-weight:700}
        .stButton>button,.stDownloadButton>button{min-height:44px;border-radius:10px;border:1px solid rgba(15,145,139,.32);background:var(--accent);color:#f4fffd!important;font-weight:700;box-shadow:0 8px 18px rgba(15,145,139,.18);white-space:nowrap;transition:transform .15s ease,box-shadow .15s ease}.stButton>button *, .stDownloadButton>button *{color:#f4fffd!important}.stButton>button:hover,.stDownloadButton>button:hover{background:#087b77;box-shadow:0 10px 22px rgba(15,145,139,.26);transform:translateY(-1px)}
        .stButton>button:focus-visible,.stDownloadButton>button:focus-visible,input:focus-visible,textarea:focus-visible{outline:3px solid rgba(216,148,62,.55);outline-offset:2px}[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow);background:var(--surface)}[data-testid="stAlert"]{border-radius:11px;border-width:1px}[data-testid="stFileUploader"]{background:var(--surface2);border:1px dashed rgba(15,145,139,.42);border-radius:12px;padding:.35rem}
        .lab-hero{background:var(--dark);border-radius:16px;padding:1.55rem 1.7rem;color:#effbf8;box-shadow:0 18px 36px rgba(23,52,58,.18);margin:.5rem 0 1.4rem}.lab-hero h2,.lab-hero p{color:#effbf8;margin:.1rem 0}.lab-hero p{opacity:.78}.eyebrow{color:#f1bc6c!important;font-size:.74rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}.step-card{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:1rem 1.1rem;height:176px;min-height:176px;box-sizing:border-box;display:flex;flex-direction:column;box-shadow:var(--shadow)}.step-card strong{color:var(--accent);font-size:.78rem}.step-card h4{margin:.35rem 0;color:var(--ink)}.step-card p{font-size:.84rem;line-height:1.45;margin:0;min-height:2.9em}
        .notice-strip{background:var(--surface2);border-left:4px solid var(--accent);padding:.8rem 1rem;border-radius:8px;margin:.8rem 0 1.1rem}.notice-strip strong{color:var(--ink)}
        @media(max-width:768px){.block-container{padding:1rem .8rem 3rem}h1{font-size:1.75rem}[data-testid="stTabs"] button[role="tab"]{min-width:98px}[data-testid="stMetric"]{min-height:92px}.lab-hero{padding:1.2rem;border-radius:13px}.step-card{height:auto;min-height:150px}}
        @media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;transition-duration:.01ms!important}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS samples(
                sample_id TEXT PRIMARY KEY,
                batch_name TEXT,
                sample_name TEXT,
                source TEXT,
                created_at TEXT,
                status TEXT,
                notes TEXT
            );
            CREATE TABLE IF NOT EXISTS sample_inputs(
                sample_id TEXT PRIMARY KEY,
                input_json TEXT NOT NULL,
                image_path TEXT,
                quality_status TEXT,
                FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
            );
            CREATE TABLE IF NOT EXISTS phase_predictions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT,
                model_name TEXT,
                model_version TEXT,
                predicted_json TEXT NOT NULL,
                created_at TEXT,
                FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
            );
            CREATE TABLE IF NOT EXISTS audit_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                action TEXT,
                created_at TEXT,
                detail TEXT
            );
            """
        )


def log_action(action: str, detail: str, source: str = "真实晶相数据") -> None:
    with db_connect() as conn:
        conn.execute(
            "INSERT INTO audit_logs(source,action,created_at,detail) VALUES(?,?,?,?)",
            (source, action, now_text(), detail),
        )


def read_table(query: str, params: tuple = ()) -> pd.DataFrame:
    with db_connect() as conn:
        return pd.read_sql_query(query, conn, params=params)


@st.cache_data
def load_real_data() -> pd.DataFrame:
    try:
        data = pd.read_csv(REAL_DATA_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        data = pd.read_csv(REAL_DATA_PATH, encoding="gb18030")
    rename_map = {
        alias: canonical
        for alias, canonical in PHASE_COLUMN_ALIASES.items()
        if alias in data.columns and canonical not in data.columns
    }
    return data.rename(columns=rename_map)


def is_phase_dataset(data: pd.DataFrame) -> bool:
    required = set(COMPOSITION_COLUMNS + PROCESS_COLUMNS)
    return required.issubset(data.columns) and bool(set(PHASE_COLUMNS).intersection(data.columns))


def phase_target_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in PHASE_COLUMNS if column in data.columns]


def phase_display_name(column: str) -> str:
    return PHASE_LABELS.get(column, column)


def display_phase_columns(data: pd.DataFrame) -> dict[str, str]:
    return {column: phase_display_name(column) for column in phase_target_columns(data)}


def composition_sum(data: pd.DataFrame) -> pd.Series:
    return data[COMPOSITION_COLUMNS].apply(pd.to_numeric, errors="coerce").sum(axis=1)


def process_available(data: pd.DataFrame) -> pd.Series:
    return data[PROCESS_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).abs().sum(axis=1).gt(0)


def phase_feature_frame(data: pd.DataFrame, include_process: bool = True) -> pd.DataFrame:
    columns = COMPOSITION_COLUMNS + (PROCESS_COLUMNS if include_process else [])
    frame = data[columns].apply(pd.to_numeric, errors="coerce").copy()
    if include_process:
        frame["process_available"] = process_available(data).astype(float).to_numpy()
        frame[PROCESS_COLUMNS] = frame[PROCESS_COLUMNS].replace(0, np.nan)
    return frame


def phase_target_frame(data: pd.DataFrame) -> pd.DataFrame:
    targets = phase_target_columns(data)
    return data[targets].apply(pd.to_numeric, errors="coerce").fillna(0).ge(.5).astype(int)


def model_pipeline() -> Pipeline | None:
    if Pipeline is None or SimpleImputer is None or ExtraTreesClassifier is None:
        return None
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=320,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def group_key(data: pd.DataFrame) -> pd.Series:
    return data[COMPOSITION_COLUMNS].round(8).astype(str).agg("|".join, axis=1)


def predict_probability_matrix(model: Pipeline, features: pd.DataFrame, target_count: int) -> np.ndarray:
    probabilities = model.predict_proba(features)
    if not isinstance(probabilities, list):
        probabilities = [probabilities]
    estimator = model.named_steps["model"]
    classes = estimator.classes_
    matrix = np.zeros((len(features), target_count), dtype=float)
    for index in range(target_count):
        probs = np.asarray(probabilities[index])
        phase_classes = np.asarray(classes[index] if isinstance(classes, list) else classes)
        if probs.ndim == 1:
            matrix[:, index] = probs
        elif probs.shape[1] == 1:
            matrix[:, index] = 1.0 if phase_classes[0] == 1 else 0.0
        else:
            one_class = np.flatnonzero(phase_classes == 1)
            matrix[:, index] = probs[:, int(one_class[0])] if len(one_class) else 0.0
    return matrix


@st.cache_resource
def train_phase_models(data: pd.DataFrame):
    if not is_phase_dataset(data) or GroupShuffleSplit is None:
        return None
    targets = phase_target_columns(data)
    y = phase_target_frame(data)
    groups = group_key(data)
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(data, y, groups)
    )
    model_specs = {
        "成分基线": COMPOSITION_COLUMNS,
        "成分+热处理工艺": COMPOSITION_COLUMNS + PROCESS_COLUMNS + ["process_available"],
    }
    artifacts = {}
    metric_rows = []
    for name, feature_columns in model_specs.items():
        include_process = name != "成分基线"
        x = phase_feature_frame(data, include_process=include_process)
        model = model_pipeline()
        if model is None:
            return None
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(x.iloc[test_idx])
        metric_rows.append(
            {
                "模型": name,
                "输入特征": "29个成分" if not include_process else "29个成分 + 4个热处理字段",
                "完全匹配率": accuracy_score(y.iloc[test_idx], pred),
                "宏平均 F1": f1_score(y.iloc[test_idx], pred, average="macro", zero_division=0),
                "微平均 F1": f1_score(y.iloc[test_idx], pred, average="micro", zero_division=0),
                "标签汉明损失": hamming_loss(y.iloc[test_idx], pred),
            }
        )
        artifacts[name] = {
            "model": model,
            "features": feature_columns,
            "include_process": include_process,
            "pred": pred,
            "y_test": y.iloc[test_idx].copy(),
            "test_idx": test_idx,
        }
    full = artifacts["成分+热处理工艺"]
    full_prob = predict_probability_matrix(full["model"], phase_feature_frame(data, True).iloc[test_idx], len(targets))
    full_per_phase = pd.DataFrame(
        {
            "晶相": [phase_display_name(target) for target in targets],
            "原始字段": targets,
            "F1": f1_score(full["y_test"], full["pred"], average=None, zero_division=0),
            "测试集样本数": full["y_test"].sum().to_numpy(),
            "测试集平均预测概率": full_prob.mean(axis=0),
        }
    ).sort_values("F1", ascending=False)
    return {
        "models": artifacts,
        "metrics": pd.DataFrame(metric_rows),
        "per_phase": full_per_phase,
        "targets": targets,
        "train_count": len(train_idx),
        "test_count": len(test_idx),
    }


def feature_importance_table(artifact: dict) -> pd.DataFrame:
    model = artifact["model"]
    imputer = model.named_steps["imputer"]
    estimator = model.named_steps["model"]
    names = imputer.get_feature_names_out(artifact["features"])
    return pd.DataFrame({"特征": names, "重要性": estimator.feature_importances_}).sort_values("重要性", ascending=False)


def default_input_row(data: pd.DataFrame) -> pd.Series:
    valid = data[process_available(data)].copy()
    if valid.empty:
        valid = data.copy()
    valid_order = valid[(valid["T_nucleation"] > 0) & (valid["T_crystal"] > 0) & (valid["T_nucleation"] < valid["T_crystal"])]
    if not valid_order.empty:
        valid = valid_order
    targets = phase_target_columns(data)
    return valid.loc[valid[targets].sum(axis=1).idxmax()]


def render_input_fields(data: pd.DataFrame, key_prefix: str) -> dict[str, float]:
    defaults = default_input_row(data)
    values: dict[str, float] = {}
    st.markdown("#### 成分输入")
    st.caption("输入值沿用原始数据表的计量方式；当前文件未提供单位说明，请在正式实验前向数据提供者确认。")
    with st.expander("展开 29 个氧化物成分", expanded=True):
        columns = st.columns(3)
        for index, column in enumerate(COMPOSITION_COLUMNS):
            default_value = pd.to_numeric(defaults[column], errors="coerce")
            default = float(default_value) if pd.notna(default_value) else 0.0
            values[column] = columns[index % 3].number_input(
                column,
                min_value=0.0,
                max_value=100.0,
                value=max(0.0, min(100.0, default)),
                step=0.01,
                format="%.4f",
                key=f"{key_prefix}_{column}",
            )
    st.markdown("#### 热处理工艺")
    columns = st.columns(4)
    for index, column in enumerate(PROCESS_COLUMNS):
        observed = pd.to_numeric(data[column], errors="coerce")
        positive = observed[observed > 0]
        fallback = float(positive.median()) if not positive.empty else 0.0
        default_raw = pd.to_numeric(defaults[column], errors="coerce")
        default = float(default_raw) if pd.notna(default_raw) and float(default_raw) > 0 else fallback
        maximum = max(float(positive.max()) * 1.5 if not positive.empty else 1.0, default + 1.0, 1.0)
        step = 1.0 if column.startswith("T_") else 0.25
        values[column] = columns[index].number_input(
            PROCESS_LABELS[column],
            min_value=0.0,
            max_value=maximum,
            value=min(default, maximum),
            step=step,
            format="%.2f",
            key=f"{key_prefix}_{column}",
            help=f"原始字段：{column}。输入 0 表示未提供该工艺条件。",
        )
    return values


def validate_input(values: dict[str, float], data: pd.DataFrame) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    total = sum(values[column] for column in COMPOSITION_COLUMNS)
    if not 98 <= total <= 102:
        errors.append(f"成分总和为 {total:.3f}%，请调整到 98% 至 102% 范围内。")
    if values["T_nucleation"] > 0 and values["T_crystal"] > 0 and values["T_nucleation"] >= values["T_crystal"]:
        errors.append("工艺顺序不成立：成核温度必须低于晶化温度。")
    if sum(values[column] for column in PROCESS_COLUMNS) == 0:
        warnings.append("没有输入热处理条件，模型将按数据集中的缺失工艺方式处理。")
    for column in COMPOSITION_COLUMNS:
        low = float(pd.to_numeric(data[column], errors="coerce").min())
        high = float(pd.to_numeric(data[column], errors="coerce").max())
        if values[column] < low or values[column] > high:
            warnings.append(f"{column}={values[column]:.3f} 超出当前数据范围 [{low:.3f}, {high:.3f}]。")
    for column in PROCESS_COLUMNS:
        positive = pd.to_numeric(data[column], errors="coerce")
        positive = positive[positive > 0]
        if values[column] > 0 and not positive.empty and (values[column] < positive.min() or values[column] > positive.max()):
            warnings.append(f"{PROCESS_LABELS[column]}超出已提供的非零数据范围。")
    return errors, warnings


def save_phase_record(
    sample_id: str,
    batch_name: str,
    sample_name: str,
    source: str,
    notes: str,
    values: dict[str, float],
    probabilities: dict[str, float],
    predicted: dict[str, int],
) -> None:
    payload = {
        phase: {"probability": round(float(probabilities[phase]), 6), "predicted": int(predicted[phase])}
        for phase in PHASE_COLUMNS
        if phase in probabilities
    }
    with db_connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO samples VALUES(?,?,?,?,?,?,?)",
            (sample_id, batch_name, sample_name, source, now_text(), "已完成晶相预测", notes),
        )
        conn.execute(
            "INSERT OR REPLACE INTO sample_inputs VALUES(?,?,?,?)",
            (sample_id, json.dumps(values, ensure_ascii=False), "", "通过"),
        )
        conn.execute("DELETE FROM phase_predictions WHERE sample_id=?", (sample_id,))
        conn.execute(
            "INSERT INTO phase_predictions(sample_id,model_name,model_version,predicted_json,created_at) VALUES(?,?,?,?,?)",
            (sample_id, "成分+热处理工艺", MODEL_VERSION, json.dumps(payload, ensure_ascii=False), now_text()),
        )
    log_action("保存晶相预测", sample_id, source)


def process_slider_spec(data: pd.DataFrame, column: str) -> tuple[float | int, float | int, tuple[float | int, float | int], float]:
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    positive = values[values > 0]
    if positive.empty:
        positive = values
    low_raw, high_raw = float(positive.min()), float(positive.max())
    if column.startswith("T_"):
        low, high = int(math.floor(low_raw)), int(math.ceil(high_raw))
        if high <= low:
            high = low + 1
        q1, q3 = np.nanpercentile(positive, [25, 75])
        default = (max(low, int(math.floor(q1))), min(high, int(math.ceil(q3))))
        if default[1] <= default[0]:
            default = (low, high)
        return low, high, default, 1
    low, high = round(math.floor(low_raw * 4) / 4, 2), round(math.ceil(high_raw * 4) / 4, 2)
    if high <= low:
        high = round(low + 0.25, 2)
    q1, q3 = np.nanpercentile(positive, [25, 75])
    default = (max(low, round(math.floor(q1 * 4) / 4, 2)), min(high, round(math.ceil(q3 * 4) / 4, 2)))
    if default[1] <= default[0]:
        default = (low, high)
    return low, high, default, 0.25


def phase_candidate_table(
    data: pd.DataFrame,
    artifact: dict,
    target_phases: list[str],
    excluded_phases: list[str],
    constraints: dict[str, tuple[float, float]],
    count: int = 5,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    model = artifact["model"]
    targets = phase_target_columns(data)
    medians = data[COMPOSITION_COLUMNS + PROCESS_COLUMNS].median(numeric_only=True)
    scales = data[COMPOSITION_COLUMNS + PROCESS_COLUMNS].std(numeric_only=True).replace(0, 1).fillna(1)
    composition_matrix = data[COMPOSITION_COLUMNS].to_numpy(dtype=float)
    for _ in range(3500):
        left = composition_matrix[int(rng.integers(0, len(composition_matrix)))]
        right = composition_matrix[int(rng.integers(0, len(composition_matrix)))]
        blend = float(rng.uniform(.2, .8))
        composition = blend * left + (1 - blend) * right
        composition += rng.normal(0, data[COMPOSITION_COLUMNS].std().fillna(0).to_numpy() * .015)
        composition = np.clip(composition, 0, None)
        total = composition.sum()
        if total <= 0:
            continue
        composition = composition * 100 / total
        candidate = {column: float(value) for column, value in zip(COMPOSITION_COLUMNS, composition)}
        for column, bounds in constraints.items():
            candidate[column] = float(rng.uniform(bounds[0], bounds[1]))
        if candidate["T_nucleation"] >= candidate["T_crystal"]:
            continue
        features = phase_feature_frame(pd.DataFrame([candidate]), include_process=True)
        predicted = model.predict(features)[0]
        probabilities = predict_probability_matrix(model, features, len(targets))[0]
        probability_by_phase = dict(zip(targets, probabilities))
        target_probability = float(np.mean([probability_by_phase[phase] for phase in target_phases]))
        excluded_probability = float(np.mean([probability_by_phase[phase] for phase in excluded_phases])) if excluded_phases else 0.0
        distance_values = []
        for column in COMPOSITION_COLUMNS + PROCESS_COLUMNS:
            value = candidate[column]
            distance_values.append(abs(value - float(medians[column])) / float(scales[column]))
        data_distance = float(np.mean(distance_values))
        score = np.clip(.78 * target_probability + .17 * (1 - excluded_probability) - .05 * min(data_distance / 2, 1), 0, 1)
        predicted_names = [phase_display_name(phase) for phase, value in probability_by_phase.items() if value >= .5]
        if not predicted_names:
            predicted_names = [phase_display_name(phase) for phase in sorted(probability_by_phase, key=probability_by_phase.get, reverse=True)[:3]]
        row = {
            **candidate,
            "目标晶相概率": round(target_probability, 4),
            "排除晶相概率": round(excluded_probability, 4),
            "综合评分": round(float(score) * 100, 2),
            "数据距离": round(data_distance, 3),
            "预测晶相": "、".join(predicted_names),
            "风险提示": "建议优先小试验证" if data_distance > 1.25 else "处于当前成分与工艺分布附近",
        }
        row.update({f"P_{phase}": round(float(probability), 6) for phase, probability in probability_by_phase.items()})
        rows.append(row)
        if len(rows) >= max(count * 8, 40):
            break
    if not rows:
        raise ValueError("当前热处理约束没有生成有效候选，请确认成核温度低于晶化温度并放宽范围。")
    result = pd.DataFrame(rows).sort_values(["综合评分", "数据距离"], ascending=[False, True]).head(count).copy()
    result.insert(0, "候选方案", [f"方案 {index}" for index in range(1, len(result) + 1)])
    return result


def sidebar() -> None:
    st.sidebar.title("LAS Phase Lab")
    st.sidebar.caption("锂铝硅微晶玻璃晶相研究工作台")
    st.sidebar.markdown("---")
    st.sidebar.success("当前使用真实数据")
    st.sidebar.caption("数据文件：data/data_merged.csv")
    st.sidebar.caption("数据来源和单位需在正式实验前确认")
    st.sidebar.markdown("---")
    st.sidebar.caption("预测结果用于科研筛选，不替代实验验证")


def page_overview(data: pd.DataFrame, artifacts: dict) -> None:
    full_metrics = artifacts["metrics"].query("模型 == '成分+热处理工艺'")
    micro_f1 = float(full_metrics["微平均 F1"].iloc[0]) if not full_metrics.empty else 0.0
    st.markdown(
        '<div class="lab-hero"><div class="eyebrow">LAS PHASE INTELLIGENCE</div><h2>基于真实数据的 LAS 微晶玻璃晶相智能预测</h2><p>将成分、热处理工艺与晶相标签连接起来，支持预测、分析和目标晶相方案筛选。</p></div>',
        unsafe_allow_html=True,
    )
    record_count = len(data)
    composition_count = data[COMPOSITION_COLUMNS].drop_duplicates().shape[0]
    process_count = int(process_available(data).sum())
    phase_count = len(phase_target_columns(data))
    a, b, c, d, e = st.columns(5)
    a.metric("真实记录", record_count)
    b.metric("独立成分", composition_count)
    c.metric("已提供工艺", process_count)
    d.metric("晶相标签", phase_count)
    e.metric("微平均 F1", f"{micro_f1:.3f}")

    st.subheader("研究工作流")
    steps = [
        ("01", "核查数据", "确认成分、工艺字段和晶相标签的质量"),
        ("02", "晶相预测", "输入新配方和热处理条件，输出晶相概率"),
        ("03", "目标设计", "以目标晶相为约束筛选候选成分与工艺"),
        ("04", "实验验证", "保存方案并回填后续制样和表征结果"),
    ]
    for column, (number, title, text) in zip(st.columns(4), steps):
        column.markdown(f'<div class="step-card"><strong>{number}</strong><h4>{title}</h4><p>{text}</p></div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("晶相出现频次")
        frequency = data[phase_target_columns(data)].sum().sort_values(ascending=False)
        frequency.index = [phase_display_name(value) for value in frequency.index]
        st.bar_chart(frequency, color="#0f918b")
    with right:
        st.subheader("数据边界")
        st.markdown(
            '<div class="notice-strip"><strong>当前系统定位</strong><br>真实成分 + 热处理工艺 → 晶相组合预测。当前数据不包含透光率、强度、热膨胀系数、晶粒尺寸或显微组织图片。</div>',
            unsafe_allow_html=True,
        )
        st.info("模型按独立成分组合分组留出测试集，避免同一配方的不同工艺记录同时进入训练集和测试集。")
        st.caption(f"当前分组测试：训练集 {artifacts['train_count']} 条，测试集 {artifacts['test_count']} 条。")


def page_phase_predict(data: pd.DataFrame, artifacts: dict) -> None:
    st.header("晶相预测")
    st.caption("输入一个新样品的真实成分和热处理工艺，预测 13 类晶相的出现概率")
    full_artifact = artifacts["models"]["成分+热处理工艺"]
    first, second, third = st.columns(3)
    sample_id = first.text_input("样品编号", value=f"PHASE-{datetime.now().strftime('%m%d-%H%M')}", key="phase_sample_id")
    batch_name = second.text_input("实验批次", value="真实数据验证批次", key="phase_batch")
    source = third.text_input("数据来源", value="待补充来源信息", key="phase_source")
    notes = st.text_area("样品备注", placeholder="例如：准备验证 β-锂辉石相的热处理窗口", height=68, key="phase_notes")
    values = render_input_fields(data, "predict")
    total = sum(values[column] for column in COMPOSITION_COLUMNS)
    if 98 <= total <= 102:
        st.success(f"成分总和 {total:.3f}%：处于检查范围内。")
    else:
        st.error(f"成分总和 {total:.3f}%：请先调整成分。")
    errors, warnings = validate_input(values, data)
    if warnings:
        with st.expander(f"输入质量提醒：{len(warnings)} 项"):
            for warning in warnings:
                st.warning(warning)
    if st.button("运行晶相预测", type="primary", use_container_width=True):
        if errors:
            for error in errors:
                st.error(error)
        else:
            features = phase_feature_frame(pd.DataFrame([values]), include_process=True)
            predicted = full_artifact["model"].predict(features)[0]
            probabilities = predict_probability_matrix(full_artifact["model"], features, len(artifacts["targets"]))[0]
            st.session_state["latest_phase_prediction"] = {
                "sample_id": sample_id,
                "values": values,
                "predicted": predicted,
                "probabilities": dict(zip(artifacts["targets"], probabilities)),
                "batch_name": batch_name,
                "source": source,
                "notes": notes,
            }
    latest = st.session_state.get("latest_phase_prediction")
    if latest and latest["sample_id"] == sample_id:
        probability_by_phase = latest["probabilities"]
        result = pd.DataFrame(
            {
                "晶相": [phase_display_name(phase) for phase in probability_by_phase],
                "原始字段": list(probability_by_phase),
                "预测状态": ["可能出现" if latest["predicted"][index] else "未判定为出现" for index in range(len(probability_by_phase))],
                "出现概率": list(probability_by_phase.values()),
            }
        ).sort_values("出现概率", ascending=False)
        possible_count = int(sum(latest["predicted"]))
        top_phase = result.iloc[0]
        a, b, c = st.columns(3)
        a.metric("预测为可能出现", possible_count)
        b.metric("最高概率晶相", top_phase["晶相"])
        c.metric("最高概率", f"{float(top_phase['出现概率']):.1%}")
        st.markdown("#### 预测结果")
        st.dataframe(result.round({"出现概率": 4}), use_container_width=True, hide_index=True)
        chart = result.set_index("晶相")["出现概率"].sort_values(ascending=True)
        st.bar_chart(chart, color="#0f918b")
        st.info("概率来自当前真实数据训练的多标签分类模型；它表示模型对标签的支持程度，不等同于实验相含量或统计置信区间。")
        save_col, download_col = st.columns(2)
        if save_col.button("保存样品与晶相预测", type="primary", use_container_width=True, key="save_phase_prediction"):
            save_phase_record(
                sample_id,
                latest["batch_name"],
                sample_id,
                latest["source"],
                latest["notes"],
                latest["values"],
                latest["probabilities"],
                dict(zip(artifacts["targets"], latest["predicted"])),
            )
            st.success(f"已保存样品 {sample_id} 的晶相预测记录。")
        download_col.download_button(
            "下载预测结果 CSV",
            result.to_csv(index=False).encode("utf-8-sig"),
            "las_phase_prediction.csv",
            "text/csv",
            use_container_width=True,
        )


def page_phase_inverse(data: pd.DataFrame, artifacts: dict) -> None:
    st.header("目标晶相设计")
    st.caption("输入希望出现的晶相和热处理约束，利用晶相模型筛选候选方案")
    target_options = phase_target_columns(data)
    most_common = data[target_options].sum().idxmax()
    target_phases = st.multiselect(
        "目标晶相",
        target_options,
        default=[most_common],
        format_func=phase_display_name,
        key="inverse_target_phases",
    )
    excluded_options = [phase for phase in target_options if phase not in target_phases]
    excluded_phases = st.multiselect(
        "希望降低风险的晶相（可选）",
        excluded_options,
        format_func=phase_display_name,
        key="inverse_excluded_phases",
    )
    st.caption("候选成分来自现有样品成分的组合扰动，工艺范围来自已提供的非零记录；结果是模型推荐，不是实验保证。")
    constraints: dict[str, tuple[float, float]] = {}
    left, right = st.columns(2)
    for index, column in enumerate(PROCESS_COLUMNS):
        low, high, default, step = process_slider_spec(data, column)
        host = left if index % 2 == 0 else right
        constraints[column] = host.slider(
            PROCESS_LABELS[column],
            min_value=low,
            max_value=high,
            value=default,
            step=step,
            key=f"inverse_{column}",
        )
    count = st.slider("候选方案数量", min_value=5, max_value=12, value=5, key="inverse_count")
    if st.button("生成目标晶相候选", type="primary", use_container_width=True):
        if not target_phases:
            st.error("请至少选择一个目标晶相。")
        elif constraints["T_nucleation"][1] >= constraints["T_crystal"][0] and constraints["T_nucleation"][0] >= constraints["T_crystal"][1]:
            st.error("当前工艺范围没有有效顺序，请调整为成核温度低于晶化温度。")
        else:
            try:
                st.session_state["phase_candidates"] = phase_candidate_table(
                    data,
                    artifacts["models"]["成分+热处理工艺"],
                    target_phases,
                    excluded_phases,
                    constraints,
                    count,
                )
            except ValueError as exc:
                st.session_state.pop("phase_candidates", None)
                st.error(str(exc))
    candidates = st.session_state.get("phase_candidates")
    if candidates is None:
        return
    st.success(f"已生成 {len(candidates)} 组目标晶相候选方案。")
    summary_columns = ["候选方案", "目标晶相概率", "排除晶相概率", "综合评分", "数据距离", "预测晶相", "风险提示"]
    st.dataframe(candidates[summary_columns].round(4), use_container_width=True, hide_index=True)
    st.markdown("#### 方案详情")
    selected_name = st.selectbox("查看方案", candidates["候选方案"].tolist(), key="selected_phase_candidate")
    selected = candidates[candidates["候选方案"] == selected_name].iloc[0]
    composition_view = pd.DataFrame(
        {"字段": COMPOSITION_COLUMNS + PROCESS_COLUMNS, "数值": [selected[column] for column in COMPOSITION_COLUMNS + PROCESS_COLUMNS]}
    )
    st.dataframe(composition_view.round(6), use_container_width=True, hide_index=True)
    st.markdown("#### 目标晶相概率")
    probability_view = pd.DataFrame(
        {
            "晶相": [phase_display_name(phase) for phase in target_options],
            "出现概率": [selected.get(f"P_{phase}", 0.0) for phase in target_options],
        }
    ).sort_values("出现概率", ascending=False)
    st.bar_chart(probability_view.set_index("晶相"), color="#d8943e")
    download_col, save_col = st.columns(2)
    download_col.download_button(
        "下载候选方案 CSV",
        candidates.to_csv(index=False).encode("utf-8-sig"),
        "las_phase_candidates.csv",
        "text/csv",
        use_container_width=True,
    )
    if save_col.button("将选中方案保存到样品记录", use_container_width=True, key="save_phase_candidate"):
        values = {column: float(selected[column]) for column in COMPOSITION_COLUMNS + PROCESS_COLUMNS}
        probability_payload = {phase: float(selected.get(f"P_{phase}", 0.0)) for phase in target_options}
        predicted_payload = {phase: int(probability >= .5) for phase, probability in probability_payload.items()}
        sample_id = f"DESIGN-{datetime.now().strftime('%m%d%H%M%S')}"
        save_phase_record(
            sample_id,
            "目标晶相设计批次",
            selected_name,
            "晶相模型推荐",
            "目标晶相候选方案，待制样和实验验证",
            values,
            probability_payload,
            predicted_payload,
        )
        st.success(f"已保存 {selected_name}，样品编号为 {sample_id}。")


def page_data_center(data: pd.DataFrame) -> None:
    st.header("数据中心")
    st.caption("查看真实数据、晶相分布、工艺完整性和可下载的原始表")
    sums = composition_sum(data)
    availability = process_available(data)
    targets = phase_target_columns(data)
    a, b, c, d, e = st.columns(5)
    a.metric("记录数", len(data))
    b.metric("独立成分", data[COMPOSITION_COLUMNS].drop_duplicates().shape[0])
    c.metric("成分和合格", f"{int(sums.between(98, 102).sum())}/{len(data)}")
    d.metric("工艺有值", int(availability.sum()))
    e.metric("晶相标签", len(targets))
    st.markdown("#### 数据筛选")
    filter_col, process_col = st.columns(2)
    phase_filter = filter_col.selectbox("按晶相筛选", ["全部"] + targets, format_func=lambda value: "全部" if value == "全部" else phase_display_name(value), key="data_phase_filter")
    process_filter = process_col.selectbox("按工艺完整性筛选", ["全部", "工艺字段有值", "工艺字段全为0"], key="data_process_filter")
    view = data.copy()
    if phase_filter != "全部":
        view = view[view[phase_filter] >= .5]
    if process_filter == "工艺字段有值":
        view = view[process_available(view)]
    elif process_filter == "工艺字段全为0":
        view = view[~process_available(view)]
    preview_columns = COMPOSITION_COLUMNS[:10] + PROCESS_COLUMNS + targets
    rename_map = display_phase_columns(data) | {column: PROCESS_LABELS[column] for column in PROCESS_COLUMNS}
    preview = view[preview_columns].head(200).rename(columns=rename_map)
    st.dataframe(preview, use_container_width=True, hide_index=True)
    st.caption(f"当前筛选得到 {len(view)} 条记录，表格最多展示前 200 条。原始字段和数值未被修改。")
    st.download_button("下载原始真实数据 CSV", data.to_csv(index=False).encode("utf-8-sig"), "data_merged_original_copy.csv", "text/csv", use_container_width=False)
    left, right = st.columns(2)
    with left:
        st.markdown("#### 晶相频次")
        frequency = data[targets].sum().sort_values(ascending=False)
        frequency.index = [phase_display_name(value) for value in frequency.index]
        st.bar_chart(frequency, color="#0f918b")
    with right:
        st.markdown("#### 数据质量")
        quality = pd.DataFrame(
            {
                "检查项": ["缺失值", "完全重复行", "成分和不在98%至102%", "工艺字段全为0"],
                "数量": [int(data.isna().sum().sum()), int(data.duplicated().sum()), int((~sums.between(98, 102)).sum()), int((~availability).sum())],
            }
        )
        st.dataframe(quality, use_container_width=True, hide_index=True)
        st.warning("工艺字段全为 0 的记录需要向数据提供者确认：0 是真实工艺值，还是缺失值编码。")


def page_records() -> None:
    st.header("样品记录")
    st.caption("查看已保存的新样品、晶相预测和目标设计方案")
    samples = read_table("SELECT * FROM samples ORDER BY created_at DESC")
    if samples.empty:
        st.info("暂时没有保存的样品。可以先在“晶相预测”或“目标晶相设计”页面保存一条记录。")
        return
    st.dataframe(samples, use_container_width=True, hide_index=True)
    st.download_button("下载样品记录 CSV", samples.to_csv(index=False).encode("utf-8-sig"), "las_phase_samples.csv", "text/csv")
    sample_id = st.selectbox("查看晶相预测详情", samples["sample_id"].tolist(), key="record_sample_id")
    predictions = read_table("SELECT * FROM phase_predictions WHERE sample_id=? ORDER BY id DESC", (sample_id,))
    if predictions.empty:
        st.info("该样品尚未保存晶相预测详情。")
        return
    latest = predictions.iloc[0]
    payload = json.loads(latest["predicted_json"])
    detail = pd.DataFrame(
        {
            "晶相": [phase_display_name(phase) for phase in payload],
            "原始字段": list(payload),
            "出现概率": [payload[phase]["probability"] for phase in payload],
            "预测状态": ["可能出现" if payload[phase]["predicted"] else "未判定为出现" for phase in payload],
        }
    ).sort_values("出现概率", ascending=False)
    st.markdown(f"#### {sample_id} 的晶相结果")
    st.dataframe(detail.round(4), use_container_width=True, hide_index=True)


def page_model_eval(data: pd.DataFrame, artifacts: dict) -> None:
    st.header("模型评估")
    st.caption("比较只使用成分的基线模型与加入热处理工艺的模型")
    st.dataframe(artifacts["metrics"].round(4), use_container_width=True, hide_index=True)
    st.markdown("#### 分晶相指标")
    st.dataframe(artifacts["per_phase"].round(4), use_container_width=True, hide_index=True)
    importance = feature_importance_table(artifacts["models"]["成分+热处理工艺"]).head(15).sort_values("重要性", ascending=True)
    st.markdown("#### 成分+热处理工艺模型的整体特征重要性")
    st.bar_chart(importance.set_index("特征")["重要性"], color="#0f918b")
    st.info("测试集按独立成分组合分组留出。完全匹配率要求 13 个晶相标签全部预测正确，通常会低于微平均 F1；少数晶相样本不足时，分晶相 F1 可能不稳定。")
    st.markdown("#### 评估边界")
    st.markdown("当前模型用于展示真实数据上的建模流程和候选筛选能力。它不代表所有 LAS 微晶玻璃配方的通用规律，也不能替代 XRD 鉴定和热处理实验。")


def page_docs() -> None:
    st.header("项目资料")
    st.markdown("#### 项目名称")
    st.info("基于真实数据驱动的 LAS 微晶玻璃晶相智能预测与目标晶相设计系统")
    st.markdown("#### 系统功能")
    st.markdown(
        "1. 读取并审计真实 LAS 成分、热处理和晶相数据。\n"
        "2. 使用成分和热处理工艺预测多个晶相标签。\n"
        "3. 以目标晶相和工艺约束筛选模型推荐候选方案。\n"
        "4. 展示模型指标、分晶相 F1 和整体特征重要性。\n"
        "5. 保存样品编号、输入条件和预测结果，形成后续实验记录入口。"
    )
    st.markdown("#### 当前数据边界")
    st.warning("当前数据没有透光率、强度、热膨胀系数、晶粒尺寸和显微组织图片。因此本系统暂不宣称性能预测或图像多模态预测。")
    st.markdown("#### 三分钟演示顺序")
    st.markdown(
        "1. 总览页展示 751 条真实记录、550 种独立成分和晶相分布。\n"
        "2. 在晶相预测页输入配方和热处理工艺，运行模型并保存结果。\n"
        "3. 在目标晶相设计页选择目标晶相，生成候选方案并查看评分。\n"
        "4. 在模型评估页展示成分基线与成分+工艺模型对比。\n"
        "5. 说明预测结果需要通过 XRD、制样和热处理实验验证。"
    )
    st.markdown("#### 建议继续补充的数据")
    st.markdown("样品编号、工艺字段中 0 值的含义、XRD 原始数据或晶相含量、测试方法、数据来源，以及后续真实实验回填结果。")


def main() -> None:
    inject_styles()
    init_db()
    sidebar()
    if not REAL_DATA_PATH.exists():
        st.error(f"没有找到真实数据文件：{REAL_DATA_PATH}")
        st.stop()
    data = load_real_data()
    if not is_phase_dataset(data):
        st.error("当前 data_merged.csv 不符合晶相数据结构，请检查成分、工艺和晶相字段。")
        st.stop()
    artifacts = train_phase_models(data)
    if artifacts is None:
        st.error("晶相模型依赖未安装或数据结构无法训练，请检查 requirements.txt。")
        st.stop()
    st.title("基于真实数据驱动的 LAS 微晶玻璃晶相智能预测与目标晶相设计系统")
    st.caption("基于真实成分、热处理工艺和晶相数据 | 预测结果用于科研筛选与实验设计")
    tabs = st.tabs(["总览", "晶相预测", "目标晶相设计", "数据中心", "样品记录", "模型评估", "项目资料"])
    with tabs[0]:
        page_overview(data, artifacts)
    with tabs[1]:
        page_phase_predict(data, artifacts)
    with tabs[2]:
        page_phase_inverse(data, artifacts)
    with tabs[3]:
        page_data_center(data)
    with tabs[4]:
        page_records()
    with tabs[5]:
        page_model_eval(data, artifacts)
    with tabs[6]:
        page_docs()


if __name__ == "__main__":
    main()
