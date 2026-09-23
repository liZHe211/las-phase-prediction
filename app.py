from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

LOCAL_VENDOR = Path(__file__).parent / ".vendor_nsga"
if LOCAL_VENDOR.exists():
    sys.path.insert(0, str(LOCAL_VENDOR))

import numpy as np
import pandas as pd
import streamlit as st

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
except ImportError:  # pragma: no cover
    NSGA2 = None
    Problem = None
    minimize = None

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
XRD_UPLOAD_DIR = DATA_DIR / "uploads" / "xrd"

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
            CREATE TABLE IF NOT EXISTS experiments(
                experiment_id TEXT PRIMARY KEY,
                sample_id TEXT NOT NULL,
                experiment_date TEXT,
                status TEXT,
                actual_T_nucleation REAL,
                actual_t1 REAL,
                actual_T_crystal REAL,
                actual_t2 REAL,
                observed_json TEXT NOT NULL,
                comparison_json TEXT NOT NULL,
                xrd_file_path TEXT,
                notes TEXT,
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_sample_id ON experiments(sample_id);
            CREATE TABLE IF NOT EXISTS audit_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                action TEXT,
                created_at TEXT,
                detail TEXT
            );
            """
        )
        ensure_experiments_schema(conn)


def ensure_experiments_schema(conn: sqlite3.Connection) -> None:
    """Upgrade the legacy performance table without deleting existing rows."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(experiments)").fetchall()}
    additions = {
        "experiment_id": "TEXT",
        "status": "TEXT",
        "actual_T_nucleation": "REAL",
        "actual_t1": "REAL",
        "actual_T_crystal": "REAL",
        "actual_t2": "REAL",
        "observed_json": "TEXT DEFAULT '{}'",
        "comparison_json": "TEXT DEFAULT '{}'",
        "xrd_file_path": "TEXT DEFAULT ''",
        "updated_at": "TEXT",
    }
    for column, definition in additions.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE experiments ADD COLUMN {column} {definition}")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_experiments_experiment_id ON experiments(experiment_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_experiments_sample_id ON experiments(sample_id)"
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


def build_experiment_plan(candidates: pd.DataFrame, plan_count: int = 5) -> pd.DataFrame:
    """Pick complementary candidates for a small validation batch."""
    if candidates.empty:
        return pd.DataFrame()
    available = candidates.copy()
    available["_score"] = pd.to_numeric(available["综合评分"], errors="coerce").fillna(0)
    available["_distance"] = pd.to_numeric(available["数据距离"], errors="coerce").fillna(np.inf)
    available["_load"] = pd.to_numeric(available["工艺负荷"], errors="coerce").fillna(np.inf)
    picks: list[int] = []

    def add_index(frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        index = int(frame.index[0])
        if index not in picks:
            picks.append(index)

    add_index(available.sort_values(["_score", "_distance"], ascending=[False, True]))
    add_index(available.sort_values(["_distance", "_score"], ascending=[True, False]))
    add_index(available.sort_values(["_load", "_score"], ascending=[True, False]))
    if len(picks) >= 2:
        baseline = available.loc[picks[1]]
        remaining = available.drop(index=picks)
        if not remaining.empty:
            process_delta = remaining[PROCESS_COLUMNS].sub(baseline[PROCESS_COLUMNS].astype(float)).abs().sum(axis=1)
            add_index(remaining.assign(_process_delta=process_delta).sort_values("_process_delta", ascending=False))
    for index in available.sort_values(["_score", "_distance"], ascending=[False, True]).index:
        add_index(available.loc[[index]])
        if len(picks) >= min(plan_count, len(available)):
            break
    plan = available.loc[picks[:plan_count]].copy()
    roles = ["高目标概率优化组", "接近历史数据稳健组", "低工艺负荷组", "工艺差异对照组"]
    plan.insert(0, "实验组别", [roles[i] if i < len(roles) else "备用验证组" for i in range(len(plan))])
    plan.insert(0, "样品编号", [f"LAS-EXP-{datetime.now().strftime('%m%d%H%M%S%f')[:15]}-{i:02d}" for i in range(1, len(plan) + 1)])
    plan["实验状态"] = "待制样"
    plan["建议验证"] = "熔制/热处理后进行XRD晶相鉴定"
    plan["计划备注"] = plan.apply(
        lambda row: f"NSGA-II推荐；{row['风险提示']}；目标晶相概率{float(row['目标晶相概率']):.1%}", axis=1
    )
    return plan.drop(columns=["_score", "_distance", "_load"], errors="ignore")


def save_experiment_plan(plan: pd.DataFrame, target_phases: list[str]) -> int:
    """Create pending samples so the plan can be continued in 实验验证."""
    saved = 0
    for _, row in plan.iterrows():
        values = {column: float(row[column]) for column in COMPOSITION_COLUMNS + PROCESS_COLUMNS}
        probability_payload = {phase: float(row.get(f"P_{phase}", 0.0)) for phase in target_phases}
        predicted_payload = {phase: int(value >= .5) for phase, value in probability_payload.items()}
        sample_id = str(row["样品编号"])
        save_phase_record(
            sample_id,
            "NSGA-II实验验证批次",
            str(row["实验组别"]),
            "NSGA-II实验计划",
            str(row["计划备注"]),
            values,
            probability_payload,
            predicted_payload,
        )
        with db_connect() as conn:
            conn.execute("UPDATE samples SET status=? WHERE sample_id=?", ("待制样", sample_id))
        log_action("生成实验计划", sample_id, "NSGA-II实验计划")
        saved += 1
    return saved


def parse_json(value: str | None, fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def load_sample_input(sample_id: str) -> dict[str, float]:
    rows = read_table("SELECT input_json FROM sample_inputs WHERE sample_id=?", (sample_id,))
    if rows.empty:
        return {}
    return parse_json(rows.iloc[0]["input_json"], {})


def load_latest_prediction(sample_id: str) -> dict:
    rows = read_table(
        "SELECT predicted_json FROM phase_predictions WHERE sample_id=? ORDER BY id DESC LIMIT 1",
        (sample_id,),
    )
    if rows.empty:
        return {}
    return parse_json(rows.iloc[0]["predicted_json"], {})


def compare_phase_result(prediction: dict, observed_phases: list[str], confirmed_empty: bool = False) -> dict:
    predicted_set = {
        phase for phase, result in prediction.items()
        if isinstance(result, dict) and int(result.get("predicted", 0)) == 1
    }
    observed_set = set(observed_phases)
    if not observed_set and not confirmed_empty:
        return {
            "result": "待核对",
            "predicted": sorted(predicted_set),
            "observed": [],
            "matched": [],
            "missed": [],
            "extra": [],
            "precision": None,
            "recall": None,
            "f1": None,
            "jaccard": None,
            "exact_match": None,
        }
    matched = predicted_set & observed_set
    missed = observed_set - predicted_set
    extra = predicted_set - observed_set
    precision = len(matched) / len(predicted_set) if predicted_set else (1.0 if not observed_set else 0.0)
    recall = len(matched) / len(observed_set) if observed_set else (1.0 if not predicted_set else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = predicted_set | observed_set
    jaccard = len(matched) / len(union) if union else 1.0
    exact_match = predicted_set == observed_set
    result = "完全一致" if exact_match else ("部分一致" if matched else "不一致")
    return {
        "result": result,
        "predicted": sorted(predicted_set),
        "observed": sorted(observed_set),
        "matched": sorted(matched),
        "missed": sorted(missed),
        "extra": sorted(extra),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "exact_match": exact_match,
    }


def save_xrd_file(uploaded_file, experiment_id: str) -> str:
    if uploaded_file is None:
        return ""
    if uploaded_file.size > 20 * 1024 * 1024:
        raise ValueError("XRD附件超过20 MB，请压缩后重新上传。")
    suffix = Path(uploaded_file.name).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".pdf", ".csv", ".txt", ".xy", ".raw"}
    if suffix not in allowed:
        raise ValueError("不支持该附件格式。请上传 PNG、JPG、PDF、CSV、TXT、XY 或 RAW 文件。")
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", experiment_id).strip("_") or "experiment"
    XRD_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"{safe_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}{suffix}"
    destination = XRD_UPLOAD_DIR / file_name
    destination.write_bytes(uploaded_file.getvalue())
    return str(destination.relative_to(ROOT)).replace("\\", "/")


def save_experiment_record(
    experiment_id: str,
    sample_id: str,
    experiment_date: str,
    status: str,
    actual_process: dict[str, float],
    observed_phases: list[str],
    confirmed_empty: bool,
    comparison: dict,
    xrd_file_path: str,
    notes: str,
) -> None:
    timestamp = now_text()
    observed_payload = {"phases": observed_phases, "confirmed_empty": confirmed_empty}
    with db_connect() as conn:
        existing = conn.execute(
            "SELECT created_at,xrd_file_path FROM experiments WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        created_at = existing["created_at"] if existing else timestamp
        stored_path = xrd_file_path or (existing["xrd_file_path"] if existing else "")
        conn.execute(
            """
            INSERT INTO experiments(
                experiment_id,sample_id,experiment_date,status,
                actual_T_nucleation,actual_t1,actual_T_crystal,actual_t2,
                observed_json,comparison_json,xrd_file_path,notes,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(experiment_id) DO UPDATE SET
                sample_id=excluded.sample_id,
                experiment_date=excluded.experiment_date,
                status=excluded.status,
                actual_T_nucleation=excluded.actual_T_nucleation,
                actual_t1=excluded.actual_t1,
                actual_T_crystal=excluded.actual_T_crystal,
                actual_t2=excluded.actual_t2,
                observed_json=excluded.observed_json,
                comparison_json=excluded.comparison_json,
                xrd_file_path=excluded.xrd_file_path,
                notes=excluded.notes,
                updated_at=excluded.updated_at
            """,
            (
                experiment_id, sample_id, experiment_date, status,
                actual_process["T_nucleation"], actual_process["t1"],
                actual_process["T_crystal"], actual_process["t2"],
                json.dumps(observed_payload, ensure_ascii=False),
                json.dumps(comparison, ensure_ascii=False),
                stored_path, notes, created_at, timestamp,
            ),
        )
        conn.execute("UPDATE samples SET status=? WHERE sample_id=?", (status, sample_id))
    log_action("保存实验验证", f"{experiment_id} / {sample_id}", "实验回填")


def experiment_export_table(experiments: pd.DataFrame) -> pd.DataFrame:
    if experiments.empty:
        return experiments
    rows = []
    for _, row in experiments.iterrows():
        observed = parse_json(row["observed_json"], {})
        comparison = parse_json(row["comparison_json"], {})
        phases = observed.get("phases", []) if isinstance(observed, dict) else []
        rows.append(
            {
                "实验编号": row["experiment_id"],
                "样品编号": row["sample_id"],
                "实验日期": row["experiment_date"],
                "状态": row["status"],
                "实际成核温度": row["actual_T_nucleation"],
                "实际成核时间": row["actual_t1"],
                "实际晶化温度": row["actual_T_crystal"],
                "实际晶化时间": row["actual_t2"],
                "实测晶相": "、".join(phase_display_name(phase) for phase in phases) or "未录入",
                "对比结论": comparison.get("result", "待核对"),
                "样品级F1": comparison.get("f1"),
                "Jaccard一致度": comparison.get("jaccard"),
                "XRD附件": row["xrd_file_path"],
                "实验备注": row["notes"],
                "更新时间": row["updated_at"],
            }
        )
    return pd.DataFrame(rows)


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


def project_bounded_composition(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: float = 100.0,
) -> np.ndarray:
    """Project rows onto a bounded composition simplex with an exact total."""
    matrix = np.atleast_2d(np.asarray(values, dtype=float))
    if lower.sum() > total + 1e-8 or upper.sum() < total - 1e-8:
        raise ValueError("成分上下限无法组成100%，请降低下限或提高上限。")
    projected = np.empty_like(matrix)
    for index, row in enumerate(matrix):
        lambda_low = float(np.min(row - upper))
        lambda_high = float(np.max(row - lower))
        for _ in range(70):
            midpoint = (lambda_low + lambda_high) / 2
            candidate = np.clip(row - midpoint, lower, upper)
            if candidate.sum() > total:
                lambda_low = midpoint
            else:
                lambda_high = midpoint
        projected[index] = np.clip(row - (lambda_low + lambda_high) / 2, lower, upper)
    return projected


def nearest_standardized_distance(
    candidates: np.ndarray,
    history: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    candidate_z = candidates / scales
    history_z = history / scales
    candidate_norm = np.sum(candidate_z ** 2, axis=1, keepdims=True)
    history_norm = np.sum(history_z ** 2, axis=1)
    squared = candidate_norm + history_norm - 2 * candidate_z @ history_z.T
    return np.sqrt(np.maximum(squared.min(axis=1), 0) / candidates.shape[1])


def history_distance_thresholds(history: np.ndarray, scales: np.ndarray) -> tuple[float, float]:
    if len(history) < 2:
        return 0.5, 1.0
    normalized = history / scales
    norms = np.sum(normalized ** 2, axis=1)
    squared = norms[:, None] + norms[None, :] - 2 * normalized @ normalized.T
    np.fill_diagonal(squared, np.inf)
    nearest = np.sqrt(np.maximum(squared.min(axis=1), 0) / history.shape[1])
    positive = nearest[nearest > 1e-9]
    if positive.size == 0:
        return 0.25, 0.5
    q75, q95 = np.quantile(positive, [0.75, 0.95])
    return max(float(q75), 0.05), max(float(q95), float(q75) + 0.05)


def nsga2_candidate_table(
    data: pd.DataFrame,
    artifact: dict,
    target_phases: list[str],
    excluded_phases: list[str],
    process_constraints: dict[str, tuple[float, float]],
    composition_constraints: dict[str, tuple[float, float]],
    count: int = 5,
    population_size: int = 48,
    generations: int = 35,
    minimize_process_load: bool = True,
) -> tuple[pd.DataFrame, dict]:
    if NSGA2 is None or Problem is None or minimize is None:
        raise RuntimeError("缺少pymoo依赖，无法运行NSGA-II。请重新安装requirements.txt。")
    targets = phase_target_columns(data)
    model = artifact["model"]
    composition_lower = np.array([composition_constraints[column][0] for column in COMPOSITION_COLUMNS], dtype=float)
    composition_upper = np.array([composition_constraints[column][1] for column in COMPOSITION_COLUMNS], dtype=float)
    if np.any(composition_lower > composition_upper):
        raise ValueError("部分成分的优化下限高于上限，请检查成分约束表。")
    if composition_lower.sum() > 100 + 1e-8 or composition_upper.sum() < 100 - 1e-8:
        raise ValueError("当前成分约束无法组成100%的配方，请调整上下限。")
    process_lower = np.array([process_constraints[column][0] for column in PROCESS_COLUMNS], dtype=float)
    process_upper = np.array([process_constraints[column][1] for column in PROCESS_COLUMNS], dtype=float)
    variable_lower = np.concatenate([composition_lower, process_lower])
    variable_upper = np.concatenate([composition_upper, process_upper])
    if process_lower[0] >= process_upper[2]:
        raise ValueError("工艺约束中不存在成核温度低于晶化温度的可行区间。")

    history_frame = data[COMPOSITION_COLUMNS + PROCESS_COLUMNS].apply(pd.to_numeric, errors="coerce").copy()
    valid_history = process_available(data) & history_frame["T_nucleation"].lt(history_frame["T_crystal"])
    history_frame = history_frame.loc[valid_history].copy()
    if history_frame.empty:
        raise ValueError("没有可用于约束优化的完整热处理历史数据。")
    for column in COMPOSITION_COLUMNS:
        history_frame[column] = history_frame[column].fillna(pd.to_numeric(data[column], errors="coerce").median())
    for column in PROCESS_COLUMNS:
        positive = pd.to_numeric(data[column], errors="coerce")
        positive = positive[positive > 0]
        fallback = float(positive.median()) if not positive.empty else float(process_constraints[column][0])
        history_frame[column] = history_frame[column].replace(0, np.nan).fillna(fallback)
    history_matrix = history_frame[COMPOSITION_COLUMNS + PROCESS_COLUMNS].to_numpy(dtype=float)
    scales = np.nanstd(history_matrix, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-9), scales, 1.0)
    distance_q75, distance_q95 = history_distance_thresholds(history_matrix, scales)

    def transform_variables(x: np.ndarray, round_process: bool = False) -> np.ndarray:
        matrix = np.atleast_2d(np.asarray(x, dtype=float)).copy()
        compositions = project_bounded_composition(
            matrix[:, :len(COMPOSITION_COLUMNS)], composition_lower, composition_upper
        )
        processes = np.clip(matrix[:, len(COMPOSITION_COLUMNS):], process_lower, process_upper)
        if round_process:
            processes[:, 0] = np.round(processes[:, 0])
            processes[:, 2] = np.round(processes[:, 2])
            processes[:, 1] = np.round(processes[:, 1] * 4) / 4
            processes[:, 3] = np.round(processes[:, 3] * 4) / 4
        return np.column_stack([compositions, processes])

    def evaluate_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame = pd.DataFrame(matrix, columns=COMPOSITION_COLUMNS + PROCESS_COLUMNS)
        features = phase_feature_frame(frame, include_process=True)
        probabilities = predict_probability_matrix(model, features, len(targets))
        target_indexes = [targets.index(phase) for phase in target_phases]
        target_probability = probabilities[:, target_indexes].mean(axis=1)
        if excluded_phases:
            excluded_indexes = [targets.index(phase) for phase in excluded_phases]
            excluded_probability = probabilities[:, excluded_indexes].max(axis=1)
        else:
            excluded_probability = np.zeros(len(matrix))
        data_distance = nearest_standardized_distance(matrix, history_matrix, scales)
        process_span = np.maximum(process_upper - process_lower, 1e-9)
        process_load = ((matrix[:, len(COMPOSITION_COLUMNS):] - process_lower) / process_span).mean(axis=1)
        return probabilities, target_probability, excluded_probability, np.column_stack([data_distance, process_load])

    objective_count = 4 if minimize_process_load else 3

    class LasPhaseOptimizationProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=len(COMPOSITION_COLUMNS) + len(PROCESS_COLUMNS),
                n_obj=objective_count,
                n_ieq_constr=1,
                xl=variable_lower,
                xu=variable_upper,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            transformed = transform_variables(x)
            _, target_probability, excluded_probability, other = evaluate_matrix(transformed)
            data_distance = other[:, 0]
            process_load = other[:, 1]
            objectives = [1 - target_probability, excluded_probability, data_distance]
            if minimize_process_load:
                objectives.append(process_load)
            out["F"] = np.column_stack(objectives)
            process_values = transformed[:, len(COMPOSITION_COLUMNS):]
            out["G"] = (process_values[:, 0] - process_values[:, 2] + 1.0)[:, None]

    rng = np.random.default_rng(42)
    historical_seed = history_matrix.copy()
    historical_seed[:, :len(COMPOSITION_COLUMNS)] = project_bounded_composition(
        historical_seed[:, :len(COMPOSITION_COLUMNS)], composition_lower, composition_upper
    )
    historical_seed[:, len(COMPOSITION_COLUMNS):] = np.clip(
        historical_seed[:, len(COMPOSITION_COLUMNS):], process_lower, process_upper
    )
    valid_seed = historical_seed[:, len(COMPOSITION_COLUMNS)] < historical_seed[:, len(COMPOSITION_COLUMNS) + 2]
    historical_seed = historical_seed[valid_seed]
    seed_count = min(len(historical_seed), population_size // 2)
    if seed_count:
        history_indexes = rng.choice(len(historical_seed), size=seed_count, replace=len(historical_seed) < seed_count)
        seed_rows = historical_seed[history_indexes]
    else:
        seed_rows = np.empty((0, len(variable_lower)))
    random_rows = rng.uniform(variable_lower, variable_upper, size=(population_size - seed_count, len(variable_lower)))
    initial_population = np.vstack([seed_rows, random_rows])

    started = time.perf_counter()
    result = minimize(
        LasPhaseOptimizationProblem(),
        NSGA2(pop_size=population_size, sampling=initial_population, eliminate_duplicates=True),
        termination=("n_gen", generations),
        seed=42,
        verbose=False,
    )
    duration = time.perf_counter() - started
    if result.pop is None:
        raise ValueError("NSGA-II没有返回候选，请放宽成分或工艺约束。")
    population = np.asarray(result.pop.get("X"), dtype=float)
    constraint_violation = np.asarray(result.pop.get("CV"), dtype=float).reshape(-1)
    ranks_raw = result.pop.get("rank")
    ranks = np.zeros(len(population), dtype=int) if ranks_raw is None else np.asarray(ranks_raw, dtype=int).reshape(-1)
    feasible = constraint_violation <= 1e-8
    population = population[feasible]
    ranks = ranks[feasible]
    if population.size == 0:
        raise ValueError("当前约束下没有满足成核温度低于晶化温度的候选。")
    transformed = transform_variables(population, round_process=True)
    valid_order = transformed[:, len(COMPOSITION_COLUMNS)] < transformed[:, len(COMPOSITION_COLUMNS) + 2]
    transformed = transformed[valid_order]
    ranks = ranks[valid_order]
    probabilities, target_probability, excluded_probability, other = evaluate_matrix(transformed)
    data_distance, process_load = other[:, 0], other[:, 1]

    rows = []
    for index, matrix_row in enumerate(transformed):
        probability_by_phase = dict(zip(targets, probabilities[index]))
        predicted_names = [phase_display_name(phase) for phase, value in probability_by_phase.items() if value >= .5]
        if not predicted_names:
            predicted_names = [
                phase_display_name(phase)
                for phase in sorted(probability_by_phase, key=probability_by_phase.get, reverse=True)[:3]
            ]
        if data_distance[index] <= distance_q75:
            risk_level = "低风险"
            risk_note = "接近已有成分与工艺分布"
        elif data_distance[index] <= distance_q95:
            risk_level = "中风险"
            risk_note = "位于历史数据稀疏区域，建议小试"
        else:
            risk_level = "高风险"
            risk_note = "偏离历史数据较远，必须优先验证"
        proximity = math.exp(-float(data_distance[index]) / max(distance_q95, 0.05))
        if minimize_process_load:
            score = (
                .62 * target_probability[index]
                + .15 * (1 - excluded_probability[index])
                + .15 * proximity
                + .08 * (1 - process_load[index])
            )
        else:
            score = .70 * target_probability[index] + .15 * (1 - excluded_probability[index]) + .15 * proximity
        candidate = {
            **dict(zip(COMPOSITION_COLUMNS + PROCESS_COLUMNS, matrix_row)),
            "Pareto层级": int(ranks[index]) + 1,
            "目标晶相概率": round(float(target_probability[index]), 6),
            "排除晶相概率": round(float(excluded_probability[index]), 6),
            "数据距离": round(float(data_distance[index]), 6),
            "工艺负荷": round(float(process_load[index]), 6),
            "综合评分": round(float(np.clip(score, 0, 1)) * 100, 2),
            "风险等级": risk_level,
            "预测晶相": "、".join(predicted_names),
            "推荐理由": f"目标晶相概率{target_probability[index]:.1%}，{risk_note}",
            "风险提示": risk_note,
            "优化算法": "NSGA-II",
        }
        candidate.update({f"P_{phase}": round(float(probability), 6) for phase, probability in probability_by_phase.items()})
        rows.append(candidate)
    if not rows:
        raise ValueError("NSGA-II没有生成可用候选，请放宽约束后重试。")
    candidate_frame = pd.DataFrame(rows)
    rounded_key = candidate_frame[COMPOSITION_COLUMNS].round(3).astype(str).agg("|".join, axis=1)
    rounded_key += "|" + candidate_frame[PROCESS_COLUMNS].round(2).astype(str).agg("|".join, axis=1)
    candidate_frame = candidate_frame.loc[~rounded_key.duplicated()].copy()
    candidate_frame = candidate_frame.sort_values(
        ["Pareto层级", "综合评分", "数据距离"], ascending=[True, False, True]
    ).head(count)
    candidate_frame.insert(0, "候选方案", [f"方案 {index}" for index in range(1, len(candidate_frame) + 1)])
    info = {
        "algorithm": "NSGA-II",
        "population_size": population_size,
        "generations": generations,
        "evaluations": population_size * generations,
        "duration_seconds": duration,
        "distance_q75": distance_q75,
        "distance_q95": distance_q95,
        "process_objective": minimize_process_load,
    }
    return candidate_frame, info


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
    st.caption("使用NSGA-II同时优化目标晶相、非目标晶相风险、历史数据距离和热处理负荷")
    target_options = phase_target_columns(data)
    most_common = data[target_options].sum().idxmax()
    target_col, excluded_col = st.columns(2)
    target_phases = target_col.multiselect(
        "目标晶相",
        target_options,
        default=[most_common],
        format_func=phase_display_name,
        key="inverse_target_phases",
    )
    excluded_options = [phase for phase in target_options if phase not in target_phases]
    excluded_phases = excluded_col.multiselect(
        "希望抑制的晶相（可选）",
        excluded_options,
        format_func=phase_display_name,
        key="inverse_excluded_phases",
    )
    st.info("数值预测来自当前ExtraTrees晶相模型；NSGA-II只负责搜索候选配方和工艺，不代表实验一定成功。")

    st.markdown("#### 热处理工艺约束")
    process_constraints: dict[str, tuple[float, float]] = {}
    left, right = st.columns(2)
    for index, column in enumerate(PROCESS_COLUMNS):
        low, high, default, step = process_slider_spec(data, column)
        host = left if index % 2 == 0 else right
        process_constraints[column] = host.slider(
            PROCESS_LABELS[column],
            min_value=low,
            max_value=high,
            value=default,
            step=step,
            key=f"inverse_{column}",
        )

    composition_numeric = data[COMPOSITION_COLUMNS].apply(pd.to_numeric, errors="coerce")
    composition_editor = pd.DataFrame(
        {
            "成分": COMPOSITION_COLUMNS,
            "历史最小": [float(composition_numeric[column].min()) for column in COMPOSITION_COLUMNS],
            "历史最大": [float(composition_numeric[column].max()) for column in COMPOSITION_COLUMNS],
            "优化下限": [float(composition_numeric[column].quantile(.01)) for column in COMPOSITION_COLUMNS],
            "优化上限": [float(composition_numeric[column].quantile(.99)) for column in COMPOSITION_COLUMNS],
        }
    ).round(5)
    with st.expander("成分搜索范围", expanded=False):
        st.caption("默认使用历史数据1%至99%分位范围。可修改优化上下限，但最终配方仍会被约束为总和100%。")
        edited_composition = st.data_editor(
            composition_editor,
            hide_index=True,
            disabled=["成分", "历史最小", "历史最大"],
            num_rows="fixed",
            use_container_width=True,
            height=430,
            key="nsga_composition_bounds",
            column_config={
                "历史最小": st.column_config.NumberColumn(format="%.5f"),
                "历史最大": st.column_config.NumberColumn(format="%.5f"),
                "优化下限": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, format="%.5f"),
                "优化上限": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, format="%.5f"),
            },
        )
        lower_sum = float(pd.to_numeric(edited_composition["优化下限"], errors="coerce").sum())
        upper_sum = float(pd.to_numeric(edited_composition["优化上限"], errors="coerce").sum())
        st.caption(f"当前成分下限合计 {lower_sum:.2f}%，上限合计 {upper_sum:.2f}%；必须满足下限合计≤100%≤上限合计。")

    option_a, option_b, option_c = st.columns(3)
    count = option_a.slider("候选方案数量", min_value=5, max_value=12, value=5, key="inverse_count")
    optimization_mode = option_b.selectbox(
        "优化精度",
        ["快速演示", "标准优化"],
        help="快速演示约1400次评价；标准优化约4320次评价。",
        key="nsga_mode",
    )
    minimize_process_load = option_c.checkbox(
        "兼顾降低热处理负荷",
        value=True,
        help="开启后，NSGA-II会把较低温度和较短时间作为第四个优化目标。",
        key="nsga_process_load",
    )
    population_size, generations = ((40, 35) if optimization_mode == "快速演示" else (72, 60))
    composition_constraints = {
        str(row["成分"]): (float(row["优化下限"]), float(row["优化上限"]))
        for _, row in edited_composition.iterrows()
    }

    if st.button("运行 NSGA-II 多目标优化", type="primary", use_container_width=True):
        if not target_phases:
            st.error("请至少选择一个目标晶相。")
        elif process_constraints["T_nucleation"][0] >= process_constraints["T_crystal"][1]:
            st.error("当前工艺范围没有有效顺序，请调整为成核温度低于晶化温度。")
        else:
            try:
                with st.spinner("NSGA-II正在搜索Pareto候选方案，请稍候..."):
                    candidates, optimization_info = nsga2_candidate_table(
                        data,
                        artifacts["models"]["成分+热处理工艺"],
                        target_phases,
                        excluded_phases,
                        process_constraints,
                        composition_constraints,
                        count=count,
                        population_size=population_size,
                        generations=generations,
                        minimize_process_load=minimize_process_load,
                    )
                st.session_state["phase_candidates"] = candidates
                st.session_state["phase_optimization_info"] = optimization_info
            except (RuntimeError, ValueError) as exc:
                st.session_state.pop("phase_candidates", None)
                st.session_state.pop("phase_optimization_info", None)
                st.error(str(exc))
    candidates = st.session_state.get("phase_candidates")
    if candidates is None:
        return
    required_columns = {"Pareto层级", "工艺负荷", "风险等级", "优化算法"}
    if not required_columns.issubset(candidates.columns):
        st.session_state.pop("phase_candidates", None)
        st.info("优化算法已升级，请重新运行NSGA-II生成候选方案。")
        return
    optimization_info = st.session_state.get("phase_optimization_info", {})
    st.success(f"已生成 {len(candidates)} 组NSGA-II候选方案。")
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("优化算法", optimization_info.get("algorithm", "NSGA-II"))
    metric_b.metric("种群规模", optimization_info.get("population_size", population_size))
    metric_c.metric("进化代数", optimization_info.get("generations", generations))
    metric_d.metric("运行时间", f"{float(optimization_info.get('duration_seconds', 0)):.1f} 秒")
    summary_columns = [
        "候选方案", "Pareto层级", "目标晶相概率", "排除晶相概率", "综合评分",
        "数据距离", "工艺负荷", "风险等级", "预测晶相", "推荐理由",
    ]
    st.dataframe(candidates[summary_columns].round(4), use_container_width=True, hide_index=True)
    st.caption("综合评分只用于候选排序，不是模型准确率；Pareto层级越小越优，数据距离越小越接近已有样品。")
    st.markdown("#### Pareto权衡")
    st.scatter_chart(
        candidates,
        x="数据距离",
        y="目标晶相概率",
        color="风险等级",
        size="综合评分",
        use_container_width=True,
    )
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
    st.markdown("#### 实验方案生成")
    st.caption("系统会从候选中自动挑选互补方案，覆盖高目标概率、接近历史数据、低工艺负荷和工艺差异对照。方案仍需导师审核后再制样。")
    plan_count = st.slider(
        "计划样品数", min_value=3, max_value=min(6, len(candidates)),
        value=min(5, len(candidates)), key="experiment_plan_count",
    )
    experiment_plan = build_experiment_plan(candidates, plan_count)
    st.dataframe(
        experiment_plan[["样品编号", "实验组别", "实验状态", "目标晶相概率", "数据距离", "工艺负荷", "风险等级", "建议验证"]].round(4),
        use_container_width=True,
        hide_index=True,
    )
    st.warning("实验前请由导师确认原料、称料、熔融温度、保温时间、成核制度和晶化制度。模型推荐不能替代工艺安全判断。")
    plan_csv = experiment_plan.to_csv(index=False).encode("utf-8-sig")
    plan_download, plan_save = st.columns(2)
    plan_download.download_button(
        "下载实验计划 CSV",
        plan_csv,
        "las_experiment_plan.csv",
        "text/csv",
        use_container_width=True,
        key="download_experiment_plan",
    )
    if plan_save.button("生成待制样样品记录", type="primary", use_container_width=True, key="save_experiment_plan"):
        saved_count = save_experiment_plan(experiment_plan, target_options)
        st.success(f"已生成 {saved_count} 个待制样样品。请前往“样品记录”查看，并在“实验验证”中回填XRD结果。")
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
            "NSGA-II目标晶相设计批次",
            selected_name,
            "NSGA-II模型推荐",
            f"NSGA-II候选方案；Pareto层级{int(selected['Pareto层级'])}；{selected['风险提示']}；待制样和XRD验证",
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


def render_experiment_comparison(prediction: dict, observed_payload: dict, comparison: dict) -> None:
    observed_phases = set(observed_payload.get("phases", []))
    predicted_phases = set(comparison.get("predicted", []))
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("预测晶相", len(predicted_phases))
    metric_b.metric("实测晶相", len(observed_phases))
    metric_c.metric("共同晶相", len(predicted_phases & observed_phases))
    f1_value = comparison.get("f1")
    metric_d.metric("样品级 F1", "待核对" if f1_value is None else f"{float(f1_value):.3f}")
    st.markdown(f"#### 对比结论：{comparison.get('result', '待核对')}")
    rows = []
    for phase in prediction:
        predicted = phase in predicted_phases
        observed = phase in observed_phases
        if predicted and observed:
            result = "命中"
        elif predicted:
            result = "额外预测"
        elif observed:
            result = "漏判"
        else:
            result = "一致未出现"
        rows.append(
            {
                "晶相": phase_display_name(phase),
                "预测概率": prediction[phase].get("probability"),
                "模型判断": "可能出现" if predicted else "未判定为出现",
                "XRD实测": "检出" if observed else "未检出",
                "对比": result,
            }
        )
    st.dataframe(pd.DataFrame(rows).round({"预测概率": 4}), use_container_width=True, hide_index=True)
    missed = [phase_display_name(phase) for phase in comparison.get("missed", [])]
    extra = [phase_display_name(phase) for phase in comparison.get("extra", [])]
    if missed:
        st.warning("XRD检出但模型未判定：" + "、".join(missed))
    if extra:
        st.info("模型判定可能出现但XRD未检出：" + "、".join(extra))
    st.caption("该对比只判断晶相标签是否一致，不代表晶相含量误差，也不能替代XRD定量分析。")


def page_experiment_validation(data: pd.DataFrame) -> None:
    st.header("实验验证")
    st.caption("将模型预测与实际热处理和XRD结果对应，形成可追溯的预测验证记录")
    samples = read_table("SELECT * FROM samples ORDER BY created_at DESC")
    experiments = read_table("SELECT * FROM experiments ORDER BY updated_at DESC")
    if samples.empty:
        st.info("目前没有可验证的样品。请先在“晶相预测”或“目标晶相设计”中保存一个样品。")
        return

    verified_count = int(experiments["status"].isin(["已验证", "验证失败"]).sum()) if not experiments.empty else 0
    status_a, status_b, status_c, status_d = st.columns(4)
    status_a.metric("待验证样品", int((~samples["sample_id"].isin(experiments["sample_id"])).sum()))
    status_b.metric("实验记录", len(experiments))
    status_c.metric("已完成核对", verified_count)
    status_d.metric("XRD附件", int(experiments["xrd_file_path"].fillna("").ne("").sum()) if not experiments.empty else 0)

    sample_options = samples["sample_id"].tolist()
    sample_id = st.selectbox("选择待验证样品", sample_options, key="experiment_sample_id")
    sample_row = samples.loc[samples["sample_id"] == sample_id].iloc[0]
    prediction = load_latest_prediction(sample_id)
    planned_input = load_sample_input(sample_id)
    summary_a, summary_b, summary_c = st.columns(3)
    summary_a.markdown(f"**批次**  \n{sample_row['batch_name'] or '未填写'}")
    summary_b.markdown(f"**数据来源**  \n{sample_row['source'] or '未填写'}")
    summary_c.markdown(f"**当前状态**  \n{sample_row['status'] or '未填写'}")
    if not prediction:
        st.warning("该样品没有可读取的晶相预测，仍可记录实验，但无法自动完成预测对比。")

    entry_tab, history_tab = st.tabs(["录入与对比", "历史验证记录"])
    with entry_tab:
        existing_for_sample = experiments[experiments["sample_id"] == sample_id] if not experiments.empty else pd.DataFrame()
        default_experiment_id = f"XRD-{sample_id}-{datetime.now().strftime('%m%d')}"
        with st.form(f"experiment_form_{sample_id}"):
            first, second, third = st.columns(3)
            experiment_id = first.text_input("实验编号", value=default_experiment_id)
            experiment_date = second.date_input("实验日期", value=datetime.now().date())
            status = third.selectbox("实验状态", ["待实验", "实验中", "已完成XRD", "已验证", "验证失败"])
            st.markdown("#### 实际热处理条件")
            process_cols = st.columns(4)
            actual_process = {}
            for index, column in enumerate(PROCESS_COLUMNS):
                raw_default = pd.to_numeric(planned_input.get(column, 0), errors="coerce")
                default_value = float(raw_default) if pd.notna(raw_default) else 0.0
                actual_process[column] = process_cols[index].number_input(
                    PROCESS_LABELS[column],
                    min_value=0.0,
                    value=max(0.0, default_value),
                    step=1.0 if column.startswith("T_") else 0.25,
                    format="%.2f",
                    help=f"模型输入中的计划值为 {default_value:.2f}；此处填写实验实际执行值。",
                )
            st.markdown("#### XRD实测晶相")
            observed_phases = st.multiselect(
                "选择XRD检出的晶相",
                phase_target_columns(data),
                format_func=phase_display_name,
                help="只选择已有XRD依据的晶相；没有检测结果时保持为空。",
            )
            confirmed_empty = st.checkbox("已完成XRD，但未检出上述列表内晶相")
            xrd_file = st.file_uploader(
                "上传XRD图谱或检测报告",
                type=["png", "jpg", "jpeg", "pdf", "csv", "txt", "xy", "raw"],
                help="支持PNG、JPG、PDF、CSV、TXT、XY和RAW，单个文件不超过20 MB。",
            )
            notes = st.text_area("实验备注", placeholder="记录制样、气氛、升温速率、保温过程或异常情况", height=90)
            submitted = st.form_submit_button("保存实验验证记录", type="primary", use_container_width=True)
        if submitted:
            errors = []
            experiment_id = experiment_id.strip()
            if not experiment_id:
                errors.append("实验编号不能为空。")
            if actual_process["T_nucleation"] > 0 and actual_process["T_crystal"] > 0 and actual_process["T_nucleation"] >= actual_process["T_crystal"]:
                errors.append("实际成核温度必须低于实际晶化温度。")
            if observed_phases and confirmed_empty:
                errors.append("已经选择实测晶相时，不能同时勾选“未检出列表内晶相”。")
            if status in {"已验证", "验证失败"} and not observed_phases and not confirmed_empty:
                errors.append("标记为已验证或验证失败前，需要录入XRD晶相或确认未检出列表内晶相。")
            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    comparison = compare_phase_result(prediction, observed_phases, confirmed_empty)
                    xrd_file_path = save_xrd_file(xrd_file, experiment_id)
                    save_experiment_record(
                        experiment_id,
                        sample_id,
                        experiment_date.isoformat(),
                        status,
                        actual_process,
                        observed_phases,
                        confirmed_empty,
                        comparison,
                        xrd_file_path,
                        notes,
                    )
                    st.success(f"实验记录 {experiment_id} 已保存。")
                    observed_payload = {"phases": observed_phases, "confirmed_empty": confirmed_empty}
                    if prediction:
                        render_experiment_comparison(prediction, observed_payload, comparison)
                except (OSError, ValueError, sqlite3.Error) as exc:
                    st.error(f"实验记录保存失败：{exc}")
        elif not existing_for_sample.empty:
            latest = existing_for_sample.iloc[0]
            latest_observed = parse_json(latest["observed_json"], {})
            latest_comparison = parse_json(latest["comparison_json"], {})
            st.markdown("#### 最近一次验证结果")
            st.caption(f"实验编号：{latest['experiment_id']}，更新时间：{latest['updated_at']}")
            if prediction:
                render_experiment_comparison(prediction, latest_observed, latest_comparison)

    with history_tab:
        if experiments.empty:
            st.info("尚无实验验证记录。")
        else:
            export_table = experiment_export_table(experiments)
            selected_status = st.selectbox(
                "按状态筛选",
                ["全部"] + ["待实验", "实验中", "已完成XRD", "已验证", "验证失败"],
                key="experiment_status_filter",
            )
            history_view = export_table if selected_status == "全部" else export_table[export_table["状态"] == selected_status]
            st.dataframe(history_view, use_container_width=True, hide_index=True)
            st.download_button(
                "下载实验验证台账 CSV",
                export_table.to_csv(index=False).encode("utf-8-sig"),
                "las_experiment_validation.csv",
                "text/csv",
                use_container_width=False,
            )
            experiment_choice = st.selectbox("查看记录附件", experiments["experiment_id"].tolist(), key="experiment_attachment_choice")
            chosen = experiments.loc[experiments["experiment_id"] == experiment_choice].iloc[0]
            attachment = str(chosen["xrd_file_path"] or "")
            if attachment:
                attachment_path = ROOT / attachment
                if attachment_path.exists():
                    if attachment_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        st.image(str(attachment_path), caption=f"{experiment_choice} 的XRD附件", use_container_width=True)
                    st.download_button(
                        "下载该XRD附件",
                        attachment_path.read_bytes(),
                        attachment_path.name,
                        use_container_width=False,
                    )
                else:
                    st.warning("数据库中存在附件路径，但当前运行环境找不到该文件。")
            else:
                st.info("该实验记录没有上传XRD附件。")


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
    tabs = st.tabs(["总览", "晶相预测", "目标晶相设计", "数据中心", "样品记录", "实验验证", "模型评估"])
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
        page_experiment_validation(data)
    with tabs[6]:
        page_model_eval(data, artifacts)


if __name__ == "__main__":
    main()
