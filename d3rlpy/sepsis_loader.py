"""Sepsis dataset loader for MIMIC-IV based sepsis treatment data.

This module provides utilities to load and preprocess the MIMIC-IV sepsis cohort
data for offline reinforcement learning experiments.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import (
    Episode,
    FIFOBuffer,
    ReplayBuffer,
    TrajectorySlicerProtocol,
    TransitionPickerProtocol,
)

__all__ = [
    "get_sepsis",
    "get_sepsis_fold",
    "OBSERVATION_COLUMNS",
    "compute_sofa_scores",
    "discretize_actions",
]


# 47 clinical observation features (no patient ID — icustayid excluded)
OBSERVATION_COLUMNS = [
    # Time-varying vitals and labs (34 features)
    "GCS", "HR", "SysBP", "DiaBP", "MeanBP", "RR", "Temp_C",
    "FiO2_100", "FiO2_1", "Potassium", "Sodium", "Chloride", "Glucose",
    "INR", "Magnesium", "Calcium", "Hb", "WBC_count", "Platelets_count",
    "PTT", "PT", "Arterial_pH", "Arterial_lactate", "paO2", "paCO2",
    "PaO2_FiO2", "HCO3", "SpO2", "BUN", "Creatinine", "SGOT",
    "SGPT", "Total_bili", "Arterial_BE",
    # SOFA subscores computed by compute_sofa_scores() (7 features)
    "paO2_FiO2", "SOFA_resp", "SOFA_coag", "SOFA_liver",
    "SOFA_cardio", "SOFA_cns", "SOFA_renal",
    # Demographics (6 features)
    "age", "gender", "Weight_kg", "mechvent", "extubated", "re_admission",
]


def compute_sofa_scores(
    df: pd.DataFrame,
    timestep_resolution: float = 4.0,
) -> pd.DataFrame:
    """Compute SOFA scores for sepsis severity assessment.
    
    Implements the Sequential Organ Failure Assessment (SOFA) score calculation
    following clinical guidelines. The score ranges from 0-24 with higher values
    indicating more severe organ dysfunction.
    
    Args:
        df: DataFrame containing required medical measurements.
        timestep_resolution: Hours between successive measurements (default: 4.0).
        
    Returns:
        DataFrame with added SOFA subscore and total columns.
        
    Note:
        Requires columns: paO2, FiO2_1, Platelets_count, Total_bili, MeanBP,
        max_dose_vaso, GCS, Creatinine, output_step
    """
    df = df.copy()

    # Respiratory (PaO2/FiO2 ratio)
    df["paO2_FiO2"] = df["paO2"] / df["FiO2_1"]
    pfi = df["paO2_FiO2"]
    df["SOFA_resp"] = np.select(
        [pfi > 400,
         (pfi >= 300) & (pfi < 400),
         (pfi >= 200) & (pfi < 300),
         (pfi >= 100) & (pfi < 200)],
        [0, 1, 2, 3],
        default=4
    ).astype(float)

    # Coagulation (Platelets)
    pl = df["Platelets_count"]
    df["SOFA_coag"] = np.select(
        [pl > 150,
         (pl >= 100) & (pl < 150),
         (pl >= 50) & (pl < 100),
         (pl >= 20) & (pl < 50)],
        [0, 1, 2, 3],
        default=4
    ).astype(float)

    # Liver (Bilirubin)
    bili = df["Total_bili"]
    df["SOFA_liver"] = np.select(
        [bili < 1.2,
         (bili >= 1.2) & (bili < 2.0),
         (bili >= 2.0) & (bili < 6.0),
         (bili >= 6.0) & (bili < 12.0)],
        [0, 1, 2, 3],
        default=4
    ).astype(float)

    # Cardiovascular (MAP and vasopressors)
    map_ = df["MeanBP"]
    vaso = df["max_dose_vaso"]
    df["SOFA_cardio"] = np.select(
        [map_ >= 70,
         (map_ < 70) & (map_ >= 65),
         map_ < 65,
         (vaso > 0) & (vaso <= 0.1),
         vaso > 0.1],
        [0, 1, 2, 3, 4]
    ).astype(float)

    # Central Nervous System (GCS)
    gcs = df["GCS"]
    df["SOFA_cns"] = np.select(
        [gcs > 14,
         (gcs > 12) & (gcs <= 14),
         (gcs > 9) & (gcs <= 12),
         (gcs > 5) & (gcs <= 9)],
        [0, 1, 2, 3],
        default=4
    ).astype(float)

    # Renal (Creatinine and urine output)
    cr = df["Creatinine"]
    uo = df["output_step"]
    thr3 = 500 * timestep_resolution / 24  # <500 mL/day threshold
    thr4 = 200 * timestep_resolution / 24  # <200 mL/day threshold

    df["SOFA_renal"] = np.select(
        [cr < 1.2,
         (cr >= 1.2) & (cr < 2.0),
         (cr >= 2.0) & (cr < 3.5),
         ((cr >= 3.5) & (cr < 5.0)) | (uo < thr3),
         (cr > 5.0) | (uo < thr4)],
        [0, 1, 2, 3, 4]
    ).astype(float)

    # Total SOFA score (sum of subscores)
    sofa_cols = [
        "SOFA_resp", "SOFA_coag", "SOFA_liver",
        "SOFA_cardio", "SOFA_cns", "SOFA_renal"
    ]
    df["SOFA_total"] = df[sofa_cols].sum(axis=1, skipna=False)

    return df


def discretize_actions(
    df: pd.DataFrame,
    fluid_col: str = "output_step",
    vaso_col: str = "median_dose_vaso",
) -> pd.DataFrame:
    """Discretize continuous treatment actions into 25 bins (5x5 grid).
    
    Maps fluid administration and vasopressor dosage into a discrete action space
    using clinically-motivated quantile-based binning.
    
    Args:
        df: DataFrame with continuous action columns.
        fluid_col: Name of fluid administration column (mL per timestep).
        vaso_col: Name of vasopressor dosage column (mcg/kg/min).
        
    Returns:
        DataFrame with added 'action_bin' column (0-24).
        
    Note:
        Fluid bins: [0, 50, 152.14, 500, inf] mL per timestep
        Vaso bins: [0, 0.08, 0.20, 0.45, inf] mcg/kg/min
        action_bin = fluid_bin * 5 + vaso_bin
    """
    df = df.copy()
    
    # Clinically-informed bin edges
    fluid_edges = [0, 50.00, 152.14, 500.00]  # mL per timestep
    vaso_edges = [0, 0.08, 0.20, 0.45]  # mcg/kg/min

    df['action_bin'] = (
        5 * np.digitize(df[fluid_col].fillna(0).values, fluid_edges, right=True)
        + np.digitize(df[vaso_col].fillna(0).values, vaso_edges, right=True)
    ).astype('int8')

    return df


def _pad_to_regular_grid(
    df: pd.DataFrame,
    freq: str = "4h",
    id_col: str = "icustayid",
    time_col: str = "timestep_dt",
    action_cols: Optional[list[str]] = None,
    ffill_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Pad ICU stays to regular 4-hour grid using forward-fill.
    
    Creates a regular time grid for each ICU stay, forward-filling observations
    and zeroing actions on synthetic timesteps.
    
    Args:
        df: DataFrame with ICU stay data.
        freq: Resampling frequency (default: "4h").
        id_col: Column name for ICU stay identifier.
        time_col: Column name for timestamp.
        action_cols: Columns to zero on padded rows.
        ffill_cols: Columns to forward-fill.
        
    Returns:
        DataFrame with regular time grid and 'is_padding' flag.
    """
    action_cols = action_cols or []
    ffill_cols = ffill_cols or []

    df = df.copy()
    df["_orig"] = 1  # Sentinel for real rows
    df = df.sort_values([id_col, time_col]).set_index(time_col)

    # Build regular grid
    padded = (
        df.groupby(id_col)
        .resample(freq)
        .first()
    )
    padded["is_padding"] = padded["_orig"].isna()

    # Restore keys and forward-fill state
    padded[id_col] = padded[id_col].ffill()
    if ffill_cols:
        padded[ffill_cols] = padded[ffill_cols].ffill().infer_objects(copy=False)

    # Zero out actions on padded rows
    for c in action_cols:
        padded.loc[padded["is_padding"], c] = 0

    # Calculate time deltas
    idx_time = padded.index.get_level_values(time_col)
    padded["delta_hours"] = (
        idx_time
        .to_series(index=padded.index)
        .groupby(padded[id_col])
        .diff()
        .dt.total_seconds()
        .div(3600)
    )

    # Flatten MultiIndex
    padded = (
        padded.reset_index(level=1)
        .rename(columns={time_col: time_col})
        .reset_index(drop=True)
        .drop(columns="_orig")
        .sort_values([id_col, time_col])
    )
    
    return padded


def _dataframe_to_replaybuffer(
    df: pd.DataFrame,
    obs_cols: list[str],
    action_cols: list[str],
    reward_col: str,
    action_space: ActionSpace,
    action_size: int,
    time_col: str = "timestep_dt",
    id_col: str = "icustayid",
    outcome_col: str = "morta_90",
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
) -> ReplayBuffer:
    """Convert preprocessed DataFrame to d3rlpy ReplayBuffer.

    Args:
        df: Preprocessed sepsis data.
        obs_cols: Observation column names.
        action_cols: Action column names.
        reward_col: Reward column name.
        action_space: DISCRETE or CONTINUOUS.
        action_size: Number of discrete actions or continuous action dimensions.
        time_col: Timestamp column name.
        id_col: ICU stay identifier column name.
        outcome_col: Binary mortality column (1=died). Used to set terminated flag.
        transition_picker: Optional transition picker override.
        trajectory_slicer: Optional trajectory slicer override.

    Returns:
        ReplayBuffer containing episodic ICU stay data.
    """
    df = df.sort_values([id_col, time_col])

    episodes = []
    for stay_id, g in df.groupby(id_col, sort=False):
        obs = g[obs_cols].to_numpy(dtype=np.float32)
        acts = g[action_cols].to_numpy(dtype=np.float32)

        # Flatten discrete actions to scalar
        if action_space == ActionSpace.DISCRETE and acts.shape[1] == 1:
            acts = acts.flatten().astype(np.int32)

        rews = g[reward_col].to_numpy(dtype=np.float32)

        # Death = absorbing terminal state; survival = episode end (not terminal)
        died = bool(g[outcome_col].iloc[-1]) if outcome_col in g.columns else True

        ep = Episode(
            observations=obs,
            actions=acts,
            rewards=rews,
            terminated=died,
        )
        episodes.append(ep)

    buf = ReplayBuffer(
        buffer=FIFOBuffer(limit=sum(len(ep) for ep in episodes)),
        episodes=episodes,
        action_space=action_space,
        action_size=action_size,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
    )
    return buf


def get_sepsis(
    data_dir: Optional[str] = None,
    action_space: ActionSpace = ActionSpace.DISCRETE,
    reward_mode: str = "terminal",
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    icu_id_filter: Optional[set] = None,
) -> ReplayBuffer:
    """Load MIMIC-IV sepsis cohort dataset.

    Each episode = one ICU stay at 4-hour timesteps.
    Observations: 47 clinical features (vitals, labs, SOFA subscores, demographics).
    Actions: Discrete (25 bins: 5 fluids × 5 vasopressors) or Continuous (2D).

    Args:
        data_dir: Directory with mimic_dataset.csv and sepsis_cohort.csv.
                  Falls back to SEPSIS_DATA_DIR env var, then <repo>/data/.
        action_space: DISCRETE (25 actions) or CONTINUOUS (2D).
        reward_mode: "terminal" (+1 survival / -1 death at last timestep, 0 elsewhere),
                     "dense" (negative SOFA score each step: lower severity = higher reward),
                     or "mixed" (α=0.5 × SOFA-change + terminal ±15).
        transition_picker: Override for transition sampling.
        trajectory_slicer: Override for trajectory sampling.
        icu_id_filter: If given, restrict the cohort to these icustayid values
                       (used by get_sepsis_fold to build train/test splits from
                       the exact same pipeline).

    Returns:
        ReplayBuffer with sepsis episodes.

    Raises:
        FileNotFoundError: If CSV files are not found.
        ValueError: If reward_mode is invalid.
    """
    if reward_mode not in ("terminal", "dense", "mixed"):
        raise ValueError(f"reward_mode must be 'terminal', 'dense', or 'mixed', got {reward_mode!r}")

    # Determine data directory
    if data_dir is None:
        if "SEPSIS_DATA_DIR" in os.environ:
            data_dir = os.environ["SEPSIS_DATA_DIR"]
        else:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_dir = os.path.join(repo_root, "data")

    mimic_path = os.path.join(data_dir, "mimic_dataset.csv")
    cohort_path = os.path.join(data_dir, "sepsis_cohort.csv")

    if not os.path.exists(mimic_path):
        raise FileNotFoundError(
            f"MIMIC dataset not found at {mimic_path}. "
            f"Set SEPSIS_DATA_DIR or pass data_dir."
        )
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(
            f"Sepsis cohort not found at {cohort_path}. "
            f"Set SEPSIS_DATA_DIR or pass data_dir."
        )

    print(f"Loading MIMIC dataset from {mimic_path}...")
    mimic_df = pd.read_csv(mimic_path)

    print(f"Loading sepsis cohort from {cohort_path}...")
    sepsis_cohort = pd.read_csv(cohort_path)

    mimic_df.loc[mimic_df["extubated"].isna(), "extubated"] = 0

    print("Computing SOFA scores...")
    mimic_df = compute_sofa_scores(mimic_df, timestep_resolution=4.0)

    print("Filtering to sepsis cohort...")
    sepsis_icu_ids = sepsis_cohort["icustayid"].unique()
    if icu_id_filter is not None:
        sepsis_icu_ids = np.array([i for i in sepsis_icu_ids if i in icu_id_filter])
    sepsis_df = mimic_df[mimic_df["icustayid"].isin(sepsis_icu_ids)].copy()

    print(f"Sepsis cohort: {len(sepsis_icu_ids)} ICU stays, {len(sepsis_df)} timesteps")

    sepsis_df["timestep_dt"] = pd.to_datetime(sepsis_df["timestep"], unit="s", utc=True)

    action_cols = ["output_total", "output_step", "median_dose_vaso", "max_dose_vaso"]
    outcome_col = "morta_90"

    # Build reward column
    if reward_mode == "terminal":
        # +1 at last timestep if survived, -1 if died, 0 elsewhere
        sepsis_df["_reward"] = 0.0
        last_step_mask = ~sepsis_df.duplicated(subset=["icustayid"], keep="last")
        sepsis_df.loc[last_step_mask & (sepsis_df[outcome_col] == 0), "_reward"] = 1.0
        sepsis_df.loc[last_step_mask & (sepsis_df[outcome_col] == 1), "_reward"] = -1.0
        reward_col = "_reward"
    elif reward_mode == "dense":
        # Dense: negative SOFA so lower severity → higher reward
        sepsis_df["_reward"] = -sepsis_df["SOFA"].astype(np.float32)
        reward_col = "_reward"
    else:  # "mixed"
        # Mixed: α=0.5 * SOFA-delta + terminal ±15
        # Compute SOFA delta within each ICU stay (SOFA_t - SOFA_{t+1})
        sepsis_df["SOFA_next"] = sepsis_df.groupby("icustayid")["SOFA"].shift(-1)
        sepsis_df["SOFA_delta"] = sepsis_df["SOFA"] - sepsis_df["SOFA_next"]

        # All timesteps get SOFA-delta reward (0.5 * delta)
        # Last timestep (NaN delta) gets only terminal ±15
        sepsis_df["_reward"] = 0.5 * sepsis_df["SOFA_delta"].fillna(0.0)

        # Add terminal component at last step
        last_step_mask = ~sepsis_df.duplicated(subset=["icustayid"], keep="last")
        sepsis_df.loc[last_step_mask & (sepsis_df[outcome_col] == 0), "_reward"] += 15.0
        sepsis_df.loc[last_step_mask & (sepsis_df[outcome_col] == 1), "_reward"] -= 15.0

        sepsis_df = sepsis_df.drop(columns=["SOFA_next", "SOFA_delta"])
        reward_col = "_reward"

    # Ensure no NaN rewards (can occur from padding)
    sepsis_df["_reward"] = sepsis_df["_reward"].fillna(0.0).astype(np.float32)

    required_cols = (
        OBSERVATION_COLUMNS + action_cols
        + [reward_col, outcome_col, "timestep_dt", "timestep", "icustayid"]
    )
    filtered_df = sepsis_df[[c for c in required_cols if c in sepsis_df.columns]].copy()

    # Double-check: no NaN in reward column before conversion
    if filtered_df[reward_col].isna().any():
        print(f"WARNING: {filtered_df[reward_col].isna().sum()} NaN values in reward column. Filling with 0.")
        filtered_df[reward_col] = filtered_df[reward_col].fillna(0.0)

    print("Padding to regular 4-hour grid...")
    padded_df = _pad_to_regular_grid(
        filtered_df,
        freq="4h",
        action_cols=action_cols,
        ffill_cols=OBSERVATION_COLUMNS,
    )

    # Padding may introduce NaN in reward column; fill with 0
    padded_df[reward_col] = padded_df[reward_col].fillna(0.0)

    if action_space == ActionSpace.DISCRETE:
        print("Discretizing actions to 25 bins (5x5 grid)...")
        padded_df = discretize_actions(padded_df)
        buffer_action_cols = ["action_bin"]
        action_size = 25
    else:
        print("Using continuous actions (fluid + vasopressor)...")
        buffer_action_cols = ["output_step", "median_dose_vaso"]
        action_size = 2

    print("Converting to ReplayBuffer...")
    buffer = _dataframe_to_replaybuffer(
        padded_df,
        obs_cols=OBSERVATION_COLUMNS,
        action_cols=buffer_action_cols,
        reward_col=reward_col,
        action_space=action_space,
        action_size=action_size,
        outcome_col=outcome_col,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
    )
    print(f"✓ Loaded {len(buffer.episodes)} episodes, {buffer.transition_count} transitions")
    return buffer


def get_sepsis_fold(
    data_dir: Optional[str] = None,
    fold: int = 0,
    n_splits: int = 5,
    action_space: ActionSpace = ActionSpace.DISCRETE,
    reward_mode: str = "terminal",
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
) -> Tuple[ReplayBuffer, ReplayBuffer]:
    """Load MIMIC-IV sepsis cohort with 5-fold cross-validation split.

    Stratifies episodes by survival outcome (icustay_died) to ensure balanced
    train/test sets across mortality rates.

    Args:
        data_dir: Path to mimic_dataset.csv and sepsis_cohort.csv.
        fold: Fold index in [0, n_splits), selects test fold; others train.
        n_splits: Number of folds (default 5).
        action_space: DISCRETE (25 actions) or CONTINUOUS.
        reward_mode: "terminal", "dense", or "mixed".
        transition_picker: Override for transition sampling.
        trajectory_slicer: Override for trajectory sampling.

    Returns:
        (train_buffer, test_buffer): ReplayBuffers for training and evaluation.

    Raises:
        FileNotFoundError: If CSV files not found.
        ValueError: If fold >= n_splits.
    """
    if fold >= n_splits:
        raise ValueError(f"fold={fold} must be < n_splits={n_splits}")

    # Determine data directory
    if data_dir is None:
        if "SEPSIS_DATA_DIR" in os.environ:
            data_dir = os.environ["SEPSIS_DATA_DIR"]
        else:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_dir = os.path.join(repo_root, "data")

    mimic_path = os.path.join(data_dir, "mimic_dataset.csv")
    cohort_path = os.path.join(data_dir, "sepsis_cohort.csv")

    if not os.path.exists(mimic_path) or not os.path.exists(cohort_path):
        raise FileNotFoundError(f"MIMIC/cohort CSV not found in {data_dir}")

    # Determine per-ICU-stay outcome for stratification, from the same source
    # get_sepsis uses (morta_90 in mimic_dataset.csv, restricted to the cohort).
    print(f"Loading cohort IDs + outcomes from {mimic_path} / {cohort_path}...")
    sepsis_cohort = pd.read_csv(cohort_path)
    mimic_df = pd.read_csv(mimic_path, usecols=["icustayid", "morta_90"])

    cohort_ids = set(sepsis_cohort["icustayid"].unique())
    per_stay = (
        mimic_df[mimic_df["icustayid"].isin(cohort_ids)]
        .groupby("icustayid")["morta_90"]
        .last()  # 90-day mortality is constant within a stay; last() is robust to padding
    )
    icu_ids = per_stay.index.to_numpy()
    outcomes = per_stay.to_numpy()

    # Stratified fold split by 90-day mortality outcome
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    fold_indices = list(skf.split(icu_ids, outcomes))
    train_idx, test_idx = fold_indices[fold]
    train_ids = set(icu_ids[train_idx].tolist())
    test_ids = set(icu_ids[test_idx].tolist())

    print(
        f"Fold {fold}/{n_splits}: train {len(train_ids)} stays "
        f"(died {int(outcomes[train_idx].sum())}), "
        f"test {len(test_ids)} stays (died {int(outcomes[test_idx].sum())})"
    )

    # Build each split through the exact same pipeline as get_sepsis, just
    # restricted to the fold's ICU stays via icu_id_filter.
    print("Building train split...")
    train_buffer = get_sepsis(
        data_dir=data_dir,
        action_space=action_space,
        reward_mode=reward_mode,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
        icu_id_filter=train_ids,
    )
    print("Building test split...")
    test_buffer = get_sepsis(
        data_dir=data_dir,
        action_space=action_space,
        reward_mode=reward_mode,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
        icu_id_filter=test_ids,
    )

    return train_buffer, test_buffer
