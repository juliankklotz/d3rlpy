"""Sepsis dataset loader for MIMIC-IV based sepsis treatment data.

This module provides utilities to load and preprocess the MIMIC-IV sepsis cohort
data for offline reinforcement learning experiments.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

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
    "OBSERVATION_COLUMNS",
    "compute_sofa_scores",
    "discretize_actions",
]


# Define observation features (49 total)
OBSERVATION_COLUMNS = [
    # Time-varying vitals and labs (34 features)
    "GCS", "HR", "SysBP", "DiaBP", "MeanBP", "RR", "Temp_C",
    "FiO2_100", "FiO2_1", "Potassium", "Sodium", "Chloride", "Glucose",
    "INR", "Magnesium", "Calcium", "Hb", "WBC_count", "Platelets_count",
    "PTT", "PT", "Arterial_pH", "Arterial_lactate", "paO2", "paCO2",
    "PaO2_FiO2", "HCO3", "SpO2", "BUN", "Creatinine", "SGOT",
    "SGPT", "Total_bili", "Arterial_BE",
    # SOFA subscores (7 features)
    "paO2_FiO2", "SOFA_resp", "SOFA_coag", "SOFA_liver",
    "SOFA_cardio", "SOFA_cns", "SOFA_renal",
    # Demographics (4 features)
    "age", "gender", "Weight_kg", "mechvent", "extubated", "re_admission",
    # ICU stay ID (1 feature)
    "icustayid",
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
) -> ReplayBuffer:
    """Convert preprocessed DataFrame to d3rlpy ReplayBuffer.
    
    Args:
        df: Preprocessed sepsis data.
        obs_cols: Observation column names.
        action_cols: Action column names.
        reward_col: Reward column name (typically SOFA score).
        action_space: DISCRETE or CONTINUOUS.
        action_size: Number of discrete actions or continuous action dimensions.
        time_col: Timestamp column name.
        id_col: ICU stay identifier column name.
        
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

        ep = Episode(
            observations=obs,
            actions=acts,
            rewards=rews,
            terminated=True,
        )
        episodes.append(ep)

    buf = ReplayBuffer(
        buffer=FIFOBuffer(limit=sum(len(ep) for ep in episodes)),
        episodes=episodes,
        action_space=action_space,
        action_size=action_size,
    )
    return buf


def get_sepsis(
    data_dir: Optional[str] = None,
    action_space: ActionSpace = ActionSpace.DISCRETE,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
) -> ReplayBuffer:
    """Load MIMIC-IV sepsis cohort dataset.
    
    Returns a ReplayBuffer containing ICU stay episodes for sepsis patients.
    Each episode represents one ICU stay with 4-hour timesteps containing:
    - Observations: 49 features (vitals, labs, SOFA subscores, demographics)
    - Actions: Discrete (25 bins) or Continuous (2D: fluid + vasopressor)
    - Rewards: SOFA score (0-24, higher = worse)
    
    Args:
        data_dir: Path to directory containing mimic_dataset.csv and sepsis_cohort.csv.
                  If None, uses environment variable or default path.
        action_space: DISCRETE (25 actions) or CONTINUOUS (2D).
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.
        
    Returns:
        ReplayBuffer with sepsis episodes.
        
    Raises:
        FileNotFoundError: If CSV files are not found.
        ValueError: If required columns are missing.
        
    Example:
        >>> from d3rlpy.datasets import get_sepsis
        >>> from d3rlpy.constants import ActionSpace
        >>> 
        >>> # Discrete action space (5x5 treatment grid)
        >>> dataset = get_sepsis(action_space=ActionSpace.DISCRETE)
        >>> print(f"Episodes: {len(dataset.episodes)}")
        >>> print(f"Transitions: {dataset.transition_count}")
        >>> 
        >>> # Continuous action space (fluid + vasopressor)
        >>> dataset = get_sepsis(action_space=ActionSpace.CONTINUOUS)
    """
    # Determine data directory
    if data_dir is None:
        # Try environment variable first
        if "SEPSIS_DATA_DIR" in os.environ:
            data_dir = os.environ["SEPSIS_DATA_DIR"]
        # Fall back to default relative path
        else:
            # Assume standard d3rlpy repo structure
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_dir = os.path.join(repo_root, "data")
    
    mimic_path = os.path.join(data_dir, "mimic_dataset.csv")
    cohort_path = os.path.join(data_dir, "sepsis_cohort.csv")
    
    # Check file existence
    if not os.path.exists(mimic_path):
        raise FileNotFoundError(
            f"MIMIC dataset not found at {mimic_path}. "
            f"Set SEPSIS_DATA_DIR environment variable or provide data_dir argument."
        )
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(
            f"Sepsis cohort not found at {cohort_path}. "
            f"Set SEPSIS_DATA_DIR environment variable or provide data_dir argument."
        )
    
    # Load data
    print(f"Loading MIMIC dataset from {mimic_path}...")
    mimic_df = pd.read_csv(mimic_path)
    
    print(f"Loading sepsis cohort from {cohort_path}...")
    sepsis_cohort = pd.read_csv(cohort_path)
    
    # Fix missing values
    mimic_df.loc[mimic_df["extubated"].isna(), "extubated"] = 0
    
    # Compute SOFA scores
    print("Computing SOFA scores...")
    mimic_df = compute_sofa_scores(mimic_df, timestep_resolution=4.0)
    
    # Filter to sepsis cohort
    print("Filtering to sepsis cohort...")
    sepsis_icu_ids = sepsis_cohort["icustayid"].unique()
    sepsis_df = mimic_df[mimic_df["icustayid"].isin(sepsis_icu_ids)].copy()
    
    print(f"Sepsis cohort: {len(sepsis_icu_ids)} ICU stays, {len(sepsis_df)} timesteps")
    
    # Convert timestamps
    sepsis_df["timestep_dt"] = pd.to_datetime(sepsis_df['timestep'], unit='s', utc=True)
    
    # Define action columns
    action_cols = ["output_total", "output_step", "median_dose_vaso", "max_dose_vaso"]
    
    # Select required columns
    required_cols = OBSERVATION_COLUMNS + action_cols + ["SOFA", "timestep_dt", "timestep"]
    filtered_df = sepsis_df[required_cols].copy()
    
    # Pad to regular 4-hour grid
    print("Padding to regular 4-hour grid...")
    padded_df = _pad_to_regular_grid(
        filtered_df,
        freq="4h",
        action_cols=action_cols,
        ffill_cols=OBSERVATION_COLUMNS,
    )
    
    # Process based on action space
    if action_space == ActionSpace.DISCRETE:
        print("Discretizing actions to 25 bins (5x5 grid)...")
        padded_df = discretize_actions(padded_df)
        buffer_action_cols = ["action_bin"]
        action_size = 25
    else:  # CONTINUOUS
        print("Using continuous actions (fluid + vasopressor)...")
        buffer_action_cols = ["output_step", "median_dose_vaso"]
        action_size = 2
    
    # Convert to ReplayBuffer
    print("Converting to ReplayBuffer...")
    buffer = _dataframe_to_replaybuffer(
        padded_df,
        obs_cols=OBSERVATION_COLUMNS,
        action_cols=buffer_action_cols,
        reward_col="SOFA",
        action_space=action_space,
        action_size=action_size,
    )
    
    # Apply transition picker and trajectory slicer if provided
    if transition_picker is not None:
        buffer._transition_picker = transition_picker
    if trajectory_slicer is not None:
        buffer._trajectory_slicer = trajectory_slicer
    
    print(f"✓ Loaded {len(buffer.episodes)} episodes, {buffer.transition_count} transitions")
    
    return buffer
