"""Tests for sepsis dataset loader."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from d3rlpy.constants import ActionSpace
from d3rlpy.sepsis_loader import (
    OBSERVATION_COLUMNS,
    compute_sofa_scores,
    discretize_actions,
    get_sepsis,
)


def test_observation_columns():
    """Test that observation columns are defined correctly."""
    assert len(OBSERVATION_COLUMNS) == 49
    assert "GCS" in OBSERVATION_COLUMNS
    assert "SOFA_resp" in OBSERVATION_COLUMNS
    assert "icustayid" in OBSERVATION_COLUMNS


def test_compute_sofa_scores():
    """Test SOFA score computation."""
    # Create minimal test data
    df = pd.DataFrame({
        "paO2": [100.0, 200.0],
        "FiO2_1": [0.5, 0.5],
        "Platelets_count": [100.0, 200.0],
        "Total_bili": [1.0, 2.5],
        "MeanBP": [70.0, 60.0],
        "max_dose_vaso": [0.0, 0.15],
        "GCS": [15.0, 10.0],
        "Creatinine": [1.0, 3.0],
        "output_step": [100.0, 50.0],
    })
    
    result = compute_sofa_scores(df, timestep_resolution=4.0)
    
    # Check that SOFA columns are added
    assert "SOFA_resp" in result.columns
    assert "SOFA_coag" in result.columns
    assert "SOFA_liver" in result.columns
    assert "SOFA_cardio" in result.columns
    assert "SOFA_cns" in result.columns
    assert "SOFA_renal" in result.columns
    assert "SOFA_total" in result.columns
    
    # Check value ranges
    assert result["SOFA_total"].min() >= 0
    assert result["SOFA_total"].max() <= 24


def test_discretize_actions():
    """Test action discretization."""
    df = pd.DataFrame({
        "output_step": [0.0, 100.0, 300.0],
        "median_dose_vaso": [0.0, 0.1, 0.3],
    })
    
    result = discretize_actions(df)
    
    assert "action_bin" in result.columns
    assert result["action_bin"].dtype == np.int8
    assert result["action_bin"].min() >= 0
    assert result["action_bin"].max() < 25


def test_discretize_actions_with_nans():
    """Test action discretization handles NaN values."""
    df = pd.DataFrame({
        "output_step": [0.0, np.nan, 300.0],
        "median_dose_vaso": [np.nan, 0.1, 0.3],
    })
    
    result = discretize_actions(df)
    
    # NaNs should be treated as 0
    assert not result["action_bin"].isna().any()
    assert result["action_bin"][0] == 0  # Both zeros


@pytest.mark.skipif(
    not os.path.exists(
        os.environ.get("SEPSIS_DATA_DIR", "../../../data")
    ),
    reason="Sepsis data not available"
)
def test_get_sepsis_discrete():
    """Test loading sepsis dataset with discrete actions."""
    # This test requires actual data files
    try:
        dataset = get_sepsis(action_space=ActionSpace.DISCRETE)
        
        assert len(dataset.episodes) > 0
        assert dataset.transition_count > 0
        assert dataset.dataset_info.action_space == ActionSpace.DISCRETE
        assert dataset.dataset_info.action_size == 25
        
        # Check observation shape
        sample = dataset.sample_transition()
        assert sample.observation.shape[-1] == 49
        
    except FileNotFoundError:
        pytest.skip("Sepsis data files not found")


@pytest.mark.skipif(
    not os.path.exists(
        os.environ.get("SEPSIS_DATA_DIR", "../../../data")
    ),
    reason="Sepsis data not available"
)
def test_get_sepsis_continuous():
    """Test loading sepsis dataset with continuous actions."""
    try:
        dataset = get_sepsis(action_space=ActionSpace.CONTINUOUS)
        
        assert len(dataset.episodes) > 0
        assert dataset.transition_count > 0
        assert dataset.dataset_info.action_space == ActionSpace.CONTINUOUS
        assert dataset.dataset_info.action_size == 2
        
        # Check action shape
        sample = dataset.sample_transition()
        assert sample.action.shape[-1] == 2
        
    except FileNotFoundError:
        pytest.skip("Sepsis data files not found")


def test_get_sepsis_missing_files():
    """Test that appropriate error is raised when files are missing."""
    with pytest.raises(FileNotFoundError, match="MIMIC dataset not found"):
        get_sepsis(data_dir="/nonexistent/path")
