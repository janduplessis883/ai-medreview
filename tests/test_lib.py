import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, mock_open

# Create a fixture to mock the load_google_sheet function
@pytest.fixture
def mock_google_sheet_data():
    data = {
        "time": ["2023-04-01 10:00:00", "2023-04-02 11:00:00"],
        "rating": ["Likely", "Extremely likely"],
        "free_text": ["This is a test free text", "Another test free text"],
        "do_better": ["Improve staff attitude", "No suggestions"],
        "surgery": [True, False],
    }
    return pd.DataFrame(data)

# Create a fixture to mock the load_local_data function
@pytest.fixture
def mock_local_data():
    data = {
        "time": ["2023-04-01 10:00:00"],
        "rating": ["Likely"],
        "free_text": ["This is a test free text"],
        "do_better": ["Improve staff attitude"],
        "surgery": [True],
        # Add other columns as needed
    }
    return pd.DataFrame(data)

# Create a fixture to mock the environment variables
@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("SECRET_PATH", "/path/to/secret")
    monkeypatch.setenv("CRONITOR_API_KEY", "mock_api_key")
    
# Test load_google_sheet
def test_load_google_sheet(mock_google_sheet_data):
    with patch("data.SheetHelper") as mock_sheet_helper:
        mock_sheet_helper.return_value.gsheet_to_df.return_value = mock_google_sheet_data
        result = load_google_sheet()
        assert result.equals(mock_google_sheet_data)

# Test load_local_data
def test_load_local_data(mock_local_data, tmp_path):
    data_path = tmp_path / "data.csv"
    mock_local_data.to_csv(data_path, index=False)
    with patch("data.DATA_PATH", str(tmp_path)):
        result = load_local_data()
        assert result.equals(mock_local_data)

# Test word_count
def test_word_count(mock_google_sheet_data):
    result = word_count(mock_google_sheet_data)
    assert "free_text_len" in result.columns
    assert "do_better_len" in result.columns
    assert result["free_text_len"][0] == 5
    assert result["free_text_len"][1] == 6
    assert result["do_better_len"][0] == 3
    assert result["do_better_len"][1] == 2

