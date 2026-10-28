import os
import subprocess
import sys

import config
from conftest import ROOT_DIR


def test_training_years_include_2025_season():
    assert 2025 in config.PROCESSING_CONFIG["target_years"]
    assert config.PROCESSING_CONFIG["target_years"] == sorted(config.PROCESSING_CONFIG["target_years"])


def test_snowpack_data_root_env_var_sets_snowpack_output(tmp_path):
    env = {**os.environ, "SNOWPACK_DATA_ROOT": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-c", "import config; print(config.IS_LOCAL_DATA_SOURCE, config.PATHS['EXTERNAL_DATA']['snowpack_output'])"],
        cwd=ROOT_DIR / "src", env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout.split() == ["True", str(tmp_path)]
