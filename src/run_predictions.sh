 #!/bin/bash

# Navigate to the project directory
cd /home/ron/forecasting_models/

# Set the environment variable
export SNOWPACK_DATA_ROOT="/ssd/snowpack/fcst"

# Activate your virtual environment
source /home/ron/forecasting_models/.venv/bin/activate

# 4. Run the script using 'python3' WITHOUT the full path
#    The shell will now find the python3 from your activated .venv
python3 src/run_prediction.py --plot --date $(date -d 'tomorrow' +\%Y-\%m-\%d) >> logs/avalanche_prediction.log 2>&1
#python3 src/run_prediction.py --date "2025-01-14" >> logs/avalanche_prediction.log 2>&1