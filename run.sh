#!/bin/bash
# This script runs the app and restapi in parallel
# It uses the `wait` command to keep the script running until both processes are finished

python src/app.py &
python src/rest_api.py &

wait