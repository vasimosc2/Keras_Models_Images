#!/bin/bash

# Define log file name
LOG_FILE="log.txt"

# Remove previous log file if exists
rm -f $LOG_FILE

# Run the Python script and log output
python3 your_script.py | tee $LOG_FILE
