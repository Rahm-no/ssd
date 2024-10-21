#!/bin/bash

# This script monitors CPU and memory usage including the execution time of the 'run_and_time.sh' command
echo "Time    CPU     GPU"

# Use ps to find the PID of the 'run_and_time.sh' script
pid=$(pgrep -f run_and_time.sh)

# Check if the PID is being correctly detected
if [ -z "$pid" ]; then
    echo "Error: run_and_time.sh is not running or its PID could not be detected."
    exit 1
else
    echo "Monitoring run_and_time.sh with PID: $pid"
fi

# Get the start time
start_time=$(date +%s)

# Monitor as long as run_and_time.sh is running
while kill -0 "$pid" 2> /dev/null; do
    current_time=$(date +%s)

    # Calculate the elapsed time
    elapsed_time=$((current_time - start_time))

    # Get CPU usage (for current system)
    cpuUsage=$(top -bn1 | awk '/Cpu/ {print $2}')


    # Get memory usage (in MB)
    memUsage=$(free -m | awk '/Mem/ {print $3}')

    # Get GPU utilization for all GPUs
    gpuUsages=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{ total += $1; count++ } END { if (count > 0) print total/count; else print 0 }')

    # Print the results
    echo "$elapsed_time s    $cpuUsage%    ${gpuUsages}%"

    # Sleep for 1 second
    sleep 1

    # Check if run_and_time.sh is still running
    pid=$(pgrep -f run_and_time.sh)
done

echo "run_and_time.sh has finished execution."
