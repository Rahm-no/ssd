#!/bin/bash

# This script monitors overall system CPU and GPU usage over time.
echo "Time  CPU   GPU "

# Get the start time
start_time=$(date +%s)

# Continuous monitoring loop
while true; do
    current_time=$(date +%s)

    # Calculate elapsed time in seconds
    elapsed_time=$((current_time - start_time))

    # Get total CPU usage using mpstat
    cpuUsage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}') 

    # Get GPU utilization for all running GPUs and calculate the average
    gpuUsages=$(nvidia-smi | tail -n +10 | awk '{print $13}' | sed 's/%//g')
    if [ -z "$gpuUsages" ]; then
        gpuUsage="0%"
    else
        total=0
        count=0
        for usage in $gpuUsages; do
            total=$((total + usage))
            count=$((count + 1))
        done
        average=$((total / count))
        gpuUsage="${average}%"
    fi

    # Output the time, CPU, and GPU usage
    echo "$elapsed_time s    $cpuUsage%    $gpuUsage"

    # Wait 1 second before the next update
    sleep 1
done
