#!/bin/bash

# Table Header
printf "%-15s %-15s %-10s\n" "benchmark" "config" "ipc"
printf "%-15s %-15s %-10s\n" "---------" "------" "---"

# Loop through the files
for file in *_run.txt; do
    # 1. Parse filename (e.g., LIB_SM6_run.txt)
    # benchmark = 1st field, config = 2nd field (delimited by _)
    benchmark=$(echo "$file" | cut -d'_' -f1)
    config=$(echo "$file" | cut -d'_' -f2)

    # 2. Extract IPC value
    # Matches "gpu_tot_ipc", splits by "=", and trims whitespace
    ipc=$(grep "gpu_tot_ipc" "$file" | awk -F'=' '{print $2}' | xargs)

    # 3. Print the row if ipc exists
    if [ -n "$ipc" ]; then
        printf "%-15s %-15s %-10s\n" "$benchmark" "$config" "$ipc"
    fi
done

