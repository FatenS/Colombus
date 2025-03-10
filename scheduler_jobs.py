# scheduler_jobs.py
import os
import re
import requests

# In-memory variable to track the last processed timestamp.
last_processed_timestamp = None

def check_for_new_files():
    global last_processed_timestamp
    data_dir = "/app/data"  # This should match the mount point in your container.
    
    try:
        all_files = os.listdir(data_dir)
    except Exception as e:
        print(f"Error accessing directory: {e}")
        return

    # Regex to match filenames like: midmarket_rates_YYYY-MM-DD_HHMM.json
    mid_pattern = re.compile(r"midmarket_rates_(\d{4}-\d{2}-\d{2}_\d{4})\.json")
    timestamps = []
    for filename in all_files:
        match = mid_pattern.search(filename)
        if match:
            timestamps.append(match.group(1))
    if not timestamps:
        print("No matching midmarket files found.")
        return

    latest_timestamp = max(timestamps)
    if last_processed_timestamp is None or latest_timestamp != last_processed_timestamp:
        print(f"New file detected with timestamp {latest_timestamp} (last processed: {last_processed_timestamp})")
        last_processed_timestamp = latest_timestamp
        try:
            # Trigger the upsert endpoint.
            response = requests.post("http://localhost:5001/admin/api/upsert-exchange-data")
            print("Triggered upsert API, response:", response.json())
        except Exception as api_err:
            print("Error triggering upsert API:", api_err)
    else:
        print("No new files detected.")
