import psutil
import time
import matplotlib.pyplot as plt
from datetime import datetime

def track_memory_usage(interval, duration, log_file):
    timestamps = []
    memory_usages = []

    end_time = time.time() + duration

    while time.time() < end_time:
        # Get the current memory usage in megabytes
        memory_mb = psutil.virtual_memory().used / (1024 ** 2)

        # Get the current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Log the data to the file
        with open(log_file, 'a') as file:
            file.write(f"{timestamp},{memory_mb}\n")

        # Append data for plotting
        timestamps.append(timestamp)
        memory_usages.append(memory_mb)

        # Wait for the specified interval
        time.sleep(interval)

    # Plot the data
    plt.plot(timestamps, memory_usages, marker='o')
    plt.xlabel('Timestamp')
    plt.ylabel('Memory Usage (MB)')
    plt.title('CPU Memory Usage Over Time')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('memory_usage_plot.png')

if __name__ == "__main__":
    # Set the monitoring interval (in seconds) and total monitoring duration (in seconds)
    monitoring_interval = 0.1
    monitoring_duration = 40

    # Set the log file name
    log_file_name = 'memory_usage_log.csv'

    # Start tracking memory usage
    track_memory_usage(monitoring_interval, monitoring_duration, log_file_name)
