import os
import datetime

class Logger:
    def __init__(self, exp_name):
        # Ensure the 'logs' directory exists
        os.makedirs('./logs', exist_ok=True)
        # Open the log file for writing
        self.file = open('./logs/{}.log'.format(exp_name), 'w')

    def log(self, content):
        # Add a timestamp to each log entry
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {content}"
        self.file.write(log_entry + '\n')
        self.file.flush()
