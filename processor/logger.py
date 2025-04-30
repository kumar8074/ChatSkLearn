# ===================================================================================
# Project: ChatSkLearn
# File: processor/logger.py
# Description: This script sets up a logging configuration for the project. It creates a directory for logs if it doesn't exist and configures the logging format and level. The log file is named with the current date and time.
#              The logging messages include the timestamp, line number, logger name, log level,
# and the actual log message.
# Author: LALAN KUMAR
# Created: [15-04-2025]
# Updated: [15-04-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import logging
import os
import sys
from datetime import datetime

# Add root path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)