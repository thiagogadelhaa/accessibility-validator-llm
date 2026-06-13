from pathlib import Path

import structlog

CSV_PATH = "./Original_full_data_new.csv"
LOCAL_SCREENSHOTS_DIR = Path("./element_screenshots/workspace/FullPipeline/element_screenshots/") 

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()