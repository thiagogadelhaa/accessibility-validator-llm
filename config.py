from pathlib import Path
import logging

import structlog

CSV_PATH = "./Original_full_data_new.csv"
LOCAL_SCREENSHOTS_DIR = Path("./element_screenshots/workspace/FullPipeline/element_screenshots/") 
LOG_FILE_PATH = Path("./experiment_results/execution_logs.json")

def setup_structured_logging():
    """
    Configura o structlog em conjunto com o logging nativo do Python 
    para salvar a telemetria em disco e no console simultaneamente.
    """
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8')
    
    console_handler = logging.StreamHandler()

    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,       # Injeta a chave "level": "info" / "error"
            structlog.stdlib.add_logger_name,     # Injeta o nome do módulo que gerou o log
            structlog.processors.TimeStamper(fmt="iso"), # Timestamp no padrão ISO 8601
            structlog.processors.JSONRenderer()   # Converte o dicionário final em string JSON
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

setup_structured_logging()

logger = structlog.get_logger()