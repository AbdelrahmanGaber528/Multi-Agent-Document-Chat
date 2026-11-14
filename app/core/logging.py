import logging
import logging.config
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"


LOG_DIR.mkdir(parents=True, exist_ok=True)


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# Current graph round, shown in every log line ("-" when not inside a round)
_current_round: ContextVar[str] = ContextVar("round", default="-")


class RoundFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.round = _current_round.get()
        return True


@contextmanager
def log_round(round_number: int):
    """Every log line inside the block shows this round number.

    with log_round(2):
        logger.info("semantic hits=%d", n)
    """
    token = _current_round.set(str(round_number))
    try:
        yield
    finally:
        _current_round.reset(token)




LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,

    "filters": {
        "round": {"()": RoundFilter},
    },

    "formatters": {
        "standard": {
            "format": (
                "%(asctime)s | %(levelname)-8s | round=%(round)s | "
                "%(name)s | %(filename)s:%(lineno)d | %(message)s"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },

    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "filters": ["round"],
            "stream": "ext://sys.stdout",
        },

        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filters": ["round"],
            "filename": str(LOG_DIR / "app.log"),
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },

    "loggers": {
        "app": {"level": "DEBUG", "propagate": True},
    },

    "root": {
        "level": "WARNING",
        "handlers": ["console", "file"],
    },
}



def setup_logging() -> None:
    logging.config.dictConfig(LOGGING_CONFIG)