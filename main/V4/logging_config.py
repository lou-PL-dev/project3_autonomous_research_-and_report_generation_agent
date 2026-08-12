"""Shared logging setup, called once from run_pipeline() so both the CLI
(recap.py) and the Flask API (app.py) get consistent, configured output.
"""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent: only the first call installs a handler, safe to call from
    every pipeline run without duplicating log lines.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
