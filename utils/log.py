""" Logging configuration for the project. """

import logging

logging.getLogger("torch").setLevel(logging.WARNING)

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
