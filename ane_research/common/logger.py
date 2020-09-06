'''Define logging utilities'''

import logging
import os
import sys

from pythonjsonlogger import jsonlogger


def initialize_logger(name: str, in_notebook: bool = False) -> logging.Logger:
  '''Initialize a logger with the given name

  Args:
    name (str): Then name of the logger
    in_notebook (bool, default: False): If not running inside a notebook, log as JSON for easy file searching

  Returns:
    logging.Logger: Initialized logger
  '''
  logger = logging.getLogger(name)
  log_handler = logging.StreamHandler(sys.stdout)

  if not in_notebook:
    formatter = jsonlogger.JsonFormatter(
      '''
        (levelname),
        (message),
        (module),
        (funcName),
        (asctime),
        (process)
      ''')
    log_handler.setFormatter(formatter)

  logger.handlers[:] = []
  logger.addHandler(log_handler)
  if os.environ.get('DEBUG'):
    logger.setLevel(logging.DEBUG)
  else:
    logger.setLevel(logging.INFO)
  return logger
