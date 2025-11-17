# Copyright 2025 ADA Reseach Group and VERONA council. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging


def setup_logging(level=logging.INFO):
    """
    Set up logging for the ada_verona package:
    - All loggers under 'ada_verona' follow the given level
    - Messages have timestamps and logger names
    - Prevents double logging by disabling propagation
    """
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(format=log_format, level=logging.NOTSET)

    # Adjust all loggers starting with 'ada_verona'
    for name, logger in logging.Logger.manager.loggerDict.items():
        if name.startswith("ada_verona") and isinstance(logger, logging.Logger):
            logger.setLevel(level)
            logger.propagate = False
            # Ensure the logger has a StreamHandler
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(log_format))
                logger.addHandler(handler)

    # Also create a main package logger for convenience
    main_logger = logging.getLogger("ada_verona")
    main_logger.setLevel(level)
    main_logger.propagate = False
    if not main_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format))
        main_logger.addHandler(handler)

    return main_logger