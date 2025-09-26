import logging

root_logname = "[ada_verona::{}]"

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.NOTSET)

experiment_logger = logging.getLogger(root_logname.format("experiment"))
experiment_logger.setLevel(logging.INFO)
