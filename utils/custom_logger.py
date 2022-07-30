import logging


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\x1b[92m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomLogger(logging.Logger):
    def __init__(self, name, level=logging.INFO):
        logging.Logger.__init__(self, name, level)

        # create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        console_handler.setFormatter(CustomFormatter())

        self.addHandler(console_handler)
        return
