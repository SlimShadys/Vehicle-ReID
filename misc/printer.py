import logging

class Logger():
    def __init__(self, level=None):
        self.depth = 0
        self.INDENT = "  "
        self.logger = logging.getLogger("MTMC")
        self.num_errors = 0
        self.level = level if level is not None else logging.DEBUG

        log_level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }

        if isinstance(self.level, str):
            self.level = log_level_map[self.level.lower()]

        self.logger.setLevel(self.level)
        self.logger.propagate = False

        # Add handler if there are no handlers attached to avoid duplicate logs
        if not self.logger.handlers:
            # Create a StreamHandler (for console output)
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)

            # Set a formatter and attach it to the handler
            formatter = logging.Formatter('%(levelname)-8s [%(asctime)s]: %(message)s')
            console_handler.setFormatter(formatter)

            # Attach the handler to the logger
            self.logger.addHandler(console_handler)

    def inc_depth(self):
        self.depth += 1

    def dec_depth(self):
        self.depth = max(0, self.depth - 1)

    def log_function(self, log_func, msg, *args):
        if len(args) > 0:
            msg = self.INDENT * self.depth + (msg % args)
        else:
            msg = self.INDENT * self.depth + msg
        log_func(msg)

    def info(self, msg, *args):
        self.log_function(self.logger.info, msg, *args)

    def debug(self, msg, *args):
        self.log_function(self.logger.debug, msg, *args)

    def warning(self, msg, *args):
        self.log_function(self.logger.warning, msg, *args)

    def error(self, msg, *args):
        self.num_errors += 1
        self.log_function(self.logger.error, msg, *args)
