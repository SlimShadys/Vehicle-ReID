import logging
import time
import numpy as np

class Timer:
    def __init__(self):
        self.t = time.time()

    def start(self):
        self.t = time.time()

    def elapsed(self):
        return time.time() - self.t

    def fetch_restart(self):
        diff = self.elapsed()
        self.start()
        return diff

    def print_restart(self, callname=""):
        print("Call ({}) took {:.5f} seconds.".format(
            callname, time.time() - self.t))
        self.t = time.time()

class PerformanceTimer:
    def __init__(self):
        self.timer = Timer()
        self.calls = {}
        self.accumulated_calls = {}

    def restart_timer(self):
        self.timer.start()

    def register_call(self, callname):
        self.calls.setdefault(callname, []).append(self.timer.fetch_restart())

    def reset(self):
        self.timer.start()
        self.calls = {}
        self.accumulated_calls = {}

    def get_benchmark(self, unit="ms", precision=4, logger: logging.Logger = None):
        multip = 1000 if unit == "ms" else 1
        mxwidth = [0] * 4
        records = [["name", "min", "mean", "max"]]

        # Regular calls
        for k, times in self.calls.items():
            times = np.array(times) * multip
            record = [k]
            for x in [times.min(), times.mean(), times.max()]:
                record.append(str(round(x, precision)))
            records.append(record)
            for i, x in enumerate(record):
                mxwidth[i] = max(mxwidth[i], len(x))

        lines = []
        for record in records:
            line = f"{record[0].ljust(mxwidth[0])}  " + " ".join([x.center(w)
                                                                 for x, w in zip(record[1:], mxwidth[1:])])
            lines.append(line)
        sepline = "-" * (sum(mxwidth) + 4)

        # use logger to print out the results
        if logger is not None:
            logger.info("\n".join([lines[0], sepline] + lines[1:] + [sepline]))

        return records