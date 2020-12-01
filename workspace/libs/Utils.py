import sys, os

class Tee_Logger(object):
    """
    Duplicates sys.stdout to a log file
    To use it in a Jupyter notebook, in the cell you want to duplicate the output call:
    TeeLog = Tee_Logger(log_file_path)
    command_you_want_to_duplicate_stdoutput

    Then in the NEXT cell call:
    TeeLog.close()

    source: https://stackoverflow.com/q/616645
    """
    def __init__(self, filename="model.log", mode="a"):
        self.stdout = sys.stdout
        self.file = open(filename, mode)
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if self.file != None:
            self.file.close()
            self.file = None
