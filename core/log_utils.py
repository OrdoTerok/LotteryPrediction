"""
Logging utilities for LotteryPrediction.
"""
import logging
import sys
import io
import json

class SilentLogger:
    """
    Logger that suppresses all output. Use for KerasTuner or any noisy library.
    """
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass
    def critical(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass
    def exception(self, *args, **kwargs): pass
    def setLevel(self, *args, **kwargs): pass
    def addHandler(self, *args, **kwargs): pass
    def removeHandler(self, *args, **kwargs): pass
    def handlers(self): return []
    def propagate(self, *args, **kwargs): pass
    def getChild(self, *args, **kwargs): return self
    def __getattr__(self, name):
        def no_op(*args, **kwargs): pass
        return no_op

class _DevNull(io.TextIOBase):
    def write(self, *args, **kwargs): pass
    def flush(self): pass

def suppress_console():
    """
    Redirect sys.stdout and sys.stderr to suppress all console output.
    Call at the start of your script or before noisy operations.
    """
    sys.stdout = _DevNull()
    sys.stderr = _DevNull()

class SuppressOutput:
    """
    Context manager to temporarily suppress stdout and stderr.
    Usage:
        with SuppressOutput():
            # Any code here won't print to console
            noisy_function()
    """
    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._original_stdout = None
        self._original_stderr = None
        self._devnull = _DevNull()
    
    def __enter__(self):
        if self.suppress_stdout:
            self._original_stdout = sys.stdout
            sys.stdout = self._devnull
        if self.suppress_stderr:
            self._original_stderr = sys.stderr
            sys.stderr = self._devnull
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
        return False

def setup_logging(log_filename=None, log_to_console=False):
    """
    Set up logging to file and optionally console with immediate flushing.
    Args:
        log_filename (str, optional): If provided, log to this file.
        log_to_console (bool, optional): If True, also log to console. Default False.
    """
    # Disable the lastResort handler that outputs to stderr
    logging.lastResort = None
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Include logger name to see which module is logging
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Remove ALL handlers from root logger and all child loggers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # Also clear handlers from all existing loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        child_logger = logging.getLogger(name)
        for handler in child_logger.handlers[:]:
            handler.close()
            child_logger.removeHandler(handler)
        child_logger.propagate = True  # Ensure they propagate to root
    
    # Custom handler class that flushes immediately after every log
    class ImmediateFlushHandler(logging.Handler):
        def __init__(self, stream):
            super().__init__()
            self.stream = stream
            
        def emit(self, record):
            try:
                msg = self.format(record)
                self.stream.write(msg + '\n')
                self.stream.flush()  # Immediate flush
            except Exception:
                self.handleError(record)
    
    # Add file handler with immediate flush
    if log_filename:
        log_file = open(log_filename, 'a', encoding='utf-8')
        fh = ImmediateFlushHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
    
    # Add console handler with immediate flush if requested
    if log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        # Ensure stdout is line-buffered
        sys.stdout.reconfigure(line_buffering=True)
    
    # Suppress noisy third-party library loggers
    third_party_loggers = [
        'tensorflow',
        'keras',
        'keras_tuner',
        'pyswarms',
        'matplotlib',
        'PIL',
        'h5py',
        'absl',
        'numba',
        'lightgbm',
        'sklearn'
    ]
    for lib_name in third_party_loggers:
        lib_logger = logging.getLogger(lib_name)
        lib_logger.setLevel(logging.ERROR)
        lib_logger.propagate = False
        # Remove all handlers from third-party loggers
        for handler in lib_logger.handlers[:]:
            handler.close()
            lib_logger.removeHandler(handler)
    
    return logger

def get_logger():
    return logging.getLogger()

def save_json(data, filename):
    """Save a dictionary or list to a JSON file."""
    import numpy as np
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert(v) for v in obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    with open(filename, 'w') as f:
        json.dump(convert(data), f, indent=2)
