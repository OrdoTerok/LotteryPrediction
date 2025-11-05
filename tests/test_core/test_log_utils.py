from core.log_utils import SilentLogger, suppress_console, setup_logging, get_logger
import logging

def test_silent_logger_methods():
    logger = SilentLogger()
    logger.info('test')
    logger.warning('test')
    logger.error('test')
    logger.debug('test')
    logger.critical('test')
    logger.log('test')
    logger.exception('test')
    logger.setLevel('INFO')
    logger.addHandler(None)
    logger.removeHandler(None)
    logger.handlers()
    logger.propagate()
    logger.getChild('child')
    assert True  # No exceptions

def test_suppress_console():
    suppress_console()
    assert True  # Should not raise

def test_setup_logging_and_get_logger(tmp_path):
    log_file = tmp_path / 'test.log'
    logger = setup_logging(str(log_file))
    logger.info('test')
    logger2 = get_logger()
    assert isinstance(logger2, logging.Logger)
