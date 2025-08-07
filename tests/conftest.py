import pytest
import sys

@pytest.fixture(scope="session", autouse=True)
def global_setup():
    import logging
    logging.basicConfig(level=logging.DEBUG)
    import loguru
    loguru.logger.remove()
    loguru.logger.add(sys.stderr, level="DEBUG")
