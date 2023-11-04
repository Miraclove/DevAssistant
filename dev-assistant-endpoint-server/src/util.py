import logging
import argparse

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('app')


import argparse

def get_parser() -> argparse.ArgumentParser:
    """
    Returns an ArgumentParser object with the following arguments:
    --port: The port number to run the server on. Default is 8000.
    --host: The host address to bind the server to. Default is '0.0.0.0'.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    return parser


    