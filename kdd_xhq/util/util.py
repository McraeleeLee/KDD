import logging
import os
import time
import pathlib
import pickle


def create_logger(prefix, root=None):
    """
    This function is complete and could be ignored.
    Create logger for logging
    :param config: object that contains configuration settings
    :return: the logger for logging
    """
    # root_output_dir = pathlib.Path(config.TRAIN.LOG_FOLDER)
    if root is None:
        root_output_dir = pathlib.Path("logs")
    else:
        root_output_dir = pathlib.Path(root)
    # set up logger
    if not root_output_dir.exists():
        print('=> [LOGGER] creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(prefix, time_str)
    final_log_file = root_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logging.getLogger('').handlers:
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)
    return logger


def pickle_read(file_path):
    # try:
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    # except:
    #     print('Pickle read error: not exists {}'.format(file_path))
    #     return None


def pickle_write(file_path, what_to_write):
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except:
            pass
    # try:
    with open(file_path, 'wb+') as f:
        pickle.dump(what_to_write, f)
        print('Pickle write to: {}'.format(file_path))
    # except:
    #     print('Pickle write error: {}'.format(file_path))
