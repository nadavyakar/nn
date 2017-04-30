from os import path
from platform import system
from linecache import getline, checkcache
from sys import exc_info
import logging
from configparser import ConfigParser

# Setting the parameters of all program
INFO_MODE = None
SPACE = None
WHITE_FILENAME = None
RED_FILENAME = None
LOG_PATH = None
DATASET_FOLDER = None
LOG_FOLDER = None
EPOCH = None
LOW_RANGE = None
HIGH_RANGE = None
DELIMITER = None
LABEL_IDX = None
INPUT_IMAGE_FOLDER = None
OUTPUT_FOLDER = None
SOURCE_FOLDER = None
THRESHOLD = None
SHIFT = None
DEBUG_MODE = None
IMAGE_SIZE = None
ADD_NOISE = None
OUTPUT_IMAGE_FOLDER = None
LEARNING_RATE = None
ALPHA = None

def config_parser():
    global INFO_MODE, SPACE, WHITE_FILENAME, RED_FILENAME, LOG_PATH, DATASET_FOLDER, LOG_FOLDER, EPOCH, LOW_RANGE, \
        HIGH_RANGE, DELIMITER, LABEL_IDX, INPUT_IMAGE_FOLDER, OUTPUT_FOLDER, SOURCE_FOLDER, THRESHOLD, SHIFT, \
        DEBUG_MODE, IMAGE_SIZE, ADD_NOISE, OUTPUT_IMAGE_FOLDER, LEARNING_RATE, ALPHA

    config = ConfigParser()
    config.read('parameters config.ini')
    # Properties of dataset
    WHITE_FILENAME = config.get("dataset properties", "White filename")
    RED_FILENAME = config.get("dataset properties", "Red filename")
    DATASET_FOLDER = config.get("dataset properties", "Dataset folder")

    # Properties of scale:
    # Epoch, log and high range for random function
    EPOCH = int(config.get("NN properties", "Epoch"))
    LOW_RANGE = float(config.get("NN properties", "Low range"))
    HIGH_RANGE = float(config.get("NN properties", "High range"))
    LEARNING_RATE = float(config.get("NN properties", "Learning rate"))
    ALPHA = float(config.get("NN properties", "Alpha"))


    # Properties of csv properties:
    # Delimiter - The character separated the information into columns
    # Label class - The index of the column where the label appears
    DELIMITER = config.get("csv properties", "delimiter")
    LABEL_IDX = int(config.get("csv properties", "label class"))

    # Properties of debugger properties:
    # Info mode - indicates that output is printed  on the screen
    # LOG_PATH - path of log file
    # Log folder - The name of the folder where the log file is located
    INFO_MODE = bool(config.get("debugger properties", "Info mode"))
    LOG_PATH = config.get("debugger properties", "Log filename")
    LOG_FOLDER = config.get("debugger properties", "Log folder")

    # Properties of print:
    # The character that breaks the line (\n - Windows and \r - Linux)
    SPACE = config.get("print properties", "Space")

    INPUT_IMAGE_FOLDER = config.get("image properties", "Images folder")
    OUTPUT_IMAGE_FOLDER = config.get("image properties", "Output image foldr")
    OUTPUT_FOLDER = config.get("image properties", "Output folder")
    SOURCE_FOLDER = config.get("image properties", "Source folder")
    THRESHOLD = int(config.get("image properties", "Threshold"))
    SHIFT = int(config.get("image properties", "Shift"))
    DEBUG_MODE = bool(config.get("image properties", "Debug mode"))
    IMAGE_SIZE = int(config.get("image properties", "Image size"))
    ADD_NOISE = bool(config.get("image properties", "Add noise"))

    LOG_PATH = path.join(LOG_FOLDER, LOG_PATH)


def space_char_by_platform():
    global SPACE
    if system() == "Linux":
        SPACE = '\r'


def print_exception():
    exc_type, exc_obj, tb = exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    checkcache(filename)
    line = getline(filename, lineno, f.f_globals)
    logging.debug(SPACE + "EXCEPTION IN ({}, LINE {} '{}'): {}'".format(filename, lineno, line, exc_obj))
    logging.info(SPACE + "EXCEPTION IN ({}, LINE {} '{}'): {}'".format(filename, lineno, line, exc_obj))


config_parser()
space_char_by_platform()






