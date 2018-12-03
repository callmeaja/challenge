import tensorflow as tf
import numpy as np
import os
import time
import datetime
from importlib.machinery import SourceFileLoader
os.chdir('Z:\Desktop\Competitions\AI Challenge')

cnn = SourceFileLoader('textCNN', 'Codes\2.0 Setting up the Pipeline.py').load_module()

filter_size = 3
