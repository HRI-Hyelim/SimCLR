import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import re
from pathlib import Path
import matplotlib.colors as mcolors
from collections import defaultdict
import seaborn as sns
from itertools import combinations
from pyts.datasets import load_gunpoint
from pyts.image import RecurrencePlot
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
from convert_to_image import extract_name, convert_to_image

current_directory_path = Path.cwd()
directory = current_directory_path.parent /'activity_resistance_ML' / 'processed' 




def save_image_after_conversion(data_loc, conversion_method, flag_save_raw_data, size):
    directory_to_save = directory / 'label' / conversion_method

    for file in data_loc:
        df = pd.read_excel(file, sheet_name = 'sensitivity')
        df['time_diff'] = df['elapsed_time'].diff()
        large_diff_rows = df[df['time_diff'] > 10]

        time_cut = large_diff_rows['elapsed_time'].to_list()
        file_name = extract_name(file)
        print(file_name)
        convert_to_image(df, file_name, time_cut, conversion_method, size, directory_to_save, flag_save_raw_data)




################### INPUT ###################
file_names = ['20240917_5-6-8-9-12_CO_100ppm_25_processed.xlsx']
data_loc = [directory / i for i in file_names]
#print(data_loc)
save_image_after_conversion(data_loc, 'GAF_sum', True, 128)
save_image_after_conversion(data_loc, 'GAF_diff', False, 128)
