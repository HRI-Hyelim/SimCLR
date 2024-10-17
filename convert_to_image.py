import numpy as np
import pandas as pd
import os
import glob
import datetime
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

current_directory_path = Path.cwd()
directory = current_directory_path.parent /'activity_resistance_ML' / 'processed' 
data_loc = sorted(directory.glob('*_processed.xlsx')) 


def convert_to_RGB(array, name, conversion_method, size):
    
    # array normalization to be the range [0,1]
    arr_min = array.min()
    arr_max = array.max()
    if arr_min == arr_max:
        image = array
    else:
        image = (array - arr_min) / (arr_max - arr_min)
    plt.figure()
    if conversion_method == 'recurrence':
        cmap = plt.get_cmap('binary')
    else:
        cmap = plt.get_cmap('rainbow')
    image_colormap = cmap(image)
    image_rgb = (image_colormap[:,:,:3]*255).astype(np.uint8)
    image_pil = Image.fromarray(image_rgb)

    resized = image_pil.resize((size,size))
    resized.save(name)

def extract_name(file):
    return file.name.split('.')[0].split('_')

def save_raw_data(df, x_data, y_data, file_name):
    temp_name = file_name.name
    file_name_to_save = file_name.parent.parent / 'raw_data' / temp_name
    plt.figure()
    sns.scatterplot(data = df, x = x_data, y = y_data)
    plt.savefig(file_name_to_save)


def convert_to_image(df, file_name, time_cut, conversion_method, size, directory_to_save, flag_save_raw_data):
    """
    
    """
    if conversion_method == 'GAF_sum':
        transform = GramianAngularField(method = 'summation')
    elif conversion_method == 'GAF_diff':
        transform = GramianAngularField(method = 'difference')
    elif conversion_method == 'recurrence':
        transform = RecurrencePlot()
    else:
        transform = MarkovTransitionField()

    def apply_transform(column, col_name, cycle_num):
        column_data = column.values.reshape(1,-1)
        after_transform = transform.fit_transform(column_data)[0]
        file_name[1] = col_name
        temp_name = '_'.join(file_name[:-1]) + cycle_num + '.png'
        save_file = directory_to_save / temp_name 
        #print(save_file)
        convert_to_RGB(after_transform, save_file, conversion_method, size)
        return save_file

    
    n = len(time_cut)
    if n > 50 or n ==0:
        cycle_name = '_c'
        for col in df.columns:
            if 'ML' in col:
                file_name_to_save = apply_transform(df[col], col, cycle_name)
                if flag_save_raw_data ==True:
                    save_raw_data(df, 'elapsed_time', col, file_name_to_save)
    
    else:
        # Case 1: Time values less than the first value in time_cut
        subset1 = df[df['elapsed_time'] < time_cut[0]]
        cycle_name = '_c0'
        for col in subset1:
            if 'ML' in col:
                file_name_to_save = apply_transform(subset1[col], col, cycle_name)
                if flag_save_raw_data ==True:
                    save_raw_data(subset1, 'elapsed_time', col, file_name_to_save)

        # Case 2: Time values between each pair in time_cut
        for i in range(n-1):
            temp1 = df[(df['elapsed_time'] >= time_cut[i]) & (df['elapsed_time'] < time_cut[i+1])]
            cycle_name = '_c' + str(i+1)
            for col in temp1:
                if 'ML' in col:
                    file_name_to_save = apply_transform(temp1[col], col,cycle_name)
                    if flag_save_raw_data ==True:
                        save_raw_data(temp1, 'elapsed_time', col, file_name_to_save)

        # Case 3: Time values greater than the last value in time_cut
        subset2 = df[df['elapsed_time'] > time_cut[-1]]
        cycle_name = '_c' + str(n)
        for col in subset2:
            if 'ML' in col:
                file_name_to_save = apply_transform(subset2[col], col, cycle_name)
                if flag_save_raw_data ==True:
                    save_raw_data(subset2, 'elapsed_time', col, file_name_to_save)


def save_image_after_conversion(data_loc, conversion_method, flag_save_raw_data, size):
    directory_to_save = current_directory_path / conversion_method

    for file in data_loc:
        df = pd.read_excel(file, sheet_name = 'sensitivity')
        df['time_diff'] = df['elapsed_time'].diff()
        large_diff_rows = df[df['time_diff'] > 10]
        time_cut = large_diff_rows['elapsed_time'].to_list()
        file_name = extract_name(file)
        convert_to_image(df, file_name, time_cut, conversion_method, size, directory_to_save, flag_save_raw_data)

save_image_after_conversion(data_loc, 'MTF', False, 128)
save_image_after_conversion(data_loc, 'GAF_sum', False, 128)
save_image_after_conversion(data_loc, 'GAF_diff', False, 128)
save_image_after_conversion(data_loc, 'recurrence', False, 128)







