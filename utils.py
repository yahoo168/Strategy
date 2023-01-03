import os
import numpy as np
def makeFolder(folder_name):
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    else:
        return None

#將調倉頻率由文字轉換至數字
def freq_to_days(freq):
    if freq == "day":
        period = 1

    elif freq == "week":
        period = 5

    elif freq == "month":
        period = 22

    elif freq == "quarter":
        period = 60

    elif freq == "year":
        period = 252

    return period

#將給定的權重矩陣總倉位調整至total_weight，預設為1
def adjust_weight(weight_df, total_weight=1):
    weight_df.replace([np.inf, -np.inf], 0, inplace=True)
    adjust_factor = weight_df.sum(axis=1)
    weight_df = weight_df.div(adjust_factor, axis=0)
    weight_df = weight_df * total_weight
    weight_df.fillna(0, inplace=True)
    return weight_df

