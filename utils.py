import os

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

