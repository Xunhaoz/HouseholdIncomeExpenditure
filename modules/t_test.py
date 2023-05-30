import os

import pandas as pd
import numpy as np


def delete_outliers_and_nas(data_list: list) -> list:
    res = []
    for data in data_list:
        data = data.dropna(ignore_index=True)
        mean = data.mean()
        std = data.std()
        data = data[(mean - 1.5 * std) <= data]
        data = data[data <= (mean + 1.5 * std)]
        res.append(data.reset_index(drop=True))
    return res


def t_score(data: np.ndarray) -> float:
    res = (np.mean(data) - 0) / ((data.std(ddof=1)) / np.sqrt(len(data)))
    return res


def hypothesis(data01: list, data02: list, score: tuple or float) -> bool:
    means = [data01[i] - data02[i] for i in range(10)]
    wage_diff = np.array(means)
    t = t_score(wage_diff)
    print(abs(t_score(wage_diff)))

    if type(score) == tuple:
        return bool(t > score or t < score)
    else:
        return bool(abs(t) > abs(score))


def t_test_example() -> None:
    selections = [['UBAN', 'COUNTRY'], ['COLLEGE', "NOSTUDY"], ["MALE", "FEMALE"]]

    path = '../static/excel/'
    for root, dirs, files in os.walk(path):
        for file in files:
            selection_filters = []
            if file.startswith("STUDY"):
                selection_filters = selections[1]
            elif file.startswith("SEX"):
                selection_filters = selections[2]
            elif file.startswith("NEWCITY"):
                selection_filters = selections[0]
            else:
                continue

            print(file, ": ", end="")
            excel_path = path + file
            excel_df = pd.read_excel(excel_path)

            category01 = []
            category02 = []

            for year in range(2011, 2022):
                category01.append(excel_df[excel_df['YEAR'] == year][selection_filters[0]])
                category02.append(excel_df[excel_df['YEAR'] == year][selection_filters[1]])

            category01 = delete_outliers_and_nas(category01)
            category02 = delete_outliers_and_nas(category02)

            category01_mean = [i.mean() for i in category01]
            category02_mean = [i.mean() for i in category02]

            hypothesis(category01_mean, category02_mean, 1.8331)

    print("FINISH")


if __name__ == '__main__':
    t_test_example()
