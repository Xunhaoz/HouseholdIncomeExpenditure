import pandas as pd
import torch
import torch.nn as nn
import globals


def change_immutable_multi_dict_to_dict(data, key=None, value=None):
    data = data.to_dict()
    for k, v in data.items():
        data[k] = int(v)
    if key and value:
        data[key] = value
    return data


def preprocess(df):
    df['REL'] = df['REL'].apply(lambda x: 1 if x == 1 else 0)
    df['MALE'] = df['SEX'].apply(lambda x: 1 if x == 1 else 0)
    df['FEMALE'] = df['SEX'].apply(lambda x: 1 if x == 2 else 0)
    df['AGE'] = df['AGE'] / 100
    df['EDU01'] = df['EDU'].apply(lambda x: 1 if x == 1 else 0)
    df['EDU02'] = df['EDU'].apply(lambda x: 1 if x == 2 else 0)
    df['EDU03'] = df['EDU'].apply(lambda x: 1 if x == 3 else 0)
    df['EDU04'] = df['EDU'].apply(lambda x: 1 if x == 4 else 0)
    df['EDU05'] = df['EDU'].apply(lambda x: 1 if x == 5 else 0)
    df['EDU06'] = df['EDU'].apply(lambda x: 1 if x == 6 else 0)
    df['EDU07'] = df['EDU'].apply(lambda x: 1 if x == 7 else 0)
    df['EDU08'] = df['EDU'].apply(lambda x: 1 if x == 8 else 0)
    df['EDU09'] = df['EDU'].apply(lambda x: 1 if x == 9 else 0)
    df['EDU10'] = df['EDU'].apply(lambda x: 1 if x == 10 else 0)
    df['IND00'] = df['IND'].apply(lambda x: 1 if x == 0 else 0)
    df['IND01'] = df['IND'].apply(lambda x: 1 if x == 1 else 0)
    df['IND02'] = df['IND'].apply(lambda x: 1 if x == 2 else 0)
    df['IND03'] = df['IND'].apply(lambda x: 1 if x == 3 else 0)
    df['IND05'] = df['IND'].apply(lambda x: 1 if x == 5 else 0)
    df['IND08'] = df['IND'].apply(lambda x: 1 if x == 8 else 0)
    df['IND35'] = df['IND'].apply(lambda x: 1 if x == 35 else 0)
    df['IND36'] = df['IND'].apply(lambda x: 1 if x == 36 else 0)
    df['IND41'] = df['IND'].apply(lambda x: 1 if x == 41 else 0)
    df['IND45'] = df['IND'].apply(lambda x: 1 if x == 45 else 0)
    df['IND55'] = df['IND'].apply(lambda x: 1 if x == 55 else 0)
    df['IND49'] = df['IND'].apply(lambda x: 1 if x == 49 else 0)
    df['IND58'] = df['IND'].apply(lambda x: 1 if x == 58 else 0)
    df['IND64'] = df['IND'].apply(lambda x: 1 if x == 64 else 0)
    df['IND67'] = df['IND'].apply(lambda x: 1 if x == 67 else 0)
    df['IND69'] = df['IND'].apply(lambda x: 1 if x == 69 else 0)
    df['IND77'] = df['IND'].apply(lambda x: 1 if x == 77 else 0)
    df['IND85'] = df['IND'].apply(lambda x: 1 if x == 85 else 0)
    df['IND86'] = df['IND'].apply(lambda x: 1 if x == 86 else 0)
    df['IND90'] = df['IND'].apply(lambda x: 1 if x == 90 else 0)
    df['IND94'] = df['IND'].apply(lambda x: 1 if x == 94 else 0)
    df['IND83'] = df['IND'].apply(lambda x: 1 if x == 83 else 0)
    df['OCC00'] = df['OCC'].apply(lambda x: 1 if x == 0 else 0)
    df['OCC01'] = df['OCC'].apply(lambda x: 1 if x == 1 else 0)
    df['OCC02'] = df['OCC'].apply(lambda x: 1 if x == 2 else 0)
    df['OCC03'] = df['OCC'].apply(lambda x: 1 if x == 3 else 0)
    df['OCC04'] = df['OCC'].apply(lambda x: 1 if x == 4 else 0)
    df['OCC05'] = df['OCC'].apply(lambda x: 1 if x == 5 else 0)
    df['OCC61'] = df['OCC'].apply(lambda x: 1 if x == 61 else 0)
    df['OCC62'] = df['OCC'].apply(lambda x: 1 if x == 62 else 0)
    df['OCC63'] = df['OCC'].apply(lambda x: 1 if x == 63 else 0)
    df['OCC07'] = df['OCC'].apply(lambda x: 1 if x == 7 else 0)
    df['OCC08'] = df['OCC'].apply(lambda x: 1 if x == 8 else 0)
    df['OCC09'] = df['OCC'].apply(lambda x: 1 if x == 9 else 0)
    df['OCC10'] = df['OCC'].apply(lambda x: 1 if x == 10 else 0)
    df['WKCLASS01'] = df['WKCLASS'].apply(lambda x: 1 if x == 1 else 0)
    df['WKCLASS02'] = df['WKCLASS'].apply(lambda x: 1 if x == 2 else 0)
    df['WKCLASS03'] = df['WKCLASS'].apply(lambda x: 1 if x == 3 else 0)
    df['WKCLASS04'] = df['WKCLASS'].apply(lambda x: 1 if x == 4 else 0)
    df['WKCLASS05'] = df['WKCLASS'].apply(lambda x: 1 if x == 5 else 0)
    df['WKCLASS06'] = df['WKCLASS'].apply(lambda x: 1 if x == 6 else 0)
    df['WKCLASS07'] = df['WKCLASS'].apply(lambda x: 1 if x == 7 else 0)
    df['WKCLASS08'] = df['WKCLASS'].apply(lambda x: 1 if x == 8 else 0)
    df['WKCLASS09'] = df['WKCLASS'].apply(lambda x: 1 if x == 9 else 0)
    df['WORKPLACE02'] = df['WORKPLACE'].apply(lambda x: 1 if x == 2 else 0)
    df['WORKPLACE03'] = df['WORKPLACE'].apply(lambda x: 1 if x == 3 else 0)
    df['WORKPLACE04'] = df['WORKPLACE'].apply(lambda x: 1 if x == 4 else 0)
    df['WORKPLACE05'] = df['WORKPLACE'].apply(lambda x: 1 if x == 5 else 0)
    df['WORKPLACE07'] = df['WORKPLACE'].apply(lambda x: 1 if x == 7 else 0)
    df['WORKPLACE08'] = df['WORKPLACE'].apply(lambda x: 1 if x == 8 else 0)
    df['WORKPLACE09'] = df['WORKPLACE'].apply(lambda x: 1 if x == 9 else 0)
    df['WORKPLACE10'] = df['WORKPLACE'].apply(lambda x: 1 if x == 10 else 0)
    df['WORKPLACE13'] = df['WORKPLACE'].apply(lambda x: 1 if x == 13 else 0)
    df['WORKPLACE14'] = df['WORKPLACE'].apply(lambda x: 1 if x == 14 else 0)
    df['WORKPLACE15'] = df['WORKPLACE'].apply(lambda x: 1 if x == 15 else 0)
    df['WORKPLACE16'] = df['WORKPLACE'].apply(lambda x: 1 if x == 16 else 0)
    df['WORKPLACE17'] = df['WORKPLACE'].apply(lambda x: 1 if x == 17 else 0)
    df['WORKPLACE18'] = df['WORKPLACE'].apply(lambda x: 1 if x == 18 else 0)
    df['WORKPLACE20'] = df['WORKPLACE'].apply(lambda x: 1 if x == 20 else 0)
    df['WORKPLACE63'] = df['WORKPLACE'].apply(lambda x: 1 if x == 63 else 0)
    df['WORKPLACE64'] = df['WORKPLACE'].apply(lambda x: 1 if x == 64 else 0)
    df['WORKPLACE65'] = df['WORKPLACE'].apply(lambda x: 1 if x == 65 else 0)
    df['WORKPLACE66'] = df['WORKPLACE'].apply(lambda x: 1 if x == 66 else 0)
    df['WORKPLACE67'] = df['WORKPLACE'].apply(lambda x: 1 if x == 67 else 0)
    df['WORKPLACE68'] = df['WORKPLACE'].apply(lambda x: 1 if x == 68 else 0)
    df['MRG'] = df['MRG'].apply(lambda x: 0 if x in [91, 93, 94, 95, 96, 97] else 1)
    df['PT'] = df['PT'] - 1

    df = df.drop(
        columns=['SEX', 'EDU', 'IND', 'OCC', 'WKCLASS', 'WORKPLACE']
    )

    return df


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(82, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.linear(x)


def load_checkpoint():
    load_path = "static/models/Model01.pt"
    model = LinearRegressionModel()
    state_dict = torch.load(load_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


def predict(data, to_index=False):
    data = torch.from_numpy(data).float()
    predict = globals.model(data.to('cuda'))
    if to_index:
        return torch.argmax(predict, dim=0).to('cpu')
    return predict.to('cpu')


def get_payment(data):
    data = change_immutable_multi_dict_to_dict(data)
    data = preprocess(data)
    res = predict(data.values)
    return round(res.detach().numpy()[0] * 100000)


def get_age_list(data):
    age_list = [i for i in range(15, 61, 5)]
    data_list = [change_immutable_multi_dict_to_dict(data, 'AGE', i) for i in range(15, 61, 5)]
    data_list.insert(0, change_immutable_multi_dict_to_dict(data))
    data_df = pd.DataFrame(data_list)
    processed_df = preprocess(data_df)
    output = predict(processed_df.to_numpy())
    res_list = [round(i * 100000) for i in output.flatten().tolist()]
    return age_list, res_list[1:], res_list[0]


def get_best_type(data, dict_data, change_type):
    index_list = [i for i, v in dict_data.items()]
    data_list = [change_immutable_multi_dict_to_dict(data, change_type, i) for i, v in dict_data.items()]
    data_df = pd.DataFrame(data_list)
    processed_df = preprocess(data_df)
    output = predict(processed_df.to_numpy(), to_index=True)
    return dict_data[index_list[output.item()]]


def get_best_occ(data):
    occ_dict = {0: "無業者（含無職業家庭之經濟戶長）", 1: "民意代表、主管及經理人員", 2: "專業人員",
                3: "技術員及助理專業人員", 4: "事務支援人員", 5: "服務及銷售工作人員", 61: "農事、畜牧及有關工作者",
                62: "林業生產人員", 63: "漁業生產人員", 7: "技藝有關工作人員", 8: "機械設備操作及組裝人員",
                9: "基層技術工及勞力工"}
    return get_best_type(data, occ_dict, 'OCC')


def get_best_ind(data):
    ind_dict = {0: "無業者（含無職業家庭之經濟戶長）", 1: "農、牧業", 2: "林業及伐木業", 3: "漁業", 5: "礦業及土石採取業",
                8: "製造業", 35: "電力及燃氣供應業", 36: "用水供應及污染整治業", 41: "營造業", 45: "批發及零售業",
                55: "住宿及餐飲業", 49: "運輸及倉儲業", 58: "資訊及通訊傳播業", 64: "金融及保險業", 67: "不動產業",
                69: "專業、科學及技術服務業", 77: "支援服務業", 85: "教育服務業", 86: "醫療保健及社會工作服務業",
                90: "藝術、娛樂及休閒服務業", 94: "其他服務業", 83: "公共行政及國防;強制性社會安全"}

    return get_best_type(data, ind_dict, 'IND')


def get_best_edu(data):
    edu_dict = {1: "不識字", 2: "自修", 3: "國小", 4: "國(初)中(初職)", 5: "高中", 6: "高職",
                7: "專科(五專前三年劃記高職)", 8: "大學", 9: "碩士", 10: "博士"}
    return get_best_type(data, edu_dict, 'EDU')


def get_best_work_place(data):
    work_place_dict = {1: "離島", 2: "宜蘭縣", 3: "桃園縣", 4: "新竹縣", 5: "苗栗縣", 7: "彰化縣", 8: "南投縣",
                       9: "雲林縣", 10: "嘉義縣", 13: "屏東縣", 14: "台東縣", 15: "花蓮縣", 16: "澎湖縣", 17: "基隆市",
                       18: "新竹市", 20: "嘉義市", 63: "臺北市", 64: "高雄市", 65: "新北市", 66: "臺中市", 67: "臺南市",
                       68: "桃園市"}
    return get_best_type(data, work_place_dict, 'WORKPLACE')
