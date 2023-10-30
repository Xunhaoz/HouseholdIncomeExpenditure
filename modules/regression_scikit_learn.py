import pandas as pd
from pathlib import Path

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import ARDRegression

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.cross_decomposition import PLSRegression

from sklearn.kernel_ridge import KernelRidge

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR


class HouseholdIncomeExpenditureDataLoader:

    def __init__(self):
        ori_csv = Path('../static/excel/all_ori.csv')
        processed_csv = Path('../static/excel/all_processed.csv')

        self.all_df = pd.read_csv(ori_csv)
        if processed_csv.exists():
            self.processed_df = pd.read_csv(processed_csv)
        else:
            self._preprocess()
            self.processed_df.to_csv(processed_csv, index=False)

    def _preprocess(self):
        self.processed_df = self.all_df.copy()
        self._apply_series('REL', 'REL', 1)
        self._apply_series('SEX', 'MALE', 1)
        self._apply_series('SEX', 'FEMALE', 2)

        self.processed_df['AGE'] = self.processed_df['AGE'] / 100
        self.processed_df['MRG'] = self.processed_df['MRG'].apply(lambda x: 0 if x in [91, 93, 94, 95, 96, 97] else 1)
        self.processed_df['PT'] = self.processed_df['PT'] - 1

        self._encoding_pipline()

        self.processed_df = self.processed_df.drop(
            columns=['YEAR', 'ID', 'PERSON', 'SEX', 'PROV', 'EDU', 'F_EDU', 'DPT', 'IND', 'OCC', 'IND2', 'OCC2', 'ECON',
                     'OUTPATIENT', 'INPATIENT', 'HEALTH_INS', 'HI_PAYER', 'HI_FEE', 'INSURE_ID1', 'INSURE_MONTH1',
                     'INSURE_ID2', 'INSURE_MONTH2', 'DEPENDENTS', 'BIRTH_Y', 'BIRTH_MONTH', 'SRN', 'ROC', 'INC_OLD',
                     'NHICLASS', 'WHOPAY', 'DEPEND', 'SI1CLASS', 'SI1MONTH', 'SI2CLASS', 'SI2MONTH',
                     'WKCLASS', 'WORK', 'WORKPLACE', 'EQUIV']
        )

    def _encoding_pipline(self):
        self._one_hot_encoding('EDU')
        self._one_hot_encoding('IND')
        self._one_hot_encoding('OCC')
        self._one_hot_encoding('WKCLASS')
        self._one_hot_encoding('WORKPLACE')

    def _one_hot_encoding(self, col_name: str):
        value_counts = self.processed_df[col_name].value_counts()

        for key in value_counts.keys():
            self.processed_df[f"{col_name}_{key}"] = self.processed_df[col_name].apply(lambda x: 1 if x == key else 0)

    def _apply_series(self, ori_col, new_col, con):
        self.processed_df[new_col] = self.processed_df[ori_col].apply(lambda x: 1 if x == con else 0)


def base_line(name, model, train_data, train_label, dict_for_record):
    scores = cross_val_score(model, train_data, train_label, scoring='neg_mean_squared_error')
    dict_for_record['name'].append(name)
    dict_for_record['score'].append(-scores.mean())
    print(f"{name} mean: {-scores.mean(): 0.2f}, std: {scores.std(): 0.2f}")


if __name__ == '__main__':
    hie_dataloader = HouseholdIncomeExpenditureDataLoader()
    train_label = hie_dataloader.processed_df['ITM40']
    train_data = hie_dataloader.processed_df.drop(columns=['ITM40'])

    dict_for_record = {
        'name': [],
        'score': []
    }

    random_forest_regressor = RandomForestRegressor(n_jobs=-1)
    base_line('random_forest_regressor', random_forest_regressor, train_data, train_label, dict_for_record)

    gradient_boosting_regressor = GradientBoostingRegressor()
    base_line('gradient_boosting_regressor', gradient_boosting_regressor, train_data, train_label, dict_for_record)

    k_neighbors_regressor = KNeighborsRegressor(n_jobs=-1)
    base_line('k_neighbors_regressor', k_neighbors_regressor, train_data, train_label, dict_for_record)

    elastic_net = ElasticNet()
    base_line('elastic_net', elastic_net, train_data, train_label, dict_for_record)

    sgd_regressor = SGDRegressor()
    base_line('sgd_regressor', sgd_regressor, train_data, train_label, dict_for_record)

    svr = SVR()
    base_line('svr', svr, train_data, train_label, dict_for_record)

    bayesian_ridge = BayesianRidge()
    base_line('bayesian_ridge', bayesian_ridge, train_data, train_label, dict_for_record)

    kernel_ridge = KernelRidge()
    base_line('kernel_ridge', kernel_ridge, train_data, train_label, dict_for_record)

    linear_regression = LinearRegression(n_jobs=-1)
    base_line('linear_regression', linear_regression, train_data, train_label, dict_for_record)

    ridge = Ridge()
    base_line('ridge', ridge, train_data, train_label, dict_for_record)

    tweedie_regressor = TweedieRegressor()
    base_line('tweedie_regressor', tweedie_regressor, train_data, train_label, dict_for_record)

    theil_sen_regressor = TheilSenRegressor(n_jobs=-1)
    base_line('theil_sen_regressor', theil_sen_regressor, train_data, train_label, dict_for_record)

    RANSAC_regressor = RANSACRegressor()
    base_line('RANSAC_regressor', RANSAC_regressor, train_data, train_label, dict_for_record)

    huber_regressor = HuberRegressor()
    base_line('huber_regressor', huber_regressor, train_data, train_label, dict_for_record)

    passive_aggressive_regressor = PassiveAggressiveRegressor()
    base_line('passive_aggressive_regressor', passive_aggressive_regressor, train_data, train_label, dict_for_record)

    gamma_regressor = GammaRegressor()
    base_line('gamma_regressor', gamma_regressor, train_data, train_label, dict_for_record)

    poisson_regressor = PoissonRegressor()
    base_line('poisson_regressor', poisson_regressor, train_data, train_label, dict_for_record)

    ARD_regression = ARDRegression()
    base_line('ARD_regression', ARD_regression, train_data, train_label, dict_for_record)

    linear_SVR = LinearSVR()
    base_line('linear_SVR', linear_SVR, train_data, train_label, dict_for_record)

    nu_SVR = NuSVR()
    base_line('nu_SVR', nu_SVR, train_data, train_label, dict_for_record)

    gaussian_process_regressor = GaussianProcessRegressor()
    base_line('gaussian_process_regressor', gaussian_process_regressor, train_data, train_label, dict_for_record)

    PLS_regression = PLSRegression()
    base_line('PLS_regression', PLS_regression, train_data, train_label, dict_for_record)

    extra_tree_regressor = ExtraTreeRegressor()
    base_line('extra_tree_regressor', extra_tree_regressor, train_data, train_label, dict_for_record)

    decision_tree_regressor = DecisionTreeRegressor()
    base_line('decision_tree_regressor', decision_tree_regressor, train_data, train_label, dict_for_record)

    ada_boost_regressor = AdaBoostRegressor()
    base_line('ada_boost_regressor', ada_boost_regressor, train_data, train_label, dict_for_record)

    bagging_regressor = BaggingRegressor(n_jobs=-1)
    base_line('bagging_regressor', bagging_regressor, train_data, train_label, dict_for_record)

    extra_trees_regressor = ExtraTreesRegressor(n_jobs=-1)
    base_line('extra_trees_regressor', extra_trees_regressor, train_data, train_label, dict_for_record)

    hist_gradient_boosting_regressor = HistGradientBoostingRegressor()
    base_line('hist_gradient_boosting_regressor', hist_gradient_boosting_regressor, train_data, train_label,
              dict_for_record)

    MLP_regressor = MLPRegressor()
    base_line('MLP_regressor', MLP_regressor, train_data, train_label, dict_for_record)

    pd.DataFrame(dict_for_record).to_csv('../static/excel/predict_record.csv')
