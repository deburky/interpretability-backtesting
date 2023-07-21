import pandas as pd
from sklearn.model_selection import train_test_split
from optbinning import BinningProcess, OptimalBinning

def bin_data(x, y, feature_name, monotonic_trend, plot):
    """
    Binner to produce bins and information value
    """
    optb = OptimalBinning(name=feature_name,
                          dtype="numerical",
                          solver="cp",
                          monotonic_trend=monotonic_trend)

    x_woe = optb.fit(x, y)

    binning_table = pd.DataFrame(optb.binning_table.build())

    return binning_table

# FICO Explainable Machine Learning Challenge dataset
# https://community.fico.com/s/explainable-machine-learning-challenge?tabset-158d9=d157e

print(f"Importing dataset")

csv_file = f"https://raw.githubusercontent.com/deburky/boosting-scorecards/main/heloc_dataset_v1.csv"
data = pd.read_csv(csv_file)
data['RiskPerformance'].replace({'Good': 0, 'Bad': 1}, inplace=True)

print(f"Importing dataset - Done")

# Special codes
special_codes = [-9, -8, -7]

# Binning fit parameters
# binning fit parameters
binning_fit_params = {
    "ExternalRiskEstimate": {"monotonic_trend": "descending"},
    "MSinceOldestTradeOpen": {"monotonic_trend": "descending"},
    "MSinceMostRecentTradeOpen": {"monotonic_trend": "descending"},
    "AverageMInFile": {"monotonic_trend": "descending"},
    "NumSatisfactoryTrades": {"monotonic_trend": "descending"},
    "NumTrades60Ever2DerogPubRec": {"monotonic_trend": "ascending"},
    "NumTrades90Ever2DerogPubRec": {"monotonic_trend": "ascending"},
    "PercentTradesNeverDelq": {"monotonic_trend": "descending"},
    "MSinceMostRecentDelq": {"monotonic_trend": "descending"},
    "NumTradesOpeninLast12M": {"monotonic_trend": "ascending"},
    "MSinceMostRecentInqexcl7days": {"monotonic_trend": "descending"},
    "NumInqLast6M": {"monotonic_trend": "ascending"},
    "NumInqLast6Mexcl7days": {"monotonic_trend": "ascending"},
    "NetFractionRevolvingBurden": {"monotonic_trend": "ascending"},
    "NetFractionInstallBurden": {"monotonic_trend": "ascending"},
    "NumBank2NatlTradesWHighUtilization": {"monotonic_trend": "ascending"},
    "NumTotalTrades": {"monotonic_trend": "auto_asc_desc"},
    "NumRevolvingTradesWBalance": {"monotonic_trend": "auto_asc_desc"},
    "NumInstallTradesWBalance": {"monotonic_trend": "auto_asc_desc"},
    "PercentTradesWBalance": {"monotonic_trend": "auto_asc_desc"},
    "MaxDelq2PublicRecLast12M": {"monotonic_trend": "auto_asc_desc"},
    "PercentInstallTrades": {"monotonic_trend": "auto_asc_desc"},
    "MaxDelqEver": {"monotonic_trend": "auto_asc_desc"},
}

# Data preparation
print(f"Perform train / test split")

variable_names = (pd.DataFrame(binning_fit_params).T).reset_index()['index'].to_list()

# Features and target
X = data[variable_names + ['RiskPerformance']].copy()
y = X.pop('RiskPerformance')

# Sampling
ix_train, ix_test = train_test_split(
    X.index,
    stratify=y,
    test_size=0.3,
    random_state=24
)

print(f"Perform train / test split - Done")

# Binning table for all variables
print(f"Perform binning for all variables")

binning_summary = pd.DataFrame()

for variable in variable_names:
    binning_table = bin_data(
        X.loc[ix_train][variable],
        y.loc[ix_train],
        feature_name=variable,
        monotonic_trend=binning_fit_params[variable]['monotonic_trend'],
        plot=False
    )

    binning_table['Variable'] = variable

    binning_summary = pd.concat([binning_summary, binning_table], axis=0)

print(f"Binning for all variables - Done")

# Apply post-processing
cols = binning_summary.columns.to_list()
cols.remove('Variable')
binning_summary = pd.concat([binning_summary['Variable'], binning_summary[cols]], axis=1)
binning_summary[['lower_bound', 'upper_bound']] = binning_summary['Bin'].str.strip('[]()').str.split(', ', expand=True)

binning_summary = binning_summary[~binning_summary['lower_bound'].isin(['Special', 'Missing'])].copy()
binning_summary = binning_summary[binning_summary['upper_bound'].notnull()].copy()

cols_to_drop = ['Non-event', 'Event', 'WoE', 'JS']
binning_summary.drop(cols_to_drop, axis=1, inplace=True)

binning_summary.rename(
    {
        'Variable': 'variable',
        'Bin': 'bin',
        'Count': 'count',
        'Count (%)': 'count_perc',
        'Event rate': 'bad_rate',
        'IV': 'iv',
    },
    axis=1,
    inplace=True
)

# Save dataframe with bins
binning_summary.to_csv('binning_summary.csv', index=False)

print(f"Binning table - saved")

# create a dataset with bins
binning_process = BinningProcess(variable_names, 
                                 special_codes=special_codes,
                                 binning_fit_params=binning_fit_params)

binning_process.fit(X.loc[ix_train], y.loc[ix_train])

data_binned = binning_process.transform(data, metric='bins')
data_binned = pd.concat([data['RiskPerformance'], data_binned], axis=1)

# Save dataframe with bins
data_binned.to_csv('dataset_with_bins.csv', index=False)

print(f"Binned data - saved")