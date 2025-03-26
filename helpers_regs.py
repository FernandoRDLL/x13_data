from os import environ
environ['OMP_NUM_THREADS'] = '6'
import pandas as pd
import numpy as np
import statsmodels.api as sm
from numpy import sqrt
from scipy.stats import t
import matplotlib.pyplot as plt
from numpy.linalg import pinv, inv
from matplotlib.ticker import ScalarFormatter
plt.rcParams['font.family'] = 'serif'
from numba import jit
from tqdm.notebook import tqdm
from scipy.stats import t

abbs = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands': 'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'United States': 'US',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'West Virginia': 'WV',
    'Virginia': 'VA',
    'Washington': 'WA',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

def replace_state_names(names, abbreviations = abbs):
    replaced_names = []
    for name in names:
        for state, abbreviation in abbreviations.items():
            name = name.replace(state.replace(" ", "_"), abbreviation)
        replaced_names.append(name)
    return replaced_names
def gen_regression_vars(df: pd.DataFrame, X_var: str, Y_var: str, n_lags: int, dummies: bool = False, freq: str = None, date_var: str = None) -> list:
    """
    Generate regression variables for OLS regression.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    X_var (str): The name of the independent variable.
    Y_var (str): The name of the dependent variable.
    n_lags (int): The number of lags to include for the independent variable.
    dummies (bool): Whether to include dummy variables for months or quarters.
    freq (str): The frequency of the data ('m' for monthly, 'q' for quarterly).
    date_var (str): The name of the date variable for generating dummies.

    Returns:
    list: A list containing the dependent variable array and the independent variables array with a constant term.
    """
    aux = df[[Y_var, X_var]].copy()
    aux['lag_y'] = aux[Y_var].shift(1)
    for i in range(n_lags):
        aux[f'lag_{i+1}'] = aux[X_var].shift(i+1)
    if dummies:
        if freq == 'm':
            dummies = pd.get_dummies(df[date_var].dt.month).astype(int).iloc[:, 1:]
        if freq == 'q':
            dummies = pd.get_dummies(((df[date_var].dt.month - 1) // 3) + 1).astype(int).iloc[:, 1:]
        aux = pd.concat([aux, dummies], axis=1)
    aux = aux.drop(columns=X_var)
    aux = aux.dropna()
    return [aux.iloc[:, 0].values, sm.add_constant(aux.iloc[:, 1:].values)]
def gen_bic_comparisons_for_maxlags_one_state(df: pd.DataFrame, reg_var: str, tgt_var: str, max_lags: int, dummies: bool = False, freq: str = None, date_var: str = None) -> dict:
    """
    Generate BIC comparisons for a range of lag lengths for a given state.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    reg_var (str): The name of the independent variable.
    tgt_var (str): The name of the dependent variable.
    max_lags (int): The maximum number of lags to include for the independent variable.
    dummies (bool): Whether to include dummy variables for months or quarters.
    freq (str): The frequency of the data ('m' for monthly, 'q' for quarterly).
    date_var (str): The name of the date variable for generating dummies.

    Returns:
    dict: A dictionary containing the BIC values for each lag length.
    """
    aux = [gen_regression_vars(df, reg_var, tgt_var, i, dummies, freq, date_var) for i in range(1, max_lags + 1)]
    n_obs = aux[-1][0].size
    bic_res = {'reg_var': reg_var}
    for i in range(len(aux)): 
        Y_bic_reg = aux[i][0][-n_obs:]
        X_bic_reg = aux[i][1][-n_obs:]
        bic_res[f'n = {i+1}'] = compute_bic(Y_bic_reg, X_bic_reg)
    return bic_res
def compute_bic(y: np.ndarray, x: np.ndarray) -> float:
    """
    Compute the Bayesian Information Criterion (BIC) for an OLS regression model.

    Parameters:
    y (np.ndarray): The dependent variable array.
    x (np.ndarray): The independent variables array with a constant term.

    Returns:
    float: The BIC value of the fitted model.
    """
    model = sm.OLS(y, x)
    results = model.fit()
    return results.bic

def std_lincomb(cov_mat: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the standard deviation of a linear combination of random variables.

    Parameters:
    cov_mat (np.ndarray): The covariance matrix of the variables.
    v (np.ndarray): The vector representing the linear combination.

    Returns:
    float: The standard deviation of the linear combination.
    """
    return sqrt(v @ cov_mat @ v)
def get_betas_cov_mat(y: np.ndarray, x: np.ndarray, errors: str = 'HAC', n_bootstrap: int = None) -> tuple:
    """
    Get the regression coefficients and covariance matrix for OLS regression.

    Parameters:
    y (np.ndarray): The dependent variable array.
    x (np.ndarray): The independent variables array with a constant term.
    errors (str): The type of errors to use ('HAC' or 'bootstrap').
    n_bootstrap (int): The number of bootstrap samples to use if errors is 'bootstrap'.

    Returns:
    tuple: A tuple containing the regression coefficients, covariance matrix, and bootstrap history (if applicable).
    """
    y = y.astype(np.float64)
    x = x.astype(np.float64)
    if errors == 'HAC':
        model = sm.OLS(y, x)
        results = model.fit()
        results_nw = results.get_robustcov_results(cov_type='HAC', maxlags=np.ceil(y.size ** 0.25).astype(int))
        return results.params, results_nw.cov_params(), None
    else:
        samples = np.random.randint(0, high=len(y), size=(n_bootstrap, len(y)))
        betas_btsp = np.array([bootstrap_sample_and_OLS(y[sample_idx], x[sample_idx]) for sample_idx in samples])
        mean_btsp, cov_betas_btsp = betas_btsp.mean(axis=0), np.cov(betas_btsp, rowvar=False)
        return mean_btsp, cov_betas_btsp, betas_btsp

@jit(nopython=True)
def bootstrap_sample_and_OLS(y_pop: np.ndarray, x_pop: np.ndarray) -> np.ndarray:
    """
    Perform OLS regression on a bootstrap sample.

    Parameters:
    y_pop (np.ndarray): The dependent variable array for the bootstrap sample.
    x_pop (np.ndarray): The independent variables array for the bootstrap sample.

    Returns:
    np.ndarray: The regression coefficients for the bootstrap sample.
    """
    return inv(x_pop.T @ x_pop) @ (x_pop.T @ y_pop)

def gen_lincomb_stats(name_y: str, name_x: str, coefs: np.ndarray, v: np.ndarray, cov_mat: np.ndarray, alpha: float, errors: str, deg_freedom: int = None, betas_history_btsp: np.ndarray = None) -> dict:
    """
    Generate statistics for a linear combination of regression coefficients.

    Parameters:
    name_y (str): The name of the dependent variable.
    name_x (str): The name of the independent variable.
    coefs (np.ndarray): The regression coefficients.
    v (np.ndarray): The vector representing the linear combination.
    cov_mat (np.ndarray): The covariance matrix of the coefficients.
    alpha (float): The significance level for confidence intervals.
    errors (str): The type of errors to use ('HAC' or 'bootstrap').
    deg_freedom (int): The degrees of freedom for the t-distribution.
    betas_history_btsp (np.ndarray): The bootstrap history of the coefficients (if applicable).

    Returns:
    dict: A dictionary containing the statistics for the linear combination.
    """
    lc_value = coefs @ v
    std_error = std_lincomb(cov_mat, v)
    t_stat = lc_value / std_error
    sig_flag = (np.abs(t_stat) > np.abs(t.ppf(alpha / 2, deg_freedom))).astype(int)
    if errors == 'HAC':
        l_ci = lc_value + t.ppf(alpha / 2, deg_freedom) * std_error
        u_ci = lc_value + t.ppf(1 - alpha / 2, deg_freedom) * std_error
        p_value = 2 * (1 - t.cdf(abs(t_stat), deg_freedom))
    else:
        lincomb_hist = betas_history_btsp @ v
        l_ci = np.percentile(lincomb_hist, alpha * 50)
        u_ci = np.percentile(lincomb_hist, 100 - alpha * 50)
        p_value = 2 * (1 - t.cdf(abs(t_stat), deg_freedom))
    stats = {
        'series name': name_y,
        'dependent variable': name_x,
        'value': lc_value,
        'std error': std_error,
        't-statistic': t_stat,
        'p_val': p_value,
        'significance': sig_flag,
        f'LB ({100 - 100 * alpha:.2f}% ci)': l_ci,
        f'UB ({100 - 100 * alpha:.2f}% ci)': u_ci,
        'error type': errors
    }
    return stats

def run_OLS_and_gen_stats(df: pd.DataFrame, X_var: str, Y_var: str, n_lags: int, dummies: bool, freq: str, date_var: str, alpha: float, errors: str = 'HAC', n_btsp: int = None) -> dict:
    """
    Run OLS regression and generate statistics for a linear combination of coefficients.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    X_var (str): The name of the independent variable.
    Y_var (str): The name of the dependent variable.
    n_lags (int): The number of lags to include for the independent variable.
    dummies (bool): Whether to include dummy variables for months or quarters.
    freq (str): The frequency of the data ('m' for monthly, 'q' for quarterly).
    date_var (str): The name of the date variable for generating dummies.
    alpha (float): The significance level for confidence intervals.
    errors (str): The type of errors to use ('HAC' or 'bootstrap').
    n_btsp (int): The number of bootstrap samples to use if errors is 'bootstrap'.

    Returns:
    dict: A dictionary containing the statistics for the linear combination.
    """
    y_tab, x_tab = gen_regression_vars(df, X_var, Y_var, n_lags, dummies, freq, date_var)
    betas, cov_betas, hist_betas = get_betas_cov_mat(y_tab, x_tab, errors, n_btsp)
    lc_vector = np.concatenate([np.zeros(2), np.ones(n_lags), np.zeros(x_tab.shape[1] - 2 - n_lags)])
    df_tstat = len(y_tab) - x_tab.shape[1]
    return gen_lincomb_stats(Y_var, X_var, betas, lc_vector, cov_betas, alpha, errors, df_tstat, hist_betas)
def apply_regex_dict(series: pd.Series, replacements: dict) -> pd.Series:
    """
    Apply a dictionary of regex replacements to a pandas Series.

    Parameters:
    series (pd.Series): The input Series to apply replacements to.
    replacements (dict): A dictionary where keys are regex patterns and values are replacement strings.

    Returns:
    pd.Series: The Series with applied replacements.
    """
    for pattern, repl in replacements.items():
        series = series.str.replace(pattern, repl, regex=True)
    return series

def gen_table_state_results(df: pd.DataFrame, var_list: list[str], Y_var: str, n_lags: int, dummies: bool, freq: str, date_var: str, alpha: float, errors: str = 'HAC', n_btsp: int = None, replacements: dict = None, custom_order: bool = False, first_row_for_co: str = None, order_df: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a table of regression results for a list of variables.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    var_list (list of str): The list of independent variables.
    Y_var (str): The name of the dependent variable.
    n_lags (int): The number of lags to include for the independent variable.
    dummies (bool): Whether to include dummy variables for months or quarters.
    freq (str): The frequency of the data ('m' for monthly, 'q' for quarterly).
    date_var (str): The name of the date variable for generating dummies.
    alpha (float): The significance level for confidence intervals.
    errors (str): The type of errors to use ('HAC' or 'bootstrap').
    n_btsp (int): The number of bootstrap samples to use if errors is 'bootstrap'.
    replacements (dict): A dictionary of regex replacements for variable names.
    custom_order (bool): Whether to use a custom order for the results table.
    first_row_for_co (str): The first row for custom ordering.
    order_df (pd.DataFrame): The DataFrame for custom ordering.

    Returns:
    tuple: A tuple containing the results table and the ordered table.
    """
    results_table = [run_OLS_and_gen_stats(df, var, Y_var, n_lags, dummies, freq, date_var, alpha, errors, n_btsp) for var in var_list]
    results_table = pd.DataFrame(results_table)
    if replacements:
        results_table['dependent variable'] = apply_regex_dict(results_table['dependent variable'], replacements)
    if custom_order:
        results_table = order_df.merge(results_table, on='dependent variable', how='left')
    else:
        results_table = results_table.sort_values('value', ascending=False)
        ordered_table = [results_table[results_table['dependent variable'] == first_row_for_co], results_table[results_table['dependent variable'] != first_row_for_co]]
        results_table = pd.concat(ordered_table, ignore_index=True)
    results_table = results_table.reset_index(drop=True)
    results_table = results_table[['dependent variable', 'value', 't-statistic', 'significance', f'LB ({100-100*alpha:.2f}% ci)', f'UB ({100-100*alpha:.2f}% ci)', 'p_val']]
    return results_table, results_table[['dependent variable']]

def make_regression_tables(df: pd.DataFrame, var_list: list[str], Y_var: str, lag_list: list[int], dummies: bool, freq: str, date_var: str, alpha: float, errors: str = 'HAC', n_btsp: int = None, replacements: dict = None, first_row_for_co: str = None) -> list[pd.DataFrame]:
    """
    Generate regression tables for a list of lag lengths.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    var_list (list of str): The list of independent variables.
    Y_var (str): The name of the dependent variable.
    lag_list (list of int): The list of lag lengths to include for the independent variable.
    dummies (bool): Whether to include dummy variables for months or quarters.
    freq (str): The frequency of the data ('m' for monthly, 'q' for quarterly).
    date_var (str): The name of the date variable for generating dummies.
    alpha (float): The significance level for confidence intervals.
    errors (str): The type of errors to use ('HAC' or 'bootstrap').
    n_btsp (int): The number of bootstrap samples to use if errors is 'bootstrap'.
    replacements (dict): A dictionary of regex replacements for variable names.
    first_row_for_co (str): The first row for custom ordering.

    Returns:
    list: A list of DataFrames containing the regression tables.
    """
    if len(lag_list) > 1:
        res_list = []
        res, order_list = gen_table_state_results(df, var_list, Y_var, lag_list[0], dummies, freq, date_var, alpha, errors, n_btsp, replacements, False, first_row_for_co, None)
        res.insert(1, 'n_lags', lag_list[0])
        res['index'] = res.index
        res_list.append(res)
        for lag in lag_list[1:]:
            res, _ = gen_table_state_results(df, var_list, Y_var, lag, dummies, freq, date_var, alpha, errors, n_btsp, replacements, True, None, order_list)
            res.insert(1, 'n_lags', lag)
            res['index'] = res.index
            res_list.append(res)
        return res_list
    elif len(lag_list) == 1:
        res, _ = gen_table_state_results(df, var_list, Y_var, lag_list[0], dummies, freq, date_var, alpha, errors, n_btsp, replacements, False, first_row_for_co, None)
        res.insert(1, 'n_lags', lag_list[0])
        res['index'] = res.index
        return [res]
    else:
        print('Not valid')

def get_betas_conf_int(y: np.ndarray, x: np.ndarray, alpha_test: float, errors: str = 'HAC', n_bootstrap: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the regression coefficients and confidence intervals for OLS regression.

    Parameters:
    y (np.ndarray): The dependent variable array.
    x (np.ndarray): The independent variables array with a constant term.
    alpha_test (float): The significance level for confidence intervals.
    errors (str): The type of errors to use ('HAC' or 'bootstrap').
    n_bootstrap (int): The number of bootstrap samples to use if errors is 'bootstrap'.

    Returns:
    tuple: A tuple containing the regression coefficients and confidence intervals.
    """
    y = y.astype(np.float64)
    x = x.astype(np.float64)
    if errors == 'HAC':
        model = sm.OLS(y, x)
        results = model.fit()
        results_nw = results.get_robustcov_results(cov_type='HAC', maxlags=np.ceil(y.size ** 0.25).astype(int))
        conf_intervals = results_nw.conf_int(alpha=alpha_test)
        return results.params, conf_intervals
    else:
        samples = np.random.randint(0, high=len(y), size=(n_bootstrap, len(y)))
        betas_btsp = np.array([bootstrap_sample_and_OLS(y[sample_idx], x[sample_idx]) for sample_idx in samples])
        mean_btsp = betas_btsp.mean(axis=0)
        conf_intervals = np.c_[np.quantile(betas_btsp, alpha_test / 2, axis=0), np.quantile(betas_btsp, 1 - alpha_test / 2, axis=0)]
        return mean_btsp, conf_intervals
def plot_state_loadings_for_n_lags(df, X_var, Y_var, title_x_graph, n_lags, dummies, date_var, freq, conf_lvl, errors = 'HAC', n_btsp = None, save = False, filename = None, show = False):
    y_tab, x_tab = gen_regression_vars(df, X_var, Y_var, n_lags, dummies, date_var, freq)
    betas, ci_alpha = get_betas_conf_int(y_tab, x_tab, 1 - conf_lvl, errors, n_btsp)
    plt.plot(np.arange(1,n_lags + 1), betas[2:2+n_lags])
    plt.fill_between(np.arange(1,n_lags + 1), ci_alpha[2:2+n_lags,0], ci_alpha[2:2+n_lags,1], color='b', alpha=.1)
    state_title = X_var.replace('cvol_gar_', '').replace('_sfh', ' SFH')
    plt.xlim([1, n_lags])
    plt.xticks(np.arange(1, n_lags+1))
    y_min = np.min(ci_alpha[2:2+n_lags,0])
    y_max = np.max(ci_alpha[2:2+n_lags,1])
    y_range = y_max - y_min
    padding = 0.5 * y_range  # Add 10% padding to the y-limits
    plt.ylim([y_min - padding, y_max + padding])
    plt.xlabel('Lag (in months)')
    plt.ylabel('Loading value')
    ax = plt.gca()
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    plt.title(f'Loadings of lagged values of {state_title} on {title_x_graph}')
    note = f"Note: The shaded area represents the {100*conf_lvl:.2f}% confidence interval of the loading estimates."
    plt.text(0, -0.15, note, ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)
    plt.tight_layout()
    if save: plt.savefig(filename, dpi = 300)
    if show: plt.show()
    plt.close()
def get_dynamic_multipliers(df, y, x, shift_order, alpha_test):
    lag_order = -shift_order 
    if lag_order < 0:
        model = sm.OLS(df[y].values[:lag_order], sm.add_constant(df[x].shift(lag_order).values[:lag_order]))
    if lag_order > 0:
        model = sm.OLS(df[y].values[lag_order:], sm.add_constant(df[x].shift(lag_order).values[lag_order:]))
    if lag_order == 0:
        model = sm.OLS(df[y].values, sm.add_constant(df[x].values))
    results = model.fit()
    return results.params[1], results.conf_int(alpha=alpha_test)[1]
def get_local_projection_dynamic_multipliers(df, y, x, shift_list, alpha_test):
    multipliers = []
    cis = []
    for shift in shift_list:
        res = get_dynamic_multipliers(df, y, x, shift, alpha_test)
        multipliers.append(res[0])
        cis.append(res[1])
    shifts = np.array(shift_list)
    multipliers = np.array(multipliers)
    cis = np.array(cis)
    return shifts, multipliers, cis
def plot_raw_local_projection_dynamic_irf(data, y, x, title_x_graph, shift_list, conf_lvl, save = False, filename = None, show = False):
    state_title = x.replace('cvol_gar_', '').replace('_sfh', ' SFH')
    plot_data = get_local_projection_dynamic_multipliers(data, y, x, shift_list, 1 - conf_lvl)        
    lag_list = plot_data[0]
    multipliers = plot_data[1]
    conf_int_bounds = plot_data[2]
    plt.figure(figsize=(16, 8))
    plt.plot(lag_list, multipliers)
    plt.fill_between(shift_list, conf_int_bounds[:, 0], conf_int_bounds[:, 1], color='b', alpha=.1)
    plt.xlabel('Shift (in months)')
    plt.ylabel('Multiplier')
    ax = plt.gca()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.yaxis.set_major_formatter(formatter)
    plt.xticks(shift_list)
    y_min = np.min(conf_int_bounds[:, 0])
    y_max = np.max(conf_int_bounds[:, 1])
    y_range = y_max - y_min
    padding = 0.25 * y_range  # Add 25% padding to the y-limits
    plt.xlim([min(shift_list), max(shift_list)])
    plt.ylim([y_min - padding, y_max + padding])
    #plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5, color='gray')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    #plt.grid()
    plt.title(f'Local Projection Dynamic IRF: {state_title} on {title_x_graph}')
    note = f"Note: The shaded area represents the {100*conf_lvl:.2f}% confidence interval of the\ndynamic multiplier estimates."
    plt.text(0, -0.20, note, ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)
    plt.tight_layout()
    if save: plt.savefig(filename, dpi = 300)
    if show: plt.show()
    plt.close()
    
def plot_normalized_local_projection_dynamic_irf(data, y, x, title_x_graph, shift_list, conf_lvl, save=False, filename=None, show=False):
    state_title = x.replace('cvol_gar_', '')
    plot_data = get_local_projection_dynamic_multipliers(data, y, x, shift_list, 1 - conf_lvl)
    lag_list = plot_data[0]
    multipliers = plot_data[1]
    multipliers_min = np.min(multipliers)
    multipliers_max = np.max(multipliers)
    multipliers_range = multipliers_max - multipliers_min
    normalized_multipliers = (multipliers - multipliers_min) / multipliers_range
    plt.figure(figsize=(16, 8))
    plt.plot(lag_list, normalized_multipliers)
    plt.xlabel('Shift (in months)')
    plt.ylabel('Normalized Multiplier')
    plt.xlim([min(shift_list), max(shift_list)])
    plt.ylim([-0.1, 1.1])  # Set y-axis to range [0, 1]
    plt.title(f'Local Projection Dynamic IRF: {state_title} on {title_x_graph}')
    plt.xticks(shift_list)
    note = "Note: The multipliers are normalized to the range [0, 1]. CIs are not included."
    plt.text(0, -0.20, note, ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)
    plt.tight_layout()
    if save: plt.savefig(filename, dpi=300)
    if show: plt.show()
    plt.close()

def generate_bar_plot(df: pd.DataFrame, n_lags: int, alpha_test: float, plot_lims: tuple, lower_ci_col: str, upper_ci_col: str, dummies: bool = False, freq: str = None, save: bool = False, filename: str = None, show: bool = False, separation_first_bar: bool = True) -> None:
    """
    Generate a horizontal bar plot with error bars for regression results.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the regression results.
    n_lags (int): The number of lags included in the regression.
    alpha_test (float): The significance level for confidence intervals.
    plot_lims (tuple): The limits for the x-axis of the plot.
    lower_ci_col (str): The column name for the lower confidence interval.
    upper_ci_col (str): The column name for the upper confidence interval.
    dummies (bool): Whether to include dummy variables in the regression.
    freq (str): The frequency of the data ('m' for monthly, 'q' for quarterly).
    save (bool): Whether to save the plot as a file.
    filename (str): The filename to save the plot.
    show (bool): Whether to display the plot.
    separation_first_bar (bool): Whether to separate the first bar from the rest.

    Returns:
    None
    """
    df = df.copy()
    df['Error Lower'] = df['value'] - df[lower_ci_col]
    df['Error Upper'] = df[upper_ci_col] - df['value']
    errors1 = df[['Error Lower', 'Error Upper']].T.values
    
    colors = plt.cm.coolwarm(np.linspace(1, 0, len(df)))
    
    plt.figure(figsize=(12, 12))
    
    # Create the bars
    bars = plt.barh(df['dependent variable'], df['value'], color=colors, capsize=4)

    if separation_first_bar:
        # Adjust the position of the first bar
        offset = -1  # The amount by which to raise the first bar
        for bar in bars:
            bar.set_height(bar.get_height() * 0.8)  # Optional: Adjust bar thickness
        
        bars[0].set_y(bars[0].get_y() + offset)
    
    # Plot error bars separately for each bar
    for i, (bar, error) in enumerate(zip(bars, errors1.T)):
        y = bar.get_y() + bar.get_height() / 2
        eb = plt.errorbar(bar.get_width(), y, xerr=np.array([error]).T, fmt='none', ecolor='black', capsize=4, elinewidth=1)
        if df['significance'].values[i] == 0: 
            eb[-1][0].set_linestyle('--')
    
    # Get current y-tick labels and positions
    yticklabels = [label.get_text() for label in plt.gca().get_yticklabels()]
    ytickpositions = np.arange(len(yticklabels), dtype=float)
    
    if separation_first_bar:
        # Adjust the position of the first label
        ytickpositions[0] += -1
    
    # Set the new y-tick labels and positions
    plt.yticks(ytickpositions, yticklabels, fontsize=8.5)
    
    if n_lags != 1: 
        plt.xlabel('Sum of Coefficients')
    else: 
        plt.xlabel('Coefficient')
    plt.title(f'Results of {n_lags} month lag regression')
    plt.xlim(plot_lims)
    plt.gca().invert_yaxis()
    plt.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
    plt.gcf().subplots_adjust(bottom=0.1)  # Adjust to create space for the notes
    note = f"Note: Solid error bars represent significance at {alpha_test*100:.2f}% confidence level; dashed error bars denote results that are not statistically significant."
    plt.text(0, -0.08, note, ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)
    dummy_freq = {'q': 'quarterly', 'm': 'monthly'}
    if dummies:
        note = f"Regression includes {dummy_freq[freq]} dummies."
    else:
        note = f"Regression does not include dummies."
    plt.text(0, -0.1, note, ha='left', va='center', transform=plt.gca().transAxes, fontsize=8)
    if save: 
        plt.savefig(filename, dpi=300)
    if show: 
        plt.show()
    else: 
        plt.close()

def compute_rolling_corr(df: pd.DataFrame, pop_df: pd.DataFrame, cols: list[str], w: int, mode: str = 'equal') -> list[float]:
    """
    Compute the rolling correlation for a given DataFrame and columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    pop_df (pd.DataFrame): The population DataFrame for weighted correlations.
    cols (list of str): The list of column names to calculate the correlation for.
    w (int): The window size for the rolling correlation.
    mode (str): The mode of correlation calculation ('equal', 'median', 'max', 'min', 'popweights').

    Returns:
    list[float]: The list of rolling correlation values.
    """
    upper_triangle_indices = np.triu_indices(len(cols), k=1)
    df2 = df[cols].copy()
    res = []
    for i in tqdm(range(df.shape[0])):
        if i < w:
            res.append(np.nan)
        else:
            corr_matrix = df2.iloc[i-(w-1):i+1].corr()
            corr_values = corr_matrix.values[upper_triangle_indices]
            if mode == 'equal':
                res.append(np.mean(corr_values))
            elif mode == 'median':
                res.append(np.median(corr_values))
            elif mode == 'max':
                res.append(np.max(corr_values))
            elif mode == 'min':
                res.append(np.min(corr_values))
            elif mode == 'popweights':
                column_pairs = [[corr_matrix.index[i], corr_matrix.columns[j]] for i, j in zip(*upper_triangle_indices)]
                pop_w = gen_pop_weights(pop_df, column_pairs, i+1)
                res.append(np.average(corr_values, None, pop_w))
    return res

def gen_pop_weights(pop_df: pd.DataFrame, column_pairs: list[list[str, str]], index: int, pair_indices: np.ndarray = None) -> np.ndarray:
    """
    Generate population weights for given column pairs.

    Parameters:
    pop_df (pd.DataFrame): The population DataFrame.
    column_pairs (list of tuple): The list of column pairs.
    index (int): The index for the population DataFrame.
    pair_indices (np.ndarray, optional): The precomputed indices for the column pairs.

    Returns:
    np.ndarray: The array of population weights.
    """
    if pair_indices is None:
        col_index = {col: i for i, col in enumerate(pop_df.columns)}
        pair_indices = np.array([(col_index[a], col_index[b]) for a, b in column_pairs])
    row = pop_df.iloc[index].values
    weights = row[pair_indices].sum(axis=1)

    return weights / weights.sum()



def avg_roll_corr_mat(df: pd.DataFrame, cols: list[str], w: int) -> np.ndarray:
    """
    Calculate the average rolling correlation matrix for a given DataFrame and columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    cols (list of str): The list of column names to calculate the correlation matrix for.
    w (int): The window size for the rolling correlation.

    Returns:
    np.ndarray: The average rolling correlation matrix.
    """
    n = len(cols)
    sum_corr = np.zeros(shape=(n, n))
    df2 = df[cols].copy()
    for i in range(df.shape[0]):
        if i < w:
            pass
        else:
            sum_corr += df2.iloc[i-(w-1):i+1].corr().values
    return sum_corr / sum_corr[0, 0]

def gen_popdf():
    pop_df = pd.read_excel('POP.xlsx', sheet_name = 1)[58:]
    pop_df.rename(columns = state_population_mapping, inplace = True)
    pop_df.reset_index(drop = True, inplace = True)
    pop_df['date'] = pd.to_datetime(pop_df['date'])
    pop_df.set_index('date', inplace=True)
    pop_df = pop_df.resample('ME').ffill()
    pop_df.reset_index(inplace=True)
    return pop_df
    
state_population_mapping = {
    'observation_date': 'date',
    'AKPOP': 'Alaska',
    'ALPOP': 'Alabama',
    'ARPOP': 'Arkansas',
    'AZPOP': 'Arizona',
    'CAPOP': 'California',
    'COPOP': 'Colorado',
    'CTPOP': 'Connecticut',
    'DCPOP': 'District_of_Columbia',
    'DEPOP': 'Delaware',
    'FLPOP': 'Florida',
    'GAPOP': 'Georgia',
    'HIPOP': 'Hawaii',
    'IAPOP': 'Iowa',
    'IDPOP': 'Idaho',
    'ILPOP': 'Illinois',
    'INPOP': 'Indiana',
    'KSPOP': 'Kansas',
    'KYPOP': 'Kentucky',
    'LAPOP': 'Louisiana',
    'MAPOP': 'Massachusetts',
    'MDPOP': 'Maryland',
    'MEPOP': 'Maine',
    'MIPOP': 'Michigan',
    'MNPOP': 'Minnesota',
    'MOPOP': 'Missouri',
    'MSPOP': 'Mississippi',
    'MTPOP': 'Montana',
    'NCPOP': 'North_Carolina',
    'NDPOP': 'North_Dakota',
    'NEPOP': 'Nebraska',
    'NHPOP': 'New_Hampshire',
    'NJPOP': 'New_Jersey',
    'NMPOP': 'New_Mexico',
    'NVPOP': 'Nevada',
    'NYPOP': 'New_York',
    'OHPOP': 'Ohio',
    'OKPOP': 'Oklahoma',
    'ORPOP': 'Oregon',
    'PAPOP': 'Pennsylvania',
    'RIPOP': 'Rhode_Island',
    'SCPOP': 'South_Carolina',
    'SDPOP': 'South_Dakota',
    'TNPOP': 'Tennessee',
    'TXPOP': 'Texas',
    'UTPOP': 'Utah',
    'VAPOP': 'Virginia',
    'VTPOP': 'Vermont',
    'WAPOP': 'Washington',
    'WIPOP': 'Wisconsin',
    'WVPOP': 'West_Virginia',
    'WYPOP': 'Wyoming'
}
def state_list():
    us_states_by_census_region = [
        # Northeast (New England and Middle Atlantic)
        "Connecticut", "Maine", "Massachusetts", "New_Hampshire", "New_Jersey", "New_York", "Pennsylvania", "Rhode_Island", "Vermont",
        
        # Midwest (East North Central and West North Central)
        "Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota", "Missouri", "Nebraska", "North_Dakota", "Ohio", "South_Dakota", "Wisconsin",
        
        # South (South Atlantic, East South Central, and West South Central)
        "Alabama", "Arkansas", "Delaware", "Florida", "Georgia", "Kentucky", "Louisiana", "Maryland", "Mississippi", "North_Carolina", "Oklahoma", "South_Carolina", "Tennessee", "Texas", "Virginia", "West_Virginia",
        
        # West (Mountain and Pacific)
        "Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho", "Montana", "Nevada", "New_Mexico", "Oregon", "Utah", "Washington", "Wyoming"
    ]
    return us_states_by_census_region
def classify_cols_garch(data):
    cols = {
        'gar_sfh' : sorted([i for i in data.columns if 'cvol' in i and 'gar' in i and 'SFH'    in i]),
        'gar_tot' : sorted([i for i in data.columns if 'cvol' in i and 'gar' in i and 'TOT'    in i]),
        'gar_mfh' : sorted([i for i in data.columns if 'cvol' in i and 'gar' in i and ' MFH'   in i]),
        'gar_smfh': sorted([i for i in data.columns if 'cvol' in i and 'gar' in i and ' S_MFH' in i]),
        'gar_lmfh': sorted([i for i in data.columns if 'cvol' in i and 'gar' in i and ' L_MFH' in i]),
        'gjr_sfh' : sorted([i for i in data.columns if 'cvol' in i and 'gjr' in i and 'SFH'    in i]),
        'gjr_tot' : sorted([i for i in data.columns if 'cvol' in i and 'gjr' in i and 'TOT'    in i]),
        'gjr_mfh' : sorted([i for i in data.columns if 'cvol' in i and 'gjr' in i and ' MFH'   in i]),
        'gjr_smfh': sorted([i for i in data.columns if 'cvol' in i and 'gjr' in i and ' S_MFH' in i]),
        'gjr_lmfh': sorted([i for i in data.columns if 'cvol' in i and 'gjr' in i and ' L_MFH' in i])
    }
    return cols
def classify_cols_bpg(data):
    cols = {
        'sfh' : sorted([i for i in data.columns if 'SFH' in i and not ('_DC' in i) and not ('_HI' in i) and not ('_AK' in i)]),
        'tot' : sorted([i for i in data.columns if 'TOT' in i and not ('_DC' in i) and not ('_HI' in i) and not ('_AK' in i) ]),
        'mfh' : sorted([i for i in data.columns if 'MFH' in i and '_L_MFH' not in i and '_S_MFH' not in i and not ('_DC' in i) and not ('_HI' in i) and not ('_AK' in i)]),
        'smfh': sorted([i for i in data.columns if '_S_MFH' in i and not ('_DC' in i) and not ('_HI' in i) and not ('_AK' in i)]),
        'lmfh': sorted([i for i in data.columns if '_L_MFH' in i and not ('_DC' in i) and not ('_HI' in i) and not ('_AK' in i)])
    }
    return cols
def classify_cols_duns(data):
    cols = {
        'gar' : sorted([i for i in data.columns if 'gar' in i]),
        'gjr' : sorted([i for i in data.columns if 'gjr' in i]),
        
    }
    return cols
def make_table_from_corr_matrix(corr_matrix_np: np.ndarray, column_names: list[str], labels: list[str] = ['Column1', 'Column2', 'Correlation']) -> pd.DataFrame:
    """
    Convert a correlation matrix to a long-format DataFrame.

    Parameters:
    corr_matrix_np (np.ndarray): The input correlation matrix as a NumPy array.
    column_names (list of str): The list of column names corresponding to the correlation matrix.
    labels (list of str): The labels for the resulting DataFrame columns.

    Returns:
    pd.DataFrame: The long-format DataFrame with correlation values.
    """
    # Convert NumPy matrix to Pandas DataFrame
    corr_df = pd.DataFrame(corr_matrix_np, index=column_names, columns=column_names)
    
    # Convert to long format
    corr_long = corr_df.stack().reset_index()
    corr_long.columns = ['Column1', 'Column2', 'Correlation']
    
    # Remove self-correlations (diagonal 1s)
    corr_long = corr_long[corr_long['Column1'] != corr_long['Column2']]
    
    # Remove duplicate pairs (keep only one of each symmetric pair)
    corr_long['sorted_pair'] = corr_long.apply(lambda x: tuple(sorted([x['Column1'], x['Column2']])), axis=1)
    corr_long = corr_long.drop_duplicates(subset='sorted_pair').drop(columns=['sorted_pair'])

    corr_long.columns = labels
    
    # Display the resulting DataFrame
    return corr_long

def merge_volatility_data(cvol_filename: str, mkt_filename: str, bonds_filename: str) -> pd.DataFrame:
    """
    Reads three Stata files, merges them on the 'date' column, and drops rows with missing values 
    in 'vol_vwretd' and 'vol_btreturn'.
    
    Parameters:
    cvol_filename (str): Path to the cvol data file.
    mkt_filename (str): Path to the market data file.
    bonds_filename (str): Path to the bonds data file.
    
    Returns:
    pd.DataFrame: Merged and cleaned DataFrame.
    """
    cvol = pd.read_stata(cvol_filename).rename(columns={'Date': 'date'})
    data_mkt = pd.read_stata(mkt_filename)
    data_bonds = pd.read_stata(bonds_filename)
    
    data = cvol.merge(data_mkt, on='date', how='left')
    data = data.merge(data_bonds, on='date', how='left')
    
    return data.dropna(subset=['vol_vwretd', 'vol_btreturn'])
