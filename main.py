#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import wilcoxon
from scipy.stats.mstats import gmean
from scipy.special import expit
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import io
import ast
import os
import json

from . import np, pd
from .nonparametric import *
from .basis import *

class CalculateValues:
    """
    Class to calculate theta values and value functions using policy estimation, reward estimation, and basis function expansion.
    
    Parameters
    ----------
    state_df : DataFrame
        DataFrame containing the state values.
    action_df : DataFrame
        DataFrame containing the action values.
    reward_df : DataFrame
        DataFrame containing the reward values.
    next_state_df : DataFrame
        DataFrame containing the next state values.
    """

    def __init__(self, state_df, action_df, reward_df, next_state_df):
        # Initialize dataframes for state, action, reward, and next state
        self.state_df = state_df
        self.action_df = action_df
        self.reward_df = reward_df
        self.next_state_df = next_state_df
        
        # Calculate the maximum lag using a rule of thumb based on the sample size
        self.max_lag = int(4*((self.state_df.shape[0]/100)** (1/3)))
        
        
        
    def fit(self, order: int = 3):
        """
        Fit the model by searching for optimal bandwidth and computing policy, reward, and basis.

        Parameters
        ----------
        search_interval : array-like, optional
            Interval to search for the optimal bandwidth for kernel density estimation.
        cv_k : int, optional
            Number of folds for cross-validation in the search for the optimal bandwidth.
        order : int, optional
            The order of the Chebyshev basis function expansion.
        
        Returns
        -------
        int : 0
            Always returns 0 after fitting.
        """
        
        self.p_basis = PolynomialBasis(order)  # Instantiate the Chebyshev basis object
        
        state_action_df = pd.concat([self.state_df, self.action_df], axis =1)
        n, d = state_action_df.shape
        std_devs = state_action_df.std()
        bandwidths = (4 / ((d + 2) * n)) ** (1 / (d + 4)) * std_devs
        self.h_x = bandwidths[:self.state_df.shape[1]].values
        self.h_a = bandwidths[self.state_df.shape[1]:].values
        print(f"'bandwidths for state': {self.h_x}")
        print(f"'bandwidths for action': {self.h_a}")
        
          
        # Compute the basis functions
        self.basis_df, self.basis_dict, self.hat_psi_next = self.compute_basis(self.h_x, self.state_df, self.next_state_df, self.p_basis)
        
        # Estimate reward and next basis function data for the actions taken
        self.est_r_sa_actual, self.est_next_psi_SA_actual = self.estimation_r_SA_next_psi_SA(self.h_x, self.h_a, self.state_df, self.action_df, self.next_state_df, self.reward_df, self.p_basis)
        
        # Estimate the reward
        self.r_pi_est = self.estimation_reward(self.h_x, self.state_df, self.reward_df) #self.reward_df
        
    
    def calculate_VQ_w(self, w: int = 10, gamma: float = 0.7):
        """
        Calculate value function V and Q-values for a given time lag `w` and discount factor `gamma`.

        Parameters
        ----------
        w : int, optional
            The time lag used in the reward estimation. Default is 10.
        gamma : float, optional
            The discount factor for future rewards. Default is 0.7.

        Returns
        -------
        original_results_w : dict
            Dictionary containing results from theta estimation.
        df : DataFrame
            DataFrame of the formatted results of theta estimation.
        latex_table : str
            The LaTeX formatted table of the results.
        est_V : ndarray, shape (n_samples, 1)
            Estimated value function V.
        est_Q_actual : ndarray, shape (n_samples, 1)
            Estimated Q-values for the actual actions taken.
        """
        
        # Step 1: Calculate theta_w and retrieve results, estimated policies, and reward data
        self.original_results_w, df, latex_table, self.est_r_pi_sa_w_actual = self.calculate_theta_w(self.state_df, self.action_df, self.next_state_df, self.reward_df, w, gamma)
        
        # Extract the coefficient estimates (theta) from the results
        theta_hat = self.original_results_w['coef_estimate']
        
        # Step 2: Compute the estimated value function V using the basis functions and theta estimates
        est_V_hat = self.basis_df.values @ theta_hat.reshape(-1,1)
        est_V_tilde = self.r_pi_est[:, np.newaxis]+self.hat_psi_next @ theta_hat.reshape(-1,1) * gamma - self.r_pi_w_est[:, np.newaxis]*(gamma**w)
        
        # Step 3: Calculate Q-values for the actual actions taken
        # Using the reward data and next-state basis function estimates for the actual actions
        est_Q_actual = self.est_r_sa_actual[:, np.newaxis]+self.est_next_psi_SA_actual @ theta_hat.reshape(-1,1) * gamma - self.est_r_pi_sa_w_actual[:, np.newaxis]*(gamma**w)
           
        return self.original_results_w, df, latex_table, est_V_hat, est_V_tilde, est_Q_actual
    
    
    def calculate_VQ_inf(self, gamma: float = 0.7):
        """
        Calculate value function V and Q-values for the infinite horizon setting with a given discount factor `gamma`.

        Parameters
        ----------
        gamma : float, optional
            The discount factor for future rewards. Default is 0.7.

        Returns
        -------
        original_results_inf : dict
            Dictionary containing results from theta estimation.
        df : DataFrame
            DataFrame of the formatted results of theta estimation.
        latex_table : str
            The LaTeX formatted table of the results.
        est_V : ndarray, shape (n_samples, 1)
            Estimated value function V.
        est_Q_actual : ndarray, shape (n_samples, 1)
            Estimated Q-values for the actual actions taken.
        """
        
        # Step 1: Calculate theta_inf and retrieve results
        # This method calculates the parameter estimates (`theta_hat`) for the infinite horizon case
        self.original_results_inf, df, latex_table = self.calculate_theta_inf(gamma)
        
        # Extract the coefficient estimates (theta) from the results
        theta_hat = self.original_results_inf['coef_estimate']
        
        # Step 2: Compute the estimated value function V using the basis functions and theta estimates
        est_V_hat = self.basis_df.values @ theta_hat.reshape(-1,1)
        est_V_tilde = self.r_pi_est[:, np.newaxis]+self.hat_psi_next @ theta_hat.reshape(-1,1) * gamma
        
        
        # Step 3: Calculate Q-values for the actual actions taken
        # Using the reward data and next-state basis function estimates for the actual actions
        est_Q_actual = self.est_r_sa_actual[:, np.newaxis]+self.est_next_psi_SA_actual @ theta_hat.reshape(-1,1) * gamma
        
        return self.original_results_inf, df, latex_table, est_V_hat, est_V_tilde, est_Q_actual
        
    
    def calculate_theta_w(self, state_df, action_df, next_state_df, reward_df, w: int = 10, gamma: float = 0.7):
        """
        Calculate theta_w based on time lag `w` and discount factor `gamma`.

        Parameters
        ----------
        state_df : DataFrame
            DataFrame containing the scaled state variables at the current time step.
        action_df : DataFrame
            DataFrame containing the actions taken in the current time step.
        next_state_df : DataFrame
            DataFrame containing the scaled state variables at the next time step (used for future predictions).
        reward_df : DataFrame
            DataFrame containing the reward values associated with the actions taken in the current state.
        w : int, optional
            The time lag used in the reward estimation. Default value is 10.
        gamma : float, optional
            The discount factor for future rewards. A value between 0 and 1 that indicates how much future rewards are discounted 
            relative to immediate rewards. Default value is 0.7.

        Returns
        -------
        original_results : DataFrame
            DataFrame containing the results from the ordinary least squares model fit, including coefficients and statistics.
        df : DataFrame
            DataFrame of basis function estimates, adjusted for the specified time lag.
        latex_table : str
            String representation of the results formatted for LaTeX output, useful for academic reports.
        est_r_pi_sa_w_actual : DataFrame
            DataFrame of the actual reward estimates for each action, adjusted according to the time lag.
        """

        # Estimate reward with time lag `w`
        self.r_pi_w_est = self.estimation_reward_w(self.h_x, state_df, reward_df, w)
        
        # Compute the right-hand side based on estimated rewards
        R_pi = self.r_pi_w_est * (gamma ** w) - self.r_pi_est 

        # Calculate the left-hand matrix for basis function estimation
        self.zeta_w = gamma * self.hat_psi_next - self.basis_df.values
        
        # Fit an ordinary least squares model with HAC standard errors
        self.model = sm.OLS(R_pi[:, np.newaxis], self.zeta_w).fit(cov_type='HAC', cov_kwds={'maxlags': self.max_lag})
        
        
        # Organize results for output
        original_results, df, latex_table = self.organizing_result(self.model)
        
        # Initialize time-windowed reward estimator
        r_pi_sa_w_hat = est_r_pi_sa_w(self.h_x, self.h_a)  
        r_pi_sa_w_hat.fit(state_df.values, action_df.values, reward_df.values, w)
        
        # Estimate time-windowed reward  for the actions taken
        est_r_pi_sa_w_actual = r_pi_sa_w_hat(state_df.values, action_df.values)

        
        
        return original_results, df, latex_table, est_r_pi_sa_w_actual
    
    def calculate_theta_inf(self, gamma: float = 0.99):
        """
        Calculate theta in the infinite-horizon case using the discount factor `gamma`.

        Parameters
        ----------
        gamma : float, optional
            The discount factor for future rewards, determining their present value.

        Returns
        -------
        original_results : DataFrame
            DataFrame containing the results from the ordinary least squares model fit, including coefficients and statistics.
        basis_function_df : DataFrame
            DataFrame of basis function estimates used in the model.
        latex_table : str
            String representation of the results formatted for LaTeX output, useful for academic reports.
        """

        # Compute the right-hand side based on estimated rewards
        R_pi = - self.r_pi_est 

        # Calculate the left-hand matrix for basis function estimation
        self.zeta_inf = gamma * self.hat_psi_next - self.basis_df.values
        
        # Fit an ordinary least squares model with HAC standard errors
        self.model = sm.OLS(R_pi[:, np.newaxis], self.zeta_inf).fit(cov_type='HAC', cov_kwds={'maxlags': self.max_lag})
        
        # Organize results for output
        original_results, df, latex_table = self.organizing_result(self.model) 
        
        
        
        return original_results, df, latex_table
        
    
    def extract_est_actual(self, est_pi_data_all, action_df, unique_actions):
        """
        Extract actual policy estimates corresponding to each action.
        
        Parameters
        ----------
        est_pi_data_all : array-like, shape (n_samples, n_actions)
            Estimated policy data for all actions.
        action_df : DataFrame
            The action data corresponding to each sample.
        unique_actions : array-like
            Unique action values.
        
        Returns
        -------
        array-like, shape (n_samples,)
            The actual policy estimates corresponding to the action taken.
        """
        
        est_pi_actual = np.zeros(est_pi_data_all.shape[0])
        
        # For each action, apply a mask to extract the actual policy estimates
        for index, action in enumerate(unique_actions):
            mask = (action_df.squeeze() == action)  # Mask for each action
            est_pi_actual[mask] = est_pi_data_all[mask, index]
            
        return est_pi_actual
    
    def extract_next_psi_SA_actual(self, next_psi_SA_value_data, action_df, unique_actions):
        """
        Extract actual policy estimates corresponding to each action.

        Parameters
        ----------
        est_pi_data_all : array-like, shape (n_samples, n_actions)
            Estimated policy data for all actions.
        action_df : DataFrame
            The action data corresponding to each sample.
        unique_actions : array-like
            Unique action values.

        Returns
        -------
        array-like, shape (n_samples,)
            The actual policy estimates corresponding to the action taken.
        """

        # Map action_df to the indices of unique_actions
        action_indices = np.searchsorted(unique_actions, action_df.squeeze())

        # Initialize the result array
        psi_next_actual = np.zeros_like(next_psi_SA_value_data[0])

        # Assign the appropriate rows from next_q_value_data to q_value_next_actual
        for i, action_index in enumerate(action_indices):
            psi_next_actual[i, :] = next_psi_SA_value_data[action_index][i, :]


        return psi_next_actual
    
    def estimation_r_SA_next_psi_SA(self, h_x, h_a, state_df, action_df, next_state_df, reward_df, basis):
        """
        Estimate the expected rewards and expected next-state basis functions for each state-action pair 
        using kernel density estimation.

        Parameters
        ----------
        h_x : float
            The bandwidth parameter for kernel density estimation, controlling the smoothness of the estimates.
        h_a : float
            The bandwidth for kernel density estimation, influencing the smoothness of the kernel function for action variables.
        state_df : DataFrame
            DataFrame of scaled state data representing current states in the estimation process.
        action_df : DataFrame
            DataFrame of actions taken in each state, corresponding to each state in `state_df`.
        next_state_df : DataFrame
            DataFrame of scaled next-state data, used to estimate the expected value of the next state.
        reward_df : DataFrame
            DataFrame of rewards corresponding to each action taken in each state.
        basis : object
            Basis functions object for the next-state estimation, providing the basis function definitions.

        Returns
        -------
        est_r_sa_actual : ndarray, shape (n_samples,)
            Estimated expected rewards corresponding to each state-action pair in the dataset.
        est_next_psi_SA_actual : ndarray
            Estimated next-state basis function values for each state-action pair in the dataset.
        """

        # Instantiate the reward estimator and fit it to the state-action-reward data
        r_sa_hat = est_r_sa(h_x, h_a)
        r_sa_hat.fit(state_df.values, action_df.values, reward_df.values)
        
        # Estimate expected rewards for the actions taken
        est_r_sa_actual = r_sa_hat(state_df.values, action_df.values) 
        
        # Instantiate the next-state basis expectation estimator and fit it to the data
        next_psi_SA = BasisNextSAExpect(h_x, h_a)
        next_psi_SA.fit(state_df.values, action_df.values, next_state_df.values, basis)
        
        # Estimate the next-state basis values for the actions taken
        est_next_psi_SA_actual = next_psi_SA(state_df.values, action_df.values)
        
        return est_r_sa_actual, est_next_psi_SA_actual

    def estimation_reward(self, h_x, state_df, reward_df):
        """
        Estimate the expected reward for each state using kernel density estimation.

        Parameters
        ----------
        h_x : float
            The bandwidth parameter for kernel density estimation, controlling the smoothness of the estimated rewards.
        state_df : DataFrame
            DataFrame containing scaled state data used for estimating rewards.
        reward_df : DataFrame
            DataFrame containing the reward values associated with each state.

        Returns
        -------
        ndarray
            Estimated reward values for each state, derived from the kernel density estimation.
        """

        # Initialize the reward estimator with specified bandwidth and regularization
        r_pi_hat = est_r_pi(h_x) # Instantiate the reward estimator class
        r_pi_hat.fit(state_df.values, reward_df.values) # Fit the estimator to the state and reward data
        
        # Estimate the expected rewards for each state based on the fitted model
        r_pi_est = r_pi_hat(state_df.values) # Generate the estimated rewards

        return r_pi_est

    def estimation_reward_w(self, h_x, state_df, reward_df, w):
        """
        Estimate the reward over a specified time window `w` using kernel density estimation.

        Parameters
        ----------
        h_x : float
            The bandwidth parameter for kernel density estimation, influencing the smoothness of the estimated rewards.
        state_df : DataFrame
            Scaled state data used for estimating rewards over the time window.
        reward_df : DataFrame
            DataFrame containing the reward values associated with each state.
        w : int
            The time window (in time steps) over which to estimate the reward.

        Returns
        -------
        ndarray
            Estimated reward values over the specified time window.
        """

        # Initialize the time-windowed reward estimator with bandwidth and regularization
        r_pi_w_hat = est_r_pi_w(h_x) # Instantiate the time-windowed reward estimator
        r_pi_w_hat.fit(state_df.values, reward_df.values, w) # Fit the estimator to the data over the specified window
        
        # Estimate the rewards over the time window `w` using the fitted model
        r_pi_w_est = r_pi_w_hat(state_df.values)

        return r_pi_w_est
    
    

    def compute_basis(self, h_x, state_df, next_state_df, basis):
        """
        Compute the Chebyshev basis functions for the current state and estimate their 
        conditional expectations for the next state using kernel regression.

        Parameters
        ----------
        h_x : float
            The bandwidth parameter for kernel density estimation, controlling the width of the kernel.
        state_df : DataFrame
            DataFrame containing the scaled current state data.
        next_state_df : DataFrame
            DataFrame containing the scaled next state data.
        basis : callable
            A function or object that computes the Chebyshev basis functions given state data.

        Returns
        -------
        basis_df : DataFrame
            DataFrame containing the computed basis functions for the current state.
        basis_dict : dict
            Dictionary of basis function definitions, including the orders and basis terms.
        hat_psi_next : ndarray
            Array containing the estimated conditional expectations of the next state's basis functions.
        """
        # Compute the Chebyshev basis functions for the current state data
        basis_df, basis_dict = basis(state_df.values)  # Generate basis function values and metadata
        
        # Instantiate the next-state expectation estimator with specified bandwidth and regularization
        next_psi = BasisNextExpect(h_x)  # Instantiate the next-state basis expectation estimator
        next_psi.fit(state_df.values, next_state_df.values, basis)

        # Estimate expectations for the next state
        hat_psi_next = next_psi(state_df.values)

        return basis_df, basis_dict, hat_psi_next
    
    
    def significance_stars(self, p_value):
        """
        Assign significance stars based on p-value.
        
        Parameters
        ----------
        p_value : float
            The p-value for statistical significance.
            
        Returns
        -------
        str
            Stars representing the level of significance ('***' for p < 0.01, '**' for p < 0.05, '*' for p < 0.1).
        """
        if p_value < 0.01:
            return "***"  # Very significant
        elif p_value < 0.05:
            return "**"   # Significant
        elif p_value < 0.1:
            return "*"    # Marginally significant
        else:
            return ""      # Not significant
        
    
    def scientific_to_latex(self, value):
        """
        Convert a numerical value to a LaTeX string in scientific notation.

        Parameters
        ----------
        value : float
            The value to convert.

        Returns
        -------
        str
            The LaTeX formatted string.
        """
        if value == 0:
            return "$0$"
        else:
            exp = int(np.floor(np.log10(np.abs(value))))

            # Return the base directly if the exponent is 0
            if exp == 0:
                return f"${value:.3f}$"
            
            base = value / 10**exp

            return f"${base:.3f} \\times 10^{{{exp}}}$"
        
    # Function to convert a string to a tuple of floats
    def convert_string_to_tuple(self, s):
        return ast.literal_eval(s)

    # Function to convert a tuple to LaTeX formatted strings
    def format_tuple_to_latex(self, values):
        latex_values = [self.scientific_to_latex(float(value)) for value in values]  # Use self.scientific_to_latex to format each value
        return f"({', '.join(latex_values)})"  # Return formatted tuple as a string
    
    def combine_pval_stars(self, row, pval_col, stars_col):
        """
        Combine p-value and significance stars into a single string.
        
        Parameters
        ----------
        row : pandas.Series
            The row of the DataFrame containing the p-value and stars.
        pval_col : str
            The name of the p-value column.
        stars_col : str
            The name of the stars column.
        
        Returns
        -------
        str
            Combined string of p-value and significance stars.
        """
        return f"{row[pval_col]} {row[stars_col]}"
    
    def organizing_result(self, model):
        """
        Organize the results from a fitted regression model into a structured format.

        Parameters
        ----------
        model : statsmodels regression model
            A fitted regression model from which to extract results.

        Returns
        -------
        original_results : dict
            A dictionary containing the original numeric results, including coefficient estimates,
            standard errors, t-values, p-values, confidence intervals, covariance matrix,
            and significance stars.
        df : DataFrame
            A DataFrame of formatted results suitable for display, including coefficients, 
            HAC standard errors, t-values, p-values, confidence intervals, and significance stars.
        latex_table : str
            A LaTeX formatted string representation of the results DataFrame for inclusion in LaTeX documents.
        """

        # Extract various statistical measures from the model
        coef_estimate = model.params # Coefficient estimate
        t_value = model.tvalues  # Extract t-statistic
        p_value = model.pvalues  # Extract p-value
        hac_std_error = model.bse  # Extract HAC standard error
        conf_interval = model.conf_int() # Extract CI
        cov_theta_hat = model.cov_HC0 # Extract HAC covariance matrix

        stars = [self.significance_stars(p) for p in p_value]

        # Format the results into a tuple for display
        formatted_results = [
            (
                f"{coef_estimate[i]:.4g}",
                f"{hac_std_error[i]:.4g}",
                f"{t_value[i]:.4g}",
                f"{p_value[i]:.4g}",
                f"({conf_interval[i, 0]:.4g}, {conf_interval[i, 1]:.4g})",
                stars[i]
            )
            for i in range(len(coef_estimate))
        ]

        # Store original numeric results in a dictionary
        original_results = {
            "coef_estimate": coef_estimate,
            "hac_std_error": hac_std_error,
            "t_value": t_value,
            "p_value": p_value,
            "95%_confidence_interval": conf_interval,
            "hac_cov_matrix": cov_theta_hat,
            "stars": stars
        }

        # Create a list to store the formatted strings
        formatted_list = []

        # Generate the corresponding list of formatted strings for each basis
        for key, value in self.basis_dict.items():
            formatted_string = f"$\\hat{{\\theta}}_{{{value[0]}, {value[1]}}}$"
            formatted_list.append(formatted_string)

        # Create DataFrame of results
        column_names = ["Coef", "HAC SE", "t-val", "p-val", "95\% CI", 'stars']
        df = pd.DataFrame(formatted_results, columns=column_names, index=formatted_list)

        # Convert numeriself columns to LaTeX scientific notation
        for col in ["Coef", "HAC SE", "t-val", "p-val"]:
            df[col] = df[col].astype(float).apply(self.scientific_to_latex)

        # Convert strings to tuples
        df["95\% CI"] = df["95\% CI"].apply(self.convert_string_to_tuple)
        # Apply the formatting function to the "95% CI" column
        df["95\% CI"] = df["95\% CI"].apply(self.format_tuple_to_latex)

        # Combine p-val (2-sided) and stars into a single column
        df['p-val'] = df.apply(lambda row: self.combine_pval_stars(row, 'p-val', 'stars'), axis=1)

        # Drop the original stars column as it's no longer needed
        df = df.drop(columns=['stars'])

        # Create LaTeX table format
        latex_table = df.to_latex(index=True, escape=False).replace(r'\toprule', r'\hline').replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline')


        return original_results, df, latex_table
    
    
    
class WilcoxonSignedRankTest:
    def __init__(self, est_Q_actual, est_V, date_df):
        """
        Initialize the WilcoxonSignedRankTest class for causal inference testing in time series.

        This class implements the Wilcoxon signed-rank test framework to assess the causal effects
        of taking actual actions versus not taking them, using state-action quality functions.

        Parameters
        ----------
        est_Q_actual : array-like
            Estimated values of the actual state-action quality function, reflecting the 
            outcomes of actions taken in the time series.
        est_V : array-like
            Estimated values of the state value function, which follows the same policy as 
            est_Q_actual, representing the expected outcomes of states under the current policy.
        """
        # Store the actual estimated quality function values
        self.est_Q_actual = est_Q_actual
        
        # Store the estimated state value function values
        self.est_V = est_V
        
        # Use highlight_indices to identify positions where vaccination  bagins bagins after
        self.highlight_indices = date_df.values.flatten() >= '2021-04-12'

    def __call__(self, displaying = True, displaying_latex = False):
        """
        Execute the Wilcoxon signed-rank test by computing tau, performing statistical tests,
        and printing results, including t-statistics, p-values, and confidence intervals.
        This testing incorporates HAC standard errors for robust inference.
        """
        # Compute the advantage values (difference between actual Q values and estimated V values)
        self.Advantage = self.est_Q_actual- self.est_V
        
        # Compute the test statistics for the Wilcoxon signed-rank test
        self.stat_lower, self.p_value_lower = self.compute_statistics(self.Advantage)
        self.stat_lower_1, self.p_value_lower_1 = self.compute_statistics(self.Advantage[self.highlight_indices == False])
        self.stat_lower_0, self.p_value_lower_0 = self.compute_statistics(self.Advantage[self.highlight_indices])
        
        # Print statistics and p-values for lower-tailed test
        print(f"\nWilcoxon-statistic (lower-tailed test): {self.stat_lower}")
        print(f"p-value (lower-tailed test): {self.p_value_lower}")
        print(f"Observation periods: {len(self.Advantage)}")
        
        # Print statistics and p-values for lower-tailed tests
        print(f"\nWilcoxon statistic (lower-tailed) for vaccination starting before: {self.stat_lower_1}")
        print(f"p-value: {self.p_value_lower_1}")
        print(f"Observation periods (before): {len(self.Advantage[~self.highlight_indices])}")

        print(f"\nWilcoxon statistic (lower-tailed) for vaccination starting after: {self.stat_lower_0}")
        print(f"p-value: {self.p_value_lower_0}")
        print(f"Observation periods (after): {len(self.Advantage[self.highlight_indices])}")

         
        # Assign significance stars based on p-values
        #stars_two_sided = self.significance_stars(self.p_value)
        #stars_upper = self.significance_stars(self.p_value_upper)
        stars_lower = self.significance_stars(self.p_value_lower)
        stars_lower_1 = self.significance_stars(self.p_value_lower_1)
        stars_lower_0 = self.significance_stars(self.p_value_lower_0)
        
        # Significance level
        alpha = 0.05

        # Two-tailed Wilcoxon test for the hypothesis of median different from 0

        print("Lower-tailed test (whether the median is less than 0):")
        if self.p_value_lower < alpha:
            print(">> Reject the null hypothesis. The sample median is considered to be less than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")
            
        print("Lower-tailed test (whether the median is less than 0) for vaccination starting before:")
        if self.p_value_lower_1 < alpha:
            print(">> Reject the null hypothesis. The sample median is considered to be less than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")
            
        print("Lower-tailed test (whether the median is less than 0) for vaccination starting after:")
        if self.p_value_lower_0 < alpha:
            print(">> Reject the null hypothesis. The sample median is considered to be less than 0.\n")
        else:
            print(">> Fail to reject the null hypothesis.\n")
        
        # Plot histogram of the advantage values
        #percentile_lower_bound = np.percentile(self.Advantage, 2.5)  # Lower 2.5 percentile
        #percentile_upper_bound = np.percentile(self.Advantage, 97.5)  # Upper 97.5 percentile

        # Filter data within the specified percentiles for the histogram
        #filtered_data = self.Advantage[(self.Advantage >= percentile_lower_bound) & (self.Advantage <= percentile_upper_bound)]
        
        
        if displaying:
            
            fig, (ax1, ax1_1, ax1_0) = plt.subplots(1, 3, figsize=(24, 6))  # 1行3列のサブプロット作成

            # Plot the first histogram
            ax1.hist(self.Advantage, bins=100, edgecolor='k')
            ax1.set_xlabel(r'$A_{\pi}(S_{t}, A_{t};w, \hat{\theta})$', fontsize=30)
            ax1.set_ylabel('Frequency', fontsize=30)
            ax1.tick_params(axis='both', labelsize=24)

            # Plot the second histogram (before)
            ax1_1.hist(self.Advantage[~self.highlight_indices], bins=100, edgecolor='k')
            ax1_1.set_xlabel(r'$A_{\pi}(S_{t}, A_{t};w, \hat{\theta})$ (before)', fontsize=30)
            ax1_1.set_ylabel('Frequency', fontsize=30)
            ax1_1.tick_params(axis='both', labelsize=24)

            # Plot the third histogram (after)
            ax1_0.hist(self.Advantage[self.highlight_indices], bins=100, edgecolor='k')
            ax1_0.set_xlabel(r'$A_{\pi}(S_{t}, A_{t};w, \hat{\theta})$ (after)', fontsize=30)
            ax1_0.set_ylabel('Frequency', fontsize=30)
            ax1_0.tick_params(axis='both', labelsize=24)

            # Adjust layout and show the plots
            plt.tight_layout()
            plt.show()
            
            # Save each plot to a BytesIO buffer
            buffers = []
            for i, ax in enumerate([ax1, ax1_1, ax1_0]):
                buf = io.BytesIO()
                fig.savefig(buf, format='eps', transparent=False, bbox_inches='tight')
                buf.seek(0)
                buffers.append(buf)
                print(f"Buffer size for plot {i}: {buf.getbuffer().nbytes} bytes")

            # Optional: Close the figure to free memory
            #plt.close(fig)
            
            
        
        else: buffers = None

        
        # Format the results into a tuple for display
        formatted_results = (
            f"{self.stat_lower:.4g}",
            f"{self.p_value_lower:.4g}",
            stars_lower,
            f"{self.stat_lower_1:.4g}",
            f"{self.p_value_lower_1:.4g}",
            stars_lower_1,
            f"{self.stat_lower_0:.4g}",
            f"{self.p_value_lower_0:.4g}",
            stars_lower_0
        )
        
        
        # Store original numeric results in a dictionary
        original_results = {
            "W_all": self.stat_lower,
            "p_value_lower": self.p_value_lower,
            "stars_lower": stars_lower,
            "W_1": self.stat_lower_1,
            "p_value_lower_1": self.p_value_lower_1,
            "stars_lower_1": stars_lower_1,
            "W_0": self.stat_lower_0,
            "p_value_lower_0": self.p_value_lower_0,
            "stars_lower_0": stars_lower_0
        }
        
        # Create a LaTeX table from the formatted results
        df = self.create_latex_table(formatted_results) 
        # Create LaTeX table format
        latex_table = df.to_latex(index=True, escape=False).replace(r'\toprule', r'\hline').replace(r'\midrule', r'\hline').replace(r'\bottomrule', r'\hline')
        
        # Print the LaTeX formatted table if desired
        if displaying_latex:
            print("LaTeX formatted table:")
            print(latex_table)
        
        return original_results, latex_table, buffers
    
    def significance_stars(self, p_value):
        """
        Assign significance stars based on p-value.
        
        Parameters
        ----------
        p_value : float
            The p-value for statistical significance.
            
        Returns
        -------
        str
            Stars representing the level of significance ('***' for p < 0.01, '**' for p < 0.05, '*' for p < 0.1).
        """
        if p_value < 0.01:
            return "***"
        elif p_value < 0.05:
            return "**"
        elif p_value < 0.1:
            return "*"
        else:
            return ""
    
    def compute_statistics(self, Advantage):
        """
        Compute statistics related to tau including standard error, t-statistic, and p-values.

        Parameters
        ----------
        Advantage : array-like
            Computed advantage values.

        Returns
        -------
        tuple
            Stat, p-value, upper stat, upper p-value, lower stat, lower p-value.
        """
        # Perform the Wilcoxon signed-rank test for two-sided, upper, and lower alternatives
        #stat, p_value = wilcoxon(Advantage, alternative='two-sided')
        #_, p_value_upper = wilcoxon(Advantage, alternative='greater')
        stat_lower, p_value_lower = wilcoxon(Advantage, alternative='less')
        
        
        return stat_lower[0], p_value_lower[0]
    
    def scientific_to_latex(self, value):
        """
        Convert a numerical value to a LaTeX string in scientific notation.

        Parameters
        ----------
        value : float
            The value to convert.

        Returns
        -------
        str
            The LaTeX formatted string.
        """
        if value == 0:
            return "$0$"
        else:
            exp = int(np.floor(np.log10(np.abs(value))))

            # Return the base directly if the exponent is 0
            if exp == 0:
                return f"${value:.3f}$"
            
            base = value / 10**exp

            return f"${base:.3f} \\times 10^{{{exp}}}$"
        
   # Function to convert a string to a tuple of floats
    def convert_string_to_tuple(self, s):
        return ast.literal_eval(s)

    # Function to convert a tuple to LaTeX formatted strings
    def format_tuple_to_latex(self, values):
        latex_values = [self.scientific_to_latex(float(value)) for value in values]
        return f"({', '.join(latex_values)})"

    
    def combine_pval_stars(self, row, pval_col, stars_col):
        return f"{row[pval_col]} {row[stars_col]}"


    def create_latex_table(self, formatted_results):
        """
        Create a DataFrame from formatted results to generate a LaTeX table.

        Parameters
        ----------
        formatted_results : tuple
            Tuple containing formatted statistical results.

        Returns
        -------
        DataFrame
            DataFrame formatted for LaTeX output.
        """
        # Convert results to a NumPy array
        results_array = np.array(formatted_results)
        # Create DataFrame of results
        index_names = ['$W$', '$p$-value', 'stars', '$W$ (before)', '$p$-value (before)', 'stars (before)', '$W$ (after)', '$p$-value (after)', 'stars (after)']
        df = pd.DataFrame(results_array, index=index_names)
        df = df.T
        
        # Convert numerical columns to LaTeX scientific notation
        for col in ['$W$', '$p$-value', '$W$ (before)', '$p$-value (before)', '$W$ (after)', '$p$-value (after)']:
            df[col] = df[col].astype(float).apply(self.scientific_to_latex)
            
        # Combine p-values and stars for various significance tests
        df['$p$-value'] = df.apply(lambda row: self.combine_pval_stars(row, '$p$-value', 'stars'), axis=1)
        df['$p$-value (before)'] = df.apply(lambda row: self.combine_pval_stars(row, '$p$-value (before)', 'stars (before)'), axis=1)
        df['$p$-value (after)'] = df.apply(lambda row: self.combine_pval_stars(row, '$p$-value (after)', 'stars (after)'), axis=1)

        # Drop the individual stars columns as they are no longer needed
        df = df.drop(columns=['stars', 'stars (before)', 'stars (after)'])

        return df