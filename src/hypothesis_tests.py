"""
Hypothesis Testing Module for Insurance Risk Analytics

This module provides reusable functions for A/B hypothesis testing on insurance data.
It supports testing for differences in Claim Frequency, Claim Severity, and Margin
across different segments (provinces, zip codes, gender, etc.).
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


def calculate_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate key performance indicators for each policy.
    
    KPIs:
    - Claim Frequency: Binary indicator (1 if claim > 0, else 0)
    - Claim Severity: Average claim amount (only for policies with claims)
    - Margin: TotalPremium - TotalClaims
    
    Args:
        df: DataFrame with insurance data
        
    Returns:
        DataFrame with additional KPI columns
    """
    df = df.copy()
    
    # Claim Frequency: Binary indicator for whether a claim occurred
    df['claim_frequency'] = (df['TotalClaims'] > 0).astype(int)
    
    # Claim Severity: Claim amount (only meaningful when claim > 0)
    df['claim_severity'] = df['TotalClaims']
    
    # Margin: Premium - Claims
    df['margin'] = df['TotalPremium'] - df['TotalClaims']
    
    return df


def segment_data(
    df: pd.DataFrame,
    group_column: str,
    control_value: str,
    test_value: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Segment data into control (Group A) and test (Group B) groups.
    
    Args:
        df: DataFrame with insurance data
        group_column: Column name to segment on (e.g., 'Province', 'Gender')
        control_value: Value for control group (Group A)
        test_value: Value for test group (Group B)
        
    Returns:
        Tuple of (control_df, test_df)
    """
    control_df = df[df[group_column] == control_value].copy()
    test_df = df[df[group_column] == test_value].copy()
    
    return control_df, test_df


def chi_square_test(
    control_group: pd.Series,
    test_group: pd.Series
) -> Tuple[float, str, Dict]:
    """
    Perform chi-squared test for categorical data (e.g., claim frequency).
    
    Args:
        control_group: Binary/categorical data for control group
        test_group: Binary/categorical data for test group
        
    Returns:
        Tuple of (p_value, test_name, test_stats)
    """
    # Create contingency table
    control_counts = control_group.value_counts()
    test_counts = test_group.value_counts()
    
    # Ensure both groups have the same categories
    all_categories = set(control_counts.index) | set(test_counts.index)
    for cat in all_categories:
        if cat not in control_counts:
            control_counts[cat] = 0
        if cat not in test_counts:
            test_counts[cat] = 0
    
    # Create contingency table
    contingency_table = pd.DataFrame({
        'Control': control_counts.sort_index(),
        'Test': test_counts.sort_index()
    })
    
    # Perform chi-squared test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    test_stats = {
        'chi2_statistic': chi2,
        'degrees_of_freedom': dof,
        'contingency_table': contingency_table,
        'expected_frequencies': expected
    }
    
    return p_value, 'Chi-squared', test_stats


def t_test(
    control_group: pd.Series,
    test_group: pd.Series,
    equal_var: bool = False
) -> Tuple[float, str, Dict]:
    """
    Perform independent t-test for numerical data (e.g., claim severity, margin).
    
    Args:
        control_group: Numerical data for control group
        test_group: Numerical data for test group
        equal_var: Whether to assume equal variances (Welch's t-test if False)
        
    Returns:
        Tuple of (p_value, test_name, test_stats)
    """
    # Remove NaN values
    control_clean = control_group.dropna()
    test_clean = test_group.dropna()
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(control_clean, test_clean, equal_var=equal_var)
    
    test_stats = {
        't_statistic': t_stat,
        'control_mean': control_clean.mean(),
        'test_mean': test_clean.mean(),
        'control_std': control_clean.std(),
        'test_std': test_clean.std(),
        'control_n': len(control_clean),
        'test_n': len(test_clean)
    }
    
    test_name = "Welch's t-test" if not equal_var else "Student's t-test"
    
    return p_value, test_name, test_stats


def z_test_proportions(
    control_group: pd.Series,
    test_group: pd.Series
) -> Tuple[float, str, Dict]:
    """
    Perform z-test for comparing proportions (e.g., claim frequency rates).
    
    Args:
        control_group: Binary data for control group
        test_group: Binary data for test group
        
    Returns:
        Tuple of (p_value, test_name, test_stats)
    """
    # Calculate proportions
    control_success = (control_group == 1).sum()
    control_n = len(control_group)
    control_prop = control_success / control_n
    
    test_success = (test_group == 1).sum()
    test_n = len(test_group)
    test_prop = test_success / test_n
    
    # Pooled proportion
    pooled_prop = (control_success + test_success) / (control_n + test_n)
    
    # Standard error
    se = np.sqrt(pooled_prop * (1 - pooled_prop) * (1/control_n + 1/test_n))
    
    # Z-statistic
    z_stat = (test_prop - control_prop) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    test_stats = {
        'z_statistic': z_stat,
        'control_proportion': control_prop,
        'test_proportion': test_prop,
        'control_success': control_success,
        'test_success': test_success,
        'control_n': control_n,
        'test_n': test_n,
        'difference': test_prop - control_prop
    }
    
    return p_value, 'Z-test for proportions', test_stats


def run_hypothesis_test(
    control_df: pd.DataFrame,
    test_df: pd.DataFrame,
    kpi_column: str,
    test_type: str = 'auto'
) -> Dict:
    """
    Run appropriate hypothesis test based on KPI type.
    
    Args:
        control_df: Control group DataFrame
        test_df: Test group DataFrame
        kpi_column: Column name of the KPI to test
        test_type: 'auto', 'chi_square', 't_test', or 'z_test'
        
    Returns:
        Dictionary with test results
    """
    control_data = control_df[kpi_column]
    test_data = test_df[kpi_column]
    
    # Auto-detect test type
    if test_type == 'auto':
        # Check if data is binary (claim frequency)
        unique_vals = set(control_data.unique()) | set(test_data.unique())
        if unique_vals <= {0, 1}:
            test_type = 'z_test'
        else:
            test_type = 't_test'
    
    # Run appropriate test
    if test_type == 'chi_square':
        p_value, test_name, test_stats = chi_square_test(control_data, test_data)
    elif test_type == 't_test':
        p_value, test_name, test_stats = t_test(control_data, test_data)
    elif test_type == 'z_test':
        p_value, test_name, test_stats = z_test_proportions(control_data, test_data)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    # Determine decision
    alpha = 0.05
    decision = 'Reject H₀' if p_value < alpha else 'Fail to reject H₀'
    
    results = {
        'p_value': p_value,
        'test_name': test_name,
        'decision': decision,
        'alpha': alpha,
        'test_stats': test_stats,
        'control_n': len(control_df),
        'test_n': len(test_df)
    }
    
    return results


def format_results(results: Dict) -> str:
    """
    Format hypothesis test results for display.
    
    Args:
        results: Dictionary from run_hypothesis_test
        
    Returns:
        Formatted string with results
    """
    output = []
    output.append(f"Test: {results['test_name']}")
    output.append(f"P-value: {results['p_value']:.6f}")
    output.append(f"Decision: {results['decision']} (α = {results['alpha']})")
    output.append(f"Control Group N: {results['control_n']}")
    output.append(f"Test Group N: {results['test_n']}")
    
    # Add test-specific statistics
    if 't_statistic' in results['test_stats']:
        stats = results['test_stats']
        output.append(f"T-statistic: {stats['t_statistic']:.4f}")
        output.append(f"Control Mean: {stats['control_mean']:.4f}")
        output.append(f"Test Mean: {stats['test_mean']:.4f}")
        output.append(f"Difference: {stats['test_mean'] - stats['control_mean']:.4f}")
    elif 'z_statistic' in results['test_stats']:
        stats = results['test_stats']
        output.append(f"Z-statistic: {stats['z_statistic']:.4f}")
        output.append(f"Control Proportion: {stats['control_proportion']:.4f}")
        output.append(f"Test Proportion: {stats['test_proportion']:.4f}")
        output.append(f"Difference: {stats['difference']:.4f}")
    elif 'chi2_statistic' in results['test_stats']:
        stats = results['test_stats']
        output.append(f"Chi2-statistic: {stats['chi2_statistic']:.4f}")
        output.append(f"Degrees of Freedom: {stats['degrees_of_freedom']}")
    
    return '\n'.join(output)


def check_statistical_equivalence(
    df: pd.DataFrame,
    group_column: str,
    control_value: str,
    test_value: str,
    covariates: List[str],
    alpha: float = 0.05
) -> Dict[str, bool]:
    """
    Check if two groups are statistically equivalent on covariates.
    
    This is used to ensure that when comparing groups (e.g., two provinces),
    they are similar on other attributes (client, vehicle, plan) so that
    observed differences can be attributed to the feature being tested.
    
    Args:
        df: DataFrame with insurance data
        group_column: Column name to segment on
        control_value: Value for control group
        test_value: Value for test group
        covariates: List of column names to check for equivalence
        alpha: Significance level for equivalence tests
        
    Returns:
        Dictionary mapping covariate names to equivalence status
    """
    control_df, test_df = segment_data(df, group_column, control_value, test_value)
    
    equivalence_results = {}
    
    for covariate in covariates:
        if covariate not in df.columns:
            continue
            
        # Check if covariate is categorical or numerical
        if df[covariate].dtype == 'object' or df[covariate].nunique() < 10:
            # Categorical: Use chi-squared test
            try:
                p_value, _, _ = chi_square_test(control_df[covariate], test_df[covariate])
                is_equivalent = p_value >= alpha  # Fail to reject H₀ means groups are similar
            except:
                is_equivalent = False
        else:
            # Numerical: Use t-test
            try:
                p_value, _, _ = t_test(control_df[covariate], test_df[covariate])
                is_equivalent = p_value >= alpha  # Fail to reject H₀ means groups are similar
            except:
                is_equivalent = False
        
        equivalence_results[covariate] = is_equivalent
    
    return equivalence_results


def generate_business_interpretation(
    hypothesis: str,
    results: Dict,
    feature_name: str,
    control_name: str,
    test_name: str,
    kpi_name: str,
    effect_size: Optional[float] = None
) -> str:
    """
    Generate a business-facing interpretation of hypothesis test results.
    
    Args:
        hypothesis: Description of the null hypothesis
        results: Dictionary from run_hypothesis_test
        feature_name: Name of the feature being tested (e.g., 'Province')
        control_name: Name of control group
        test_name: Name of test group
        kpi_name: Name of the KPI (e.g., 'Claim Frequency')
        effect_size: Optional effect size (e.g., percentage difference)
        
    Returns:
        Business interpretation string
    """
    if results['decision'] == 'Reject H₀':
        if effect_size is not None:
            interpretation = (
                f"We reject H₀ for {feature_name} (p < {results['alpha']}). "
                f"{test_name} exhibits a {effect_size:.1%} {kpi_name.lower()} "
                f"difference compared to {control_name}, suggesting that "
                f"{feature_name.lower()} is a significant risk driver. "
                f"A pricing adjustment based on {feature_name.lower()} may be warranted."
            )
        else:
            interpretation = (
                f"We reject H₀ for {feature_name} (p < {results['alpha']}). "
                f"There is a statistically significant difference in {kpi_name.lower()} "
                f"between {test_name} and {control_name}, suggesting that "
                f"{feature_name.lower()} is a significant risk driver."
            )
    else:
        interpretation = (
            f"We fail to reject H₀ for {feature_name} (p = {results['p_value']:.4f}). "
            f"There is no statistically significant difference in {kpi_name.lower()} "
            f"between {test_name} and {control_name}. "
            f"Current pricing for {feature_name.lower()} appears adequate."
        )
    
    return interpretation
