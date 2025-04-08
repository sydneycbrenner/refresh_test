import pandas as pd
import numpy as np

# Define indices for summary dataframes
dataframe_index = ['Realized IR', 'Cumulative Net Capital Gains',
                   'Total Pre-Tax Return - Net', 'Tracking Error',
                   'Sharpe Ratio', 'Max Drawdown', 'Turnover',
                   'Annual Volatility', 'Benchmark Return']

def generate_sample_data(experiment, universe, product, frontier_keys, frontier_values, year):
    """
    Generate sample data for backtests with realistic finance metrics
    """
    # Create base values that make sense for finance metrics
    base_values = {
        'Realized IR': np.random.uniform(0.8, 1.5),
        'Cumulative Net Capital Gains': np.random.uniform(0.15, 0.4),
        'Total Pre-Tax Return - Net': np.random.uniform(0.08, 0.25),
        'Tracking Error': np.random.uniform(0.01, 0.04),
        'Sharpe Ratio': np.random.uniform(0.9, 1.8),
        'Max Drawdown': np.random.uniform(-0.15, -0.05),
        'Turnover': np.random.uniform(0.2, 0.6),
        'Annual Volatility': np.random.uniform(0.1, 0.2),
        'Benchmark Return': np.random.uniform(0.05, 0.12)
    }
    
    # Adjust 2023 vs 2024 - make 2024 slightly better in most metrics
    if year == 2024:
        adjustments = {
            'Realized IR': 0.1,
            'Cumulative Net Capital Gains': 0.03,
            'Total Pre-Tax Return - Net': 0.02,
            'Tracking Error': -0.005,
            'Sharpe Ratio': 0.15,
            'Max Drawdown': 0.02,  # Less negative
            'Turnover': -0.05,
            'Annual Volatility': -0.01,
            'Benchmark Return': 0.01
        }
        
        for key in base_values:
            base_values[key] += adjustments.get(key, 0)
    
    # Small random variation based on experiment, universe, product and frontier points
    experiment_factor = 0.95 if experiment == "BIGs" else 1.05
    universe_factor = 0.98 if universe == "FR1" else 1.02
    product_factor = 1.03 if product == "RC" else 0.97
    
    # Apply factors
    for key in base_values:
        base_values[key] *= experiment_factor * universe_factor * product_factor
        
        # Apply frontier point adjustments (custom logic per metric)
        if frontier_keys and frontier_values:
            if 'Pct BIGs' in frontier_keys:
                idx = frontier_keys.index('Pct BIGs')
                pct_bigs = frontier_values[idx]
                # More BIGs tends to improve IR and returns but may increase turnover
                if key == 'Realized IR':
                    base_values[key] *= (1 + pct_bigs * 0.4)
                elif key == 'Total Pre-Tax Return - Net':
                    base_values[key] *= (1 + pct_bigs * 0.3)
                elif key == 'Turnover':
                    base_values[key] *= (1 + pct_bigs * 0.15)
            
            if 'Pct Donation' in frontier_keys:
                idx = frontier_keys.index('Pct Donation')
                pct_donation = frontier_values[idx]
                # Donations might decrease capital gains but improve tracking
                if key == 'Cumulative Net Capital Gains':
                    base_values[key] *= (1 - pct_donation * 0.5)
                elif key == 'Tracking Error':
                    base_values[key] *= (1 - pct_donation * 0.3)
    
    return pd.Series(base_values, index=dataframe_index)

def get_available_experiments():
    """Return list of available experiments"""
    return ['Official Book', 'BIGs', 'BIGs Plus Donations']

def get_available_universes():
    """Return list of available universes"""
    return ['FR3', 'FR1']

def get_available_products():
    """Return list of available products"""
    return ['RC', 'CLS']

def get_frontier_points(experiment):
    """Return frontier points for a given experiment"""
    frontier_points = {
        'BIGs': {'Pct BIGs': [0, 0.2, 0.4, 0.6]},
        'BIGs Plus Donations': {
            'Pct BIGs': [0, 0.2, 0.4, 0.6],
            'Pct Donation': [0.05, 0.1, 0.15, 0.2]
        },
        'Official Book': {}
    }
    return frontier_points.get(experiment, {})

def generate_summary_data_tables_comparison(experiment, universe, product, frontier_keys=None, frontier_values=None, years=None):
    """
    Generate sample summary data for comparison tables
    """
    if years is None:
        years = [2023, 2024]
    
    result = {}
    
    for year in years:
        data = generate_sample_data(
            experiment=experiment,
            universe=universe,
            product=product,
            frontier_keys=frontier_keys,
            frontier_values=frontier_values,
            year=year
        )
        result[f"{year}"] = data
    
    return pd.DataFrame(result)

def generate_correlation_data(experiment, universe_product_combos):
    """
    Generate correlation data across different combinations
    """
    metrics = dataframe_index
    columns = []
    
    # Create column names for each universe/product combo
    for univ, prod, frontier_keys, frontier_values in universe_product_combos:
        if frontier_keys and frontier_values:
            frontier_str = "_".join([f"{k}={v}" for k, v in zip(frontier_keys, frontier_values)])
            col_name = f"{univ}_{prod}_{frontier_str}"
        else:
            col_name = f"{univ}_{prod}"
        columns.append(col_name)
    
    # Generate a dataframe with differences between 2024 and 2023
    result = pd.DataFrame(index=metrics, columns=columns)
    
    for col_idx, (univ, prod, frontier_keys, frontier_values) in enumerate(universe_product_combos):
        df = generate_summary_data_tables_comparison(
            experiment=experiment,
            universe=univ,
            product=prod,
            frontier_keys=frontier_keys,
            frontier_values=frontier_values
        )
        
        # Calculate 2024-2023 differences
        diff = df['2024'] - df['2023']
        result.iloc[:, col_idx] = diff
    
    return result
