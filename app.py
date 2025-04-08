import streamlit as st
import pandas as pd
import numpy as np

# Define a flag to track if plotting is available
plotting_available = True

try:
    # Ensure matplotlib works in non-interactive mode
    import matplotlib
    matplotlib.use('Agg')  # Required for Streamlit
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plotting_available = False
    st.error("Matplotlib or Seaborn not available. Charts will not be displayed.")

import load_data

# Set page configuration
st.set_page_config(
    page_title="Backtest Comparison Tool",
    page_icon="📊",
    layout="wide"
)

def style_df(df, tolerance):
    """
    Apply styling to the comparison dataframe based on tolerance
    """
    # Create an empty dataframe of same shape as input
    styled = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Only process if we have both years
    if '2023' in df.columns and '2024' in df.columns:
        # Calculate percent difference
        pct_diff = (df['2024'] - df['2023']) / df['2023'].abs()
        
        # Style based on threshold
        for idx in df.index:
            diff = pct_diff.loc[idx]
            
            if diff > tolerance:
                styled.loc[idx, '2024'] = 'background-color: rgba(0, 128, 0, 0.2)'  # Green for improvement
            elif diff < -tolerance:
                styled.loc[idx, '2024'] = 'background-color: rgba(255, 0, 0, 0.2)'  # Red for degradation
    
    return styled

def format_df(df):
    """Format the dataframe with appropriate number formats"""
    if '2023' in df.columns and '2024' in df.columns:
        # Calculate percent difference if not already present
        if 'Pct Diff' not in df.columns:
            df['Pct Diff'] = (df['2024'] - df['2023']) / df['2023'].abs()
    
    # Format based on metric type
    formatted = df.copy()
    
    for col in formatted.columns:
        if col != 'Pct Diff':
            # Format numbers as percentages or decimals based on metric
            formatted[col] = formatted[col].apply(lambda x: 
                f"{x:.2%}" if isinstance(x, (int, float)) and abs(x) < 5 else 
                f"{x:.2f}" if isinstance(x, (int, float)) else x)
        else:
            # Format percent difference
            formatted[col] = formatted[col].apply(lambda x: 
                f"{x:.2%}" if isinstance(x, (int, float)) else x)
    
    return formatted

def get_experiment_filters():
    """
    Creates UI elements for filtering experiments and returns selected values
    """
    # Get available options
    experiments = load_data.get_available_experiments()
    
    # Create experiment selector
    selected_experiment = st.selectbox(
        "Select Experiment",
        options=experiments, 
        key="summary_experiment_selector"
    )
    
    # Get available universes and create selector
    universes = load_data.get_available_universes()
    selected_universe = st.selectbox(
        "Select Universe", 
        options=universes,
        key="summary_universe_selector"
    )
    
    # Get available products and create selector
    products = load_data.get_available_products()
    selected_product = st.selectbox(
        "Select Product", 
        options=products,
        key="summary_product_selector"
    )
    
    # Get and display frontier points
    frontier_points = load_data.get_frontier_points(selected_experiment)
    frontier_keys = []
    frontier_values = []
    
    if frontier_points:
        st.subheader("Frontier Points")
        for key, values in frontier_points.items():
            frontier_keys.append(key)
            selected_value = st.selectbox(
                f"Select {key}",
                options=values,
                key=f"frontier_{key}"
            )
            frontier_values.append(selected_value)
    
    return selected_experiment, selected_universe, selected_product, frontier_keys, frontier_values

def create_strategy_id(experiment, universe, product, frontier_keys, frontier_values):
    """Create a unique identifier for a strategy"""
    frontier_str = ""
    if frontier_keys and frontier_values:
        frontier_str = " - " + ", ".join([f"{k}={v}" for k, v in zip(frontier_keys, frontier_values)])
    
    return f"{experiment} - {universe} - {product}{frontier_str}"

def plot_comparison(df, title):
    """Create a bar chart comparing 2023 and 2024 metrics"""
    if not plotting_available:
        return None
        
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = df.index.tolist()
        x = np.arange(len(metrics))
        width = 0.35
        
        # Extract values, handle percentage strings if needed
        vals_2023 = df['2023'].values
        vals_2024 = df['2024'].values
        
        # Create bars
        bars1 = ax.bar(x - width/2, vals_2023, width, label='2023', color='#1f77b4')
        bars2 = ax.bar(x + width/2, vals_2024, width, label='2024', color='#ff7f0e')
        
        # Add labels and title
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                if abs(height) < 0.01:  # Very small values
                    format_str = '{:.3f}'
                elif abs(height) < 5:  # Medium values
                    format_str = '{:.2f}'
                else:  # Large values
                    format_str = '{:.1f}'
                    
                ax.annotate(format_str.format(height),
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -15),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation=0)
        
        autolabel(bars1)
        autolabel(bars2)
        
        fig.tight_layout()
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {e}")
        # Return a simple placeholder text instead
        return None

def create_comparison_heatmap(df):
    """Create a heatmap from the correlation/difference data"""
    if not plotting_available:
        return None
        
    try:
        # Create a new figure with adequate size
        plt.figure(figsize=(12, 10))
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Ensure the figure is cleared
        plt.clf()
        
        # Make sure data is numeric
        numeric_df = df.astype(float)
        
        # Generate the heatmap with simplified parameters
        heatmap = sns.heatmap(
            numeric_df, 
            annot=True, 
            cmap="RdBu_r", 
            center=0, 
            fmt=".2f", 
            ax=ax
        )
        
        # Set title and formatting
        ax.set_title("2024 vs 2023 Differences Across Configurations")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Ensure proper layout
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error creating heatmap: {e}")
        st.write("Detailed error information:", str(e))
        # Return data table as fallback
        return None

def main():
    st.title("Backtest Comparison Tool")
    st.markdown("Compare 2023 and 2024 backtest results across different configurations")
    
    # Sidebar for strategy selection
    with st.sidebar:
        st.header("Strategy Selection")
        
        # Select comparison mode
        comparison_mode = st.radio(
            "Select Comparison Mode",
            ["Compare 2023 & 2024", "Compare Any Items", "Correlation Table"]
        )
        
        # Display experiment filters and load data
        experiment, universe, product, frontier_keys, frontier_values = get_experiment_filters()
        
        st.header("Display Settings")
        
        # Set tolerance for difference highlighting
        tolerance = st.slider(
            "Difference Tolerance (%)", 
            min_value=0.0, 
            max_value=20.0, 
            value=5.0
        ) / 100  # Convert percentage to decimal
        
        # Store comparison selections in the session state
        if "comparison_strategies" not in st.session_state:
            st.session_state.comparison_strategies = []
        
        # Add current selection to comparison
        if st.button("Add Current Selection to Comparison"):
            strategy_id = create_strategy_id(experiment, universe, product, frontier_keys, frontier_values)
            strategy_data = {
                "id": strategy_id,
                "experiment": experiment,
                "universe": universe,
                "product": product,
                "frontier_keys": frontier_keys,
                "frontier_values": frontier_values
            }
            
            # Check if already in list
            if not any(s["id"] == strategy_id for s in st.session_state.comparison_strategies):
                st.session_state.comparison_strategies.append(strategy_data)
                st.success(f"Added: {strategy_id}")
            else:
                st.warning(f"Strategy already in comparison: {strategy_id}")
        
        # Display selected strategies
        if st.session_state.comparison_strategies:
            st.subheader("Selected Strategies")
            
            for i, strategy in enumerate(st.session_state.comparison_strategies):
                col1, col2 = st.columns([0.8, 0.2])
                
                with col1:
                    st.write(f"**{i + 1}. {strategy['id']}**")
                
                with col2:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.comparison_strategies.pop(i)
                        st.rerun()
            
            if st.button("Clear All Comparisons"):
                st.session_state.comparison_strategies = []
                st.rerun()
    
    # Main content area
    if comparison_mode == "Compare 2023 & 2024":
        st.header("2023 vs 2024 Comparison")
        
        # Fetch data for current selection
        df = load_data.generate_summary_data_tables_comparison(
            experiment=experiment,
            universe=universe,
            product=product,
            frontier_keys=frontier_keys,
            frontier_values=frontier_values
        )
        
        # Calculate percent differences
        df['Pct Diff'] = (df['2024'] - df['2023']) / df['2023'].abs()
        
        # Create two columns for visualization and data
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Show the comparison chart
            title = create_strategy_id(experiment, universe, product, frontier_keys, frontier_values)
            fig = plot_comparison(df, title)
            if fig is not None:
                st.pyplot(fig)
            else:
                st.write("Unable to display chart. Showing data table only.")
        
        with col2:
            # Show the styled data table
            st.subheader("Data Comparison")
            formatted_df = format_df(df)
            styled_df = formatted_df.style.apply(lambda x: style_df(df, tolerance), axis=None)
            st.dataframe(styled_df)
    
    elif comparison_mode == "Compare Any Items":
        st.header("Multi-Strategy Comparison")
        
        if not st.session_state.comparison_strategies:
            st.info("Add strategies from the sidebar to begin comparison")
        else:
            comparison_dfs = {}
            
            for strategy in st.session_state.comparison_strategies:
                # Get data for this strategy
                strategy_df = load_data.generate_summary_data_tables_comparison(
                    experiment=strategy['experiment'],
                    universe=strategy['universe'],
                    product=strategy['product'],
                    frontier_keys=strategy['frontier_keys'],
                    frontier_values=strategy['frontier_values']
                )
                
                # Use 2024 values for comparison
                comparison_dfs[strategy['id']] = strategy_df['2024']
            
            if comparison_dfs:
                # Create a comparison dataframe
                merged_df = pd.DataFrame(comparison_dfs)
                
                # Display the comparison table
                st.subheader("Strategy Performance Comparison (2024)")
                st.dataframe(merged_df.style.highlight_max(axis=1, color='rgba(0, 128, 0, 0.2)'))
                
                
                # Show metrics across strategies
                st.subheader("Performance Metrics Across Strategies")
                
                # Create a plot for each metric
                for metric in merged_df.index:
                    if plotting_available:
                        try:
                            fig, ax = plt.subplots(figsize=(12, 4))
                            values = merged_df.loc[metric]
                            ax.bar(values.index, values.values, color='skyblue')
                            ax.set_title(f"{metric} Comparison")
                            ax.set_xticklabels(values.index, rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error creating plot for {metric}: {e}")
                            st.write(f"{metric} values: {merged_df.loc[metric].to_dict()}")
                    else:
                        # If plotting is not available, just show the data
                        st.subheader(f"{metric} Comparison")
                        st.write(merged_df.loc[metric])
    
    elif comparison_mode == "Correlation Table":
        st.header("2024 vs 2023 Difference Heatmap")
        
        # Need to generate combinations for the correlation/difference table
        if not st.session_state.comparison_strategies:
            st.info(
                "Add various universe/product combinations from the sidebar to create a difference heatmap. "
                "The heatmap will show (2024-2023) differences across all combinations."
            )
        else:
            # Prepare universe-product combinations for the correlation data
            universe_product_combos = [
                (
                    strategy['universe'], 
                    strategy['product'], 
                    strategy['frontier_keys'], 
                    strategy['frontier_values']
                )
                for strategy in st.session_state.comparison_strategies
            ]
            
            # Generate correlation/difference data
            corr_df = load_data.generate_correlation_data(experiment, universe_product_combos)
            
            # Display the data table first (always works)
            st.subheader("Difference Data (2024 - 2023)")
            st.write("Values show the difference between 2024 and 2023 metrics (positive is better)")
            st.dataframe(corr_df.style.background_gradient(cmap="RdBu_r", axis=None))
            
            # Try to create and display heatmap
            try:
                if plotting_available:
                    st.subheader("Heatmap Visualization")
                    
                    # Simplified alternative approach for heatmap
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Keep only numeric columns
                    heatmap_data = corr_df.copy()
                    
                    # Basic heatmap (simplified for reliability)
                    ax = sns.heatmap(
                        heatmap_data, 
                        annot=True, 
                        fmt=".2f", 
                        cmap="RdBu_r",
                        center=0
                    )
                    
                    plt.title("2024 vs 2023 Differences")
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Show metrics per strategy as bar charts (alternative visualization)
                    st.subheader("Difference by Metric")
                    
                    # Plot each metric as a separate bar chart
                    for metric in corr_df.index:
                        try:
                            metric_data = corr_df.loc[metric]
                            
                            fig, ax = plt.subplots(figsize=(10, 4))
                            bars = ax.bar(metric_data.index, metric_data.values)
                            
                            # Color bars based on value
                            for i, bar in enumerate(bars):
                                bar.set_color('green' if metric_data.values[i] > 0 else 'red')
                                
                            ax.set_title(f"{metric}: 2024 vs 2023 Difference")
                            ax.set_ylabel("Difference")
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not display bar chart for {metric}")
            except Exception as e:
                st.error(f"Error creating visualization: {e}")
                st.write("Showing data table only as a fallback.")
            

if __name__ == "__main__":
    main()
