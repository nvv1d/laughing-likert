import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import base64
import tempfile
from io import StringIO, BytesIO
from datetime import datetime, timedelta
import uuid
import json
import hashlib
import time
from utils import (
    load_data, 
    identify_likert_columns, 
    detect_reverse_items, 
    reverse_code,
    check_sampling,
    cluster_items,
    determine_factors,
    extract_weights,
    cronbach_alpha,
    bootstrap_alpha,
    simulate_responses,
    create_network_graph,
    save_html_report,
    run_hybrid_analysis
)

# Page configuration
st.set_page_config(
    page_title="Likert Scale Pattern Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define our login credentials
DEFAULT_USERNAME = "Admin"
DEFAULT_PASSWORD = "101066"  # This would typically be stored as a hash
MAX_LOGIN_ATTEMPTS = 3
LOCKOUT_DURATION_MINUTES = 15

# Initialize session state variables for login system
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0
if 'lockout_until' not in st.session_state:
    st.session_state.lockout_until = None
if 'lockout_count' not in st.session_state:
    st.session_state.lockout_count = 0

# Minimal CSS for the login form
login_css = """
<style>
.login-page {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
}

.login-container {
    max-width: 300px;
    width: 100%;
    text-align: center;
}

.stButton > button {
    width: 100%;
    background-color: #4051B5 !important;
    color: white !important;
}

.login-footer {
    margin-top: 20px;
    font-size: 12px;
    color: #888;
}

.error-message {
    background-color: #FFEBEE;
    color: #D32F2F;
    padding: 10px;
    border-radius: 4px;
    margin-bottom: 15px;
    font-size: 14px;
}

.locked-message {
    background-color: #FFF8E1;
    color: #F57F17;
    padding: 10px;
    border-radius: 4px;
    margin-bottom: 15px;
    font-size: 14px;
}
</style>
"""

# Check if user is locked out
def is_locked_out():
    if st.session_state.lockout_until is not None:
        if datetime.now() < st.session_state.lockout_until:
            remaining = st.session_state.lockout_until - datetime.now()
            minutes = remaining.seconds // 60
            seconds = remaining.seconds % 60
            return True, f"{minutes} minutes and {seconds} seconds"
        else:
            # Reset lockout but keep count
            st.session_state.lockout_until = None
            st.session_state.login_attempts = 0
            return False, ""
    return False, ""

# Function to handle login
def process_login(username, password):
    # Check if user is locked out
    locked, time_remaining = is_locked_out()
    if locked:
        st.error(f"Account is temporarily locked. Please try again in {time_remaining}.")
        return
    
    # Verify credentials
    if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
        st.session_state.logged_in = True
        st.session_state.login_attempts = 0
        st.session_state.lockout_count = 0
        st.rerun()
    else:
        # Increment failed attempts
        st.session_state.login_attempts += 1
        attempts_left = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
        
        if attempts_left <= 0:
            # Lock the account
            st.session_state.lockout_count += 1
            # Increase lockout time based on number of past lockouts (15 min, 30 min, 60 min, etc.)
            lockout_duration = LOCKOUT_DURATION_MINUTES * (2 ** (st.session_state.lockout_count - 1))
            st.session_state.lockout_until = datetime.now() + timedelta(minutes=lockout_duration)
            st.error(f"Too many failed attempts. Your account is locked for {lockout_duration} minutes.")
        else:
            st.error(f"Invalid username or password. {attempts_left} attempts remaining.")

# Display login if not logged in
if not st.session_state.logged_in:
    # Display the login form with custom styling
    st.markdown(login_css, unsafe_allow_html=True)
    
    # Create a simple login container
    st.markdown("""
    <div class="login-page">
        <div class="login-container">
    """, unsafe_allow_html=True)
    
    locked, time_remaining = is_locked_out()
    if locked:
        st.markdown(f"""
        <div class="locked-message">
            Account is temporarily locked. Please try again in {time_remaining}.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Create the login form
        with st.form("login_form"):
            username = st.text_input("Username", value="Admin")
            password = st.text_input("Password", type="password")
            
            # Display error message if there were previous attempts
            if st.session_state.login_attempts > 0:
                attempts_left = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                st.markdown(f"""
                <div class="error-message">
                    Invalid username or password. {attempts_left} attempts remaining.
                </div>
                """, unsafe_allow_html=True)
            
            submit = st.form_submit_button("Log In")
            
            if submit:
                process_login(username, password)
    
    # Simple footer
    st.markdown("""
        <div class="login-footer">
            © 2025 Analysis Tool
        </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Don't show anything else if not logged in
    st.stop()

# Main Application (only shown if logged in)
st.title("Likert Scale Pattern Analysis")
st.markdown("""
This application helps you analyze Likert scale survey data, extract response patterns, 
and generate simulated data based on those patterns. Upload your data file to get started.
""")

# Sidebar for inputs and controls
with st.sidebar:
    # Add logout button at the top of the sidebar
    st.markdown("""
    <style>
    .logout-btn {
        position: absolute;
        top: 0.5rem;
        right: 1rem;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 0.3rem;
        color: #6c757d;
        text-decoration: none;
        font-size: 0.8rem;
        cursor: pointer;
    }
    .logout-btn:hover {
        background-color: #e9ecef;
        color: #343a40;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("Logout", key="logout_btn"):
        # Clear session state and log out
        st.session_state.logged_in = False
        st.session_state.login_attempts = 0
        st.rerun()
    
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    st.header("Analysis Settings")
    
    # Initialize analysis method in session state if not present (default to Hybrid)
    if 'analysis_method' not in st.session_state:
        st.session_state.analysis_method = "Hybrid (Python + R)"
    
    # Create the selectbox WITHOUT using the session state key to avoid the warning
    # Instead, we'll update session state manually when it changes
    analysis_method = st.selectbox(
        "Analysis Method",
        ["Python Only", "R Only", "Hybrid (Python + R)"],
        index=["Python Only", "R Only", "Hybrid (Python + R)"].index(st.session_state.analysis_method),
        key="analysis_method_selector",  # Use a different key
        help="Select which analysis engine to use"
    )
    
    # Update session state if the value changed
    if analysis_method != st.session_state.analysis_method:
        st.session_state.analysis_method = analysis_method
    
    # Display message when analysis method changes
    if st.session_state.get('_previous_analysis_method') != analysis_method:
        if 'clusters' in st.session_state or 'weights' in st.session_state:
            st.warning("Analysis method changed! Results will be recalculated with the new method.")
            # Clear previous analysis results but keep data
            for key in ['clusters', 'weights', 'alphas', 'alpha_ci', 'sim_data', 'n_factors_detected']:
                if key in st.session_state:
                    del st.session_state[key]
        st.session_state['_previous_analysis_method'] = analysis_method
    
    if uploaded_file is not None:
        # Advanced settings (collapsed by default)
        with st.expander("Advanced Settings"):
            min_cats = st.slider("Minimum Categories", 2, 10, 4, 
                                 help="Minimum number of categories to identify Likert items")
            max_cats = st.slider("Maximum Categories", min_cats, 15, 7, 
                                 help="Maximum number of categories to identify Likert items")
            reverse_threshold = st.slider("Reverse Item Threshold", -1.0, 0.0, -0.2, 0.05,
                                         help="Correlation threshold to identify reverse-coded items")
            
            if analysis_method in ["Python Only", "Hybrid (Python + R)"]:
                n_clusters = st.number_input("Number of Clusters", min_value=0, value=0, 
                                           help="Number of item clusters (0 for automatic)")
                n_factors = st.number_input("Number of Factors", min_value=0, value=0,
                                          help="Number of factors (0 for automatic)")
            else:
                # Set defaults for R Only method
                n_clusters = 0
                n_factors = 0
            
            # Default number of simulations (we'll use a direct input in the Simulation tab)

# Only show the rest if a file is uploaded
if uploaded_file is not None:
    # Load the data
    try:
        df = load_data(uploaded_file)
        
        # Show the data preview
        st.header("Data Preview")
        st.write(f"Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")
        st.dataframe(df)
        
        # Show data information
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Responses", df.shape[0])
        with col2:
            st.metric("Number of Variables", df.shape[1])
            
        # Analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Data Preparation", 
            "Item Analysis", 
            "Pattern Extraction", 
            "Simulation", 
            "Reports"
        ])
        
        # Put results in session state for persistence across reruns
        if 'likert_items' not in st.session_state:
            st.session_state.likert_items = identify_likert_columns(df, min_cats, max_cats)
        if 'reverse_items' not in st.session_state:
            st.session_state.reverse_items = []
        if 'clusters' not in st.session_state:
            st.session_state.clusters = {}
        if 'weights' not in st.session_state:
            st.session_state.weights = {}
        if 'alphas' not in st.session_state:
            st.session_state.alphas = {}
        if 'alpha_ci' not in st.session_state:
            st.session_state.alpha_ci = {}
        if 'sim_data' not in st.session_state:
            st.session_state.sim_data = None
            
        # For debugging, add a button to clear the session state
        with st.sidebar:
            if st.button("Reset Analysis"):
                for key in ['likert_items', 'reverse_items', 'clusters', 'weights', 
                            'alphas', 'alpha_ci', 'sim_data', 'n_factors_detected']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            
        # Tab 1: Data Preparation
        with tab1:
            st.header("Data Preparation")
            
            # Automatically identify Likert items
            st.subheader("Identified Likert Items")
            likert_items = st.session_state.likert_items
            
            if len(likert_items) == 0:
                st.warning("No Likert scale items detected. Please check your data format.")
            else:
                st.success(f"Detected {len(likert_items)} Likert scale items")
                likert_selected = st.multiselect(
                    "Confirm Likert Items (unselect any non-Likert items)",
                    options=likert_items,
                    default=likert_items
                )
                
                # Update Likert items based on user selection
                if likert_selected != likert_items:
                    st.session_state.likert_items = likert_selected
                    likert_items = likert_selected
                    
                # Show distribution of all items in an expandable manner
                if likert_items:
                    with st.expander("Item Distributions", expanded=True):
                        st.subheader("Item Distributions")
                        
                        # Create a dropdown to select an item to visualize
                        selected_item = st.selectbox(
                            "Select an item to view its distribution",
                            options=likert_items
                        )
                        
                        # Show the distribution of the selected item
                        fig = px.histogram(
                            df, x=selected_item, 
                            nbins=len(df[selected_item].unique()),
                            title=f"Distribution of {selected_item}",
                            color_discrete_sequence=['#3366CC']
                        )
                        fig.update_layout(bargap=0.1)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create a multiselect to choose which items to display
                        items_to_show = st.multiselect(
                            "Select items to compare distributions",
                            options=likert_items,
                            default=likert_items[:min(3, len(likert_items))]
                        )
                        
                        if items_to_show:
                            # Create a comparison figure
                            fig = go.Figure()
                            for item in items_to_show:
                                counts = df[item].value_counts().sort_index()
                                fig.add_trace(go.Bar(
                                    x=counts.index,
                                    y=counts.values,
                                    name=item
                                ))
                            
                            fig.update_layout(
                                title="Item Distribution Comparison",
                                xaxis_title="Response Value",
                                yaxis_title="Count",
                                barmode='group'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # Detect and handle reverse-coded items
                st.subheader("Reverse-Coded Items Detection")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    reverse_threshold = st.slider(
                        "Reverse Item Detection Threshold", 
                        -1.0, 0.0, 0.0, 0.05,
                        help="Correlation threshold to identify reverse-coded items"
                    )
                
                with col2:
                    detect_button = st.button("Detect Reverse-Coded Items")
                
                # Create dedicated containers that will persist on the page
                reverse_count_container = st.empty()
                reverse_items_list = st.empty()
                
                if detect_button:
                    with st.spinner("Analyzing item correlations to detect reverse-coded items..."):
                        # Add more detailed analysis here
                        reverse_items = detect_reverse_items(df, likert_items, reverse_threshold)
                        st.session_state.reverse_items = reverse_items
                        
                        # Get correlation details for reverse items
                        corr_details = {}
                        corr_matrix = df[likert_items].corr()
                        
                        for item in reverse_items:
                            # Find items with highest negative correlation to this reverse item
                            negative_correlations = corr_matrix[item].sort_values().head(3)
                            corr_details[item] = negative_correlations.to_dict()
                        
                        st.session_state.reverse_details = corr_details
                        
                        # Log the detected items for debugging
                        print(f"Detected {len(reverse_items)} reverse-coded items: {reverse_items}")
                        
                        # Show results immediately
                        if len(reverse_items) > 0:
                            reverse_count_container.success(f"📋 Detected {len(reverse_items)} reverse-coded items")
                            reverse_items_list.markdown(f"### Reverse-coded items:\n**{', '.join(reverse_items)}**")
                        else:
                            reverse_count_container.info("No reverse-coded items detected in this dataset")
                
                # Always show the results, even if not from current button press
                elif 'reverse_items' in st.session_state and st.session_state.reverse_items:
                    # Only display if we have items
                    reverse_count_container.success(f"📋 Detected {len(st.session_state.reverse_items)} reverse-coded items")
                    reverse_items_list.markdown(f"### Reverse-coded items:\n**{', '.join(st.session_state.reverse_items)}**")
                
                # Detailed reverse items container (expandable analysis)
                if 'reverse_items' in st.session_state and st.session_state.reverse_items:
                    # Create a clear separator to make the expanded analysis more distinct
                    st.markdown("---")
                    st.subheader("Reverse-Coded Items Analysis")
                    
                    # Show detailed information about each reverse item
                    for i, item in enumerate(st.session_state.reverse_items):
                        with st.expander(f"Reverse item {i+1}: {item}", expanded=i==0):
                            st.write("This item appears to be reverse-coded compared to other items.")
                            
                            if 'reverse_details' in st.session_state:
                                st.write("#### Strongest negative correlations with other items:")
                                
                                details = st.session_state.reverse_details.get(item, {})
                                for other_item, corr_value in details.items():
                                    st.metric(
                                        f"Correlation with {other_item}", 
                                        f"{corr_value:.3f}",
                                        delta="Negative correlation confirms reverse coding"
                                    )
                            
                            # Show before/after histograms if we apply reverse coding
                            if len(st.session_state.reverse_items) > 0:
                                col1, col2 = st.columns(2)
                                
                                # Left: before reverse coding
                                with col1:
                                    st.write("##### Original Distribution")
                                    counts = df[item].value_counts().sort_index()
                                    fig = px.bar(
                                        x=counts.index, 
                                        y=counts.values,
                                        title="Before Reverse Coding"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Right: after reverse coding  
                                with col2:
                                    st.write("##### After Reverse Coding (Preview)")
                                    scale_min = int(df[likert_items].min().min())
                                    scale_max = int(df[likert_items].max().max())
                                    
                                    # Reverse code this single item for preview
                                    reversed_values = scale_min + scale_max - df[item]
                                    counts = reversed_values.value_counts().sort_index()
                                    
                                    fig = px.bar(
                                        x=counts.index, 
                                        y=counts.values,
                                        title="After Reverse Coding"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                
                    # Moved the button outside of the expanders for better visibility
                    if st.button("Apply Reverse Coding to All Items"):
                        scale_min = int(df[likert_items].min().min())
                        scale_max = int(df[likert_items].max().max())
                        df = reverse_code(df, st.session_state.reverse_items, scale_min, scale_max)
                        st.success(f"Reverse coding applied to {len(st.session_state.reverse_items)} items")
                
                # Sampling adequacy tests
                st.subheader("Sampling Adequacy")
                with st.spinner("Checking sampling adequacy..."):
                    sampling = check_sampling(df, likert_items)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("KMO", f"{sampling['kmo']:.3f}", 
                                  delta="+Good" if sampling['kmo'] > 0.6 else "-Poor")
                    with col2:
                        st.metric("Bartlett's p-value", f"{sampling['bartlett_p']:.3e}", 
                                  delta="+Significant" if sampling['bartlett_p'] < 0.05 else "-Not Significant")
        
        # Tab 2: Item Analysis
        with tab2:
            st.header("Item Analysis")
            
            if len(st.session_state.likert_items) == 0:
                st.warning("Please identify Likert items in the Data Preparation tab first")
            else:
                # Cluster the items
                st.subheader("Item Clustering")
                
                if st.button("Cluster Items"):
                    with st.spinner("Clustering items..."):
                        clusters = cluster_items(
                            df, st.session_state.likert_items, 
                            n_clusters if n_clusters > 0 else None
                        )
                        st.session_state.clusters = clusters
                
                if st.session_state.clusters:
                    st.success(f"Identified {len(st.session_state.clusters)} clusters")
                    
                    # Calculate reliability for each cluster
                    alphas = {}
                    alpha_ci = {}
                    for sc, items in st.session_state.clusters.items():
                        if len(items) > 1:  # Need at least 2 items for reliability
                            alphas[sc] = cronbach_alpha(df, items)
                            alpha_ci[sc] = bootstrap_alpha(df, items)
                    
                    st.session_state.alphas = alphas
                    st.session_state.alpha_ci = alpha_ci
                    
                    # Display each cluster and its reliability
                    for sc, items in st.session_state.clusters.items():
                        with st.expander(f"Cluster {sc} ({len(items)} items)"):
                            st.write(", ".join(items))
                            
                            if sc in alphas:
                                st.metric(
                                    "Cronbach's Alpha", 
                                    f"{alphas[sc]:.3f}",
                                    delta=f"CI: [{alpha_ci[sc][0]:.3f}, {alpha_ci[sc][1]:.3f}]"
                                )
                                
                                # Item correlations within cluster
                                corr = df[items].corr()
                                fig = px.imshow(
                                    corr, 
                                    title=f"Item Correlations (Cluster {sc})",
                                    color_continuous_scale="Blues"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                # Create enhanced network graph of item relationships
                st.subheader("Item Relationships Network")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    # Add layout selection
                    layout_option = st.selectbox(
                        "Graph Layout",
                        options=["force", "circular", "kamada_kawai", "spectral"],
                        index=0,
                        help="Select the algorithm used to position nodes in the graph"
                    )
                
                with col2:
                    # Add correlation threshold slider
                    corr_threshold = st.slider(
                        "Correlation Threshold",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.3,
                        step=0.05,
                        help="Minimum correlation required to display a connection between items"
                    )
                
                with col3:
                    # Add button to generate the graph
                    generate_graph = st.button("Generate Network Graph")
                
                # Create the graph when the button is clicked
                if generate_graph:
                    with st.spinner("Creating enhanced network visualization..."):
                        fig = create_network_graph(
                            df, 
                            st.session_state.likert_items,
                            threshold=corr_threshold,
                            layout=layout_option
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("""
                        **How to read this graph:**
                        - **Nodes** represent survey items, sized by their importance (higher correlation sum = larger node)
                        - **Colors** indicate communities of related items detected by clustering
                        - **Lines** represent correlations between items, with thickness proportional to correlation strength
                        - Hover over nodes to see details about each item and its strongest connections
                        - Drag nodes to reposition them for better viewing
                        """)
        
        # Tab 3: Pattern Extraction
        with tab3:
            st.header("Pattern Extraction")
            
            if not st.session_state.clusters:
                st.warning("Please cluster items in the Item Analysis tab first")
            else:
                # Determine factors
                if 'n_factors_detected' not in st.session_state:
                    with st.spinner("Determining optimal number of factors..."):
                        n_factors_detected = determine_factors(df, st.session_state.likert_items)
                        st.session_state.n_factors_detected = n_factors_detected
                
                st.info(f"Optimal number of factors: {st.session_state.n_factors_detected}")
                
                # Extract weights
                if st.button("Extract Item Weights"):
                    with st.spinner("Extracting item weights..."):
                        weights = extract_weights(
                            df, 
                            st.session_state.clusters,
                            n_factors if n_factors > 0 else st.session_state.n_factors_detected
                        )
                        st.session_state.weights = weights
                
                if st.session_state.weights:
                    st.success("Item weights extracted successfully")
                    
                    # Visualize weights by cluster
                    for sc, items in st.session_state.clusters.items():
                        # Extract weights in a format suitable for visualization
                        vis_weights = {}
                        for item in items:
                            if item in st.session_state.weights:
                                w_data = st.session_state.weights[item]
                                if isinstance(w_data, dict):
                                    if w_data.get('is_distribution', False):
                                        # For distribution-based weights, use average value
                                        if 'weights' in w_data:
                                            dist = w_data['weights']
                                            # Calculate weighted average
                                            try:
                                                avg = sum(float(k) * v for k, v in dist.items()) / sum(dist.values())
                                                vis_weights[item] = avg
                                            except:
                                                vis_weights[item] = 0.5  # Fallback
                                    else:
                                        # For factor analysis weights
                                        vis_weights[item] = w_data.get('weight', 0.5)
                                else:
                                    # Legacy format (simple value)
                                    vis_weights[item] = w_data
                        
                        if vis_weights:
                            fig = px.bar(
                                x=list(vis_weights.keys()),
                                y=list(vis_weights.values()),
                                title=f"Item Weights (Cluster {sc})"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a more detailed weights table showing all information
                    weights_rows = []
                    for item, w_data in st.session_state.weights.items():
                        if isinstance(w_data, dict):
                            if w_data.get('is_distribution', False):
                                # Distribution-based
                                weights_rows.append({
                                    'Item': item,
                                    'Weight Type': 'Distribution', 
                                    'Weight': str(w_data.get('weights', {}))[:30] + '...' if len(str(w_data.get('weights', {}))) > 30 else str(w_data.get('weights', {}))
                                })
                            else:
                                # Factor-based
                                weights_rows.append({
                                    'Item': item,
                                    'Weight Type': 'Factor Loading', 
                                    'Weight': w_data.get('weight', 0.0)
                                })
                        else:
                            # Legacy format
                            weights_rows.append({
                                'Item': item,
                                'Weight Type': 'Simple', 
                                'Weight': w_data
                            })
                    
                    # Display detailed weights table
                    weights_df = pd.DataFrame(weights_rows)
                    st.dataframe(weights_df)
                    
                    # Download weights - create a simpler version for download
                    simple_weights = {}
                    for item, w_data in st.session_state.weights.items():
                        if isinstance(w_data, dict):
                            if w_data.get('is_distribution', False):
                                # For distribution, use the weights dict
                                simple_weights[item] = str(w_data.get('weights', {}))
                            else:
                                # For factor analysis weights
                                simple_weights[item] = w_data.get('weight', 0.0)
                        else:
                            # Legacy format
                            simple_weights[item] = w_data
                    
                    # Create simple CSV for download
                    download_df = pd.DataFrame({
                        'Item': simple_weights.keys(),
                        'Weight': simple_weights.values()
                    })
                    weights_csv = download_df.to_csv(index=False)
                    
                    st.download_button(
                        "Download Weights as CSV",
                        weights_csv,
                        file_name=f"likert_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        # Tab 4: Simulation
        with tab4:
            st.header("Response Simulation")
            
            if not st.session_state.weights:
                st.warning("Please extract item weights in the Pattern Extraction tab first")
            else:
                st.subheader("Simulate New Responses")
                
                noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05, 
                                       help="Higher values create more variable responses")
                
                # Allow user to specify number of simulations (default 100)
                num_simulations = st.number_input("Number of responses to simulate", 
                                               min_value=10, 
                                               max_value=10000, 
                                               value=100,
                                               step=10,
                                               help="Enter the number of simulated responses you want to generate")
                
                if st.button("Simulate Responses"):
                    with st.spinner(f"Simulating {num_simulations} responses..."):
                        # Make sure all clustered items are in the weights
                        updated_weights = st.session_state.weights.copy()
                        
                        # Check for missing items from clusters that need to be included
                        if st.session_state.clusters:
                            # Get all unique items from clusters
                            all_cluster_items = set()
                            for items in st.session_state.clusters.values():
                                all_cluster_items.update(items)
                            
                            # Add any missing items with default distribution weights
                            for item in all_cluster_items:
                                if item not in updated_weights and item in df.columns:
                                    # Use the actual distribution from the data
                                    try:
                                        counts = df[item].value_counts().sort_index()
                                        values = counts.index.values
                                        counts_values = counts.values
                                        
                                        # Normalize to sum to 1
                                        if sum(counts_values) > 0:
                                            normalized_weights = counts_values / sum(counts_values)
                                            weight_dict = {val: weight for val, weight in zip(values, normalized_weights)}
                                            
                                            updated_weights[item] = {
                                                'is_distribution': True,
                                                'weights': weight_dict
                                            }
                                        else:
                                            # If no data, use equal weights
                                            updated_weights[item] = {
                                                'is_distribution': False,
                                                'weight': 0.5
                                            }
                                    except Exception as e:
                                        st.warning(f"Could not create weights for {item}: {str(e)}")
                                        updated_weights[item] = {
                                            'is_distribution': False,
                                            'weight': 0.5
                                        }
                        
                        # Now simulate with complete weights
                        sim_data = simulate_responses(updated_weights, num_simulations, noise_level)
                        st.session_state.sim_data = sim_data
                        
                        # Update the session state weights with the augmented weights to ensure consistency
                        st.session_state.weights = updated_weights
                
                if st.session_state.sim_data is not None:
                    st.success(f"Generated {len(st.session_state.sim_data)} simulated responses")
                    
                    # Show preview of simulated data
                    st.subheader("Simulated Data Preview")
                    with st.expander("View simulated data", expanded=True):
                        # Allow user to select how many rows to display (limit to the number of simulated responses)
                        total_rows = len(st.session_state.sim_data)
                        num_rows = st.slider("Number of rows to display", 5, min(total_rows, 1000), min(10, total_rows))
                        
                        # Show data preview with selected number of rows (respect user selection)
                        st.dataframe(st.session_state.sim_data.head(num_rows))
                        
                        # Show summary statistics
                        st.subheader("Summary Statistics")
                        st.dataframe(st.session_state.sim_data.describe())
                    
                    # Compare original vs simulated data
                    st.subheader("Original vs Simulated Distributions")
                    
                    # Option to select multiple items to compare or single item
                    compare_type = st.radio(
                        "Comparison type", 
                        ["Show one item in detail", "Show multiple items side by side"]
                    )
                    
                    items = list(st.session_state.weights.keys())
                    
                    if compare_type == "Show one item in detail":
                        # Select a single item to visualize in detail
                        selected_item = st.selectbox("Select item to visualize", items)
                        
                        if selected_item:
                            # Create histograms with overlay
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=df[selected_item],
                                name="Original",
                                opacity=0.7,
                                marker_color="blue"
                            ))
                            fig.add_trace(go.Histogram(
                                x=st.session_state.sim_data[selected_item],
                                name="Simulated",
                                opacity=0.7,
                                marker_color="red"
                            ))
                            fig.update_layout(
                                title=f"Distribution Comparison: {selected_item}",
                                xaxis_title="Response Value",
                                yaxis_title="Count",
                                barmode="overlay"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show percentages side by side
                            st.subheader(f"Response Percentages: {selected_item}")
                            
                            # Create a DataFrame with percentages for comparison
                            orig_counts = df[selected_item].value_counts(normalize=True).sort_index() * 100
                            sim_counts = st.session_state.sim_data[selected_item].value_counts(normalize=True).sort_index() * 100
                            
                            comparison_df = pd.DataFrame({
                                'Response': sorted(set(orig_counts.index) | set(sim_counts.index)),
                                'Original (%)': [orig_counts.get(i, 0) for i in sorted(set(orig_counts.index) | set(sim_counts.index))],
                                'Simulated (%)': [sim_counts.get(i, 0) for i in sorted(set(orig_counts.index) | set(sim_counts.index))],
                            })
                            
                            # Format as percentages with 1 decimal point
                            comparison_df['Original (%)'] = comparison_df['Original (%)'].apply(lambda x: f"{x:.1f}%")
                            comparison_df['Simulated (%)'] = comparison_df['Simulated (%)'].apply(lambda x: f"{x:.1f}%")
                            
                            st.dataframe(comparison_df)
                            
                    else:  # Show multiple items
                        # Select multiple items to compare
                        selected_items = st.multiselect(
                            "Select items to compare (max 6 recommended)",
                            options=items,
                            default=items[:min(3, len(items))]
                        )
                        
                        if selected_items:
                            # Create a grid of subplots
                            cols = min(2, len(selected_items))
                            rows = (len(selected_items) + 1) // 2
                            
                            fig = make_subplots(rows=rows, cols=cols, 
                                              subplot_titles=[f"Item: {item}" for item in selected_items])
                            
                            # Add traces for each item
                            for i, item in enumerate(selected_items):
                                row = i // cols + 1
                                col = i % cols + 1
                                
                                # Original data
                                fig.add_trace(
                                    go.Histogram(
                                        x=df[item],
                                        name=f"Original {item}",
                                        opacity=0.7,
                                        marker_color="blue",
                                        showlegend=(i == 0)  # Only show legend once
                                    ),
                                    row=row, col=col
                                )
                                
                                # Simulated data
                                fig.add_trace(
                                    go.Histogram(
                                        x=st.session_state.sim_data[item],
                                        name=f"Simulated {item}",
                                        opacity=0.7,
                                        marker_color="red",
                                        showlegend=(i == 0)  # Only show legend once
                                    ),
                                    row=row, col=col
                                )
                                
                                # Update layout for each subplot
                                fig.update_xaxes(title_text="Response Value", row=row, col=col)
                                if col == 1:  # Only for left column
                                    fig.update_yaxes(title_text="Count", row=row, col=col)
                            
                            # Update overall layout
                            fig.update_layout(
                                title="Distribution Comparison: Multiple Items",
                                height=300 * rows,
                                barmode="overlay"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Download simulated data
                    sim_csv = st.session_state.sim_data.to_csv(index=False)
                    st.download_button(
                        "Download Simulated Data as CSV",
                        sim_csv,
                        file_name=f"simulated_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Statistical comparison of real vs simulated data
                    st.markdown("---")
                    st.subheader("Statistical Comparison: Real vs Simulated Data")
                    
                    # Add a button to perform comprehensive statistical comparison
                    analyze_button = st.button("Compare Real vs Simulated Data")
                    
                    if analyze_button or 'show_stat_analysis' in st.session_state:
                        st.session_state.show_stat_analysis = True
                        
                        try:
                            # 1. Basic descriptive statistics comparison
                            with st.expander("Descriptive Statistics Comparison", expanded=True):
                                # Calculate descriptive statistics
                                real_desc = df[st.session_state.likert_items].describe().T
                                sim_desc = st.session_state.sim_data[st.session_state.likert_items].describe().T
                                
                                # Calculate additional statistics
                                real_desc['var'] = df[st.session_state.likert_items].var()
                                sim_desc['var'] = st.session_state.sim_data[st.session_state.likert_items].var()
                                
                                real_desc['skew'] = df[st.session_state.likert_items].skew()
                                sim_desc['skew'] = st.session_state.sim_data[st.session_state.likert_items].skew()
                                
                                real_desc['kurtosis'] = df[st.session_state.likert_items].kurtosis()
                                sim_desc['kurtosis'] = st.session_state.sim_data[st.session_state.likert_items].kurtosis()
                                
                                # Calculate mean absolute differences between real and simulated statistics
                                stats_diff = {}
                                metrics = ['mean', 'std', 'var', 'skew', 'kurtosis', 'min', '25%', '50%', '75%', 'max']
                                for metric in metrics:
                                    if metric in real_desc.columns and metric in sim_desc.columns:
                                        diff = abs(real_desc[metric] - sim_desc[metric])
                                        stats_diff[metric] = diff.mean()
                                
                                # Create comparison charts for key metrics
                                selected_stats = st.multiselect(
                                    "Select statistics to compare", 
                                    options=list(stats_diff.keys()),
                                    default=['mean', 'std', 'var']
                                )
                                
                                for stat in selected_stats:
                                    st.subheader(f"Comparison of {stat.capitalize()}")
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(
                                        x=real_desc.index,
                                        y=real_desc[stat],
                                        name="Original",
                                        marker_color='blue'
                                    ))
                                    fig.add_trace(go.Bar(
                                        x=sim_desc.index,
                                        y=sim_desc[stat],
                                        name="Simulated",
                                        marker_color='red'
                                    ))
                                    fig.update_layout(
                                        title=f"{stat.capitalize()} Comparison",
                                        barmode='group',
                                        height=400
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Overall similarity scores
                                st.subheader("Similarity Metrics")
                                similarity_df = pd.DataFrame({
                                    'Statistic': list(stats_diff.keys()),
                                    'Mean Absolute Difference': [stats_diff[k] for k in stats_diff.keys()],
                                    'Similarity Score (%)': [max(0, 100 - 100 * stats_diff[k]) for k in stats_diff.keys()]
                                })
                                st.dataframe(similarity_df.sort_values('Similarity Score (%)', ascending=False))
                                
                                # Overall similarity score
                                overall_similarity = similarity_df['Similarity Score (%)'].mean()
                                st.metric("Overall Statistical Similarity", f"{overall_similarity:.2f}%")
                            
                            # 2. Correlation structure comparison
                            with st.expander("Correlation Structure Comparison", expanded=True):
                                # Calculate correlation matrices
                                real_corr = df[st.session_state.likert_items].corr()
                                sim_corr = st.session_state.sim_data[st.session_state.likert_items].corr()
                                
                                # Calculate correlation matrix difference
                                corr_diff = abs(real_corr - sim_corr)
                                
                                # Display side-by-side correlation heatmaps
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.write("Original Correlation Matrix")
                                    fig = px.imshow(
                                        real_corr, 
                                        title="Original Data Correlations",
                                        color_continuous_scale="Blues",
                                        zmin=-1, zmax=1
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    st.write("Simulated Correlation Matrix")
                                    fig = px.imshow(
                                        sim_corr, 
                                        title="Simulated Data Correlations",
                                        color_continuous_scale="Reds",
                                        zmin=-1, zmax=1
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col3:
                                    st.write("Difference Matrix")
                                    fig = px.imshow(
                                        corr_diff, 
                                        title="Correlation Difference (Absolute)",
                                        color_continuous_scale="Greens",
                                        zmin=0, zmax=2
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Calculate overall correlation similarity
                                mean_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean()
                                corr_similarity = max(0, 100 - (mean_diff * 100))
                                st.metric("Correlation Structure Similarity", f"{corr_similarity:.2f}%")
                            
                            # 3. Distribution comparison
                            with st.expander("Distribution Comparison", expanded=True):
                                # Select items to compare
                                dist_items = st.multiselect(
                                    "Select items for distribution comparison",
                                    options=st.session_state.likert_items,
                                    default=st.session_state.likert_items[:min(3, len(st.session_state.likert_items))]
                                )
                                
                                if dist_items:
                                    # Calculate KL divergence for each item
                                    kl_divergences = {}
                                    js_distances = {}
                                    
                                    for item in dist_items:
                                        # Calculate distribution proportions
                                        real_dist = df[item].value_counts(normalize=True).sort_index()
                                        sim_dist = st.session_state.sim_data[item].value_counts(normalize=True).sort_index()
                                        
                                        # Ensure both distributions have the same categories
                                        all_values = sorted(set(real_dist.index) | set(sim_dist.index))
                                        real_probs = np.array([real_dist.get(val, 0) for val in all_values])
                                        sim_probs = np.array([sim_dist.get(val, 0) for val in all_values])
                                        
                                        # Avoid zero probabilities for KL divergence
                                        real_probs = np.clip(real_probs, 1e-10, 1.0)
                                        sim_probs = np.clip(sim_probs, 1e-10, 1.0)
                                        
                                        # Normalize to sum to 1
                                        real_probs = real_probs / real_probs.sum()
                                        sim_probs = sim_probs / sim_probs.sum()
                                        
                                        try:
                                            # Calculate KL divergence: D_KL(P||Q)
                                            kl_div = np.sum(real_probs * np.log(real_probs / sim_probs))
                                            kl_divergences[item] = kl_div
                                            
                                            # Calculate Jensen-Shannon distance
                                            m_dist = 0.5 * (real_probs + sim_probs)
                                            js_div = 0.5 * np.sum(real_probs * np.log(real_probs / m_dist)) + 0.5 * np.sum(sim_probs * np.log(sim_probs / m_dist))
                                            js_distances[item] = np.sqrt(js_div)  # JS distance is sqrt of JS divergence
                                        except Exception as e:
                                            st.warning(f"Could not calculate divergence for item {item}: {str(e)}")
                                    
                                    # Display the divergence measures
                                    divergence_df = pd.DataFrame({
                                        'Item': list(kl_divergences.keys()),
                                        'KL Divergence': list(kl_divergences.values()),
                                        'JS Distance': list(js_distances.values()),
                                        'Distribution Similarity (%)': [max(0, 100 - (js * 100)) for js in js_distances.values()]
                                    })
                                    st.dataframe(divergence_df.sort_values('Distribution Similarity (%)', ascending=False))
                                    
                                    # Overall distribution similarity
                                    dist_similarity = divergence_df['Distribution Similarity (%)'].mean()
                                    st.metric("Overall Distribution Similarity", f"{dist_similarity:.2f}%")
                            
                            # 4. Reliability Measures (Cronbach's Alpha comparison)
                            if st.session_state.clusters:
                                with st.expander("Reliability Comparison", expanded=True):
                                    # Calculate alphas for original and simulated data by cluster
                                    sim_alphas = {}
                                    
                                    # First verify all required columns exist in simulated data
                                    available_items = set(st.session_state.sim_data.columns)
                                    
                                    # Make sure all items exist in the simulated data
                                    valid_items = {}
                                    for sc, items in st.session_state.clusters.items():
                                        # Check if all items in this cluster exist in the simulated data
                                        valid_cluster_items = [item for item in items if item in st.session_state.sim_data.columns]
                                        if len(valid_cluster_items) > 1:  # Need at least 2 items for reliability
                                            valid_items[sc] = valid_cluster_items
                                    
                                    if valid_items:
                                        alpha_data = []
                                        
                                        for sc, items in valid_items.items():
                                            try:
                                                orig_alpha = st.session_state.alphas.get(sc, 0)
                                                sim_alpha = cronbach_alpha(st.session_state.sim_data, items)
                                                
                                                alpha_diff = abs(orig_alpha - sim_alpha)
                                                alpha_similarity = max(0, 100 - (alpha_diff * 100))
                                                
                                                alpha_data.append({
                                                    'Cluster': sc,
                                                    'Items': len(items),
                                                    'Original Alpha': orig_alpha,
                                                    'Simulated Alpha': sim_alpha,
                                                    'Difference': alpha_diff,
                                                    'Similarity (%)': alpha_similarity
                                                })
                                            except Exception as e:
                                                st.warning(f"Could not calculate reliability for cluster {sc}: {str(e)}")
                                        
                                        # Display results table
                                        alpha_df = pd.DataFrame(alpha_data)
                                        if not alpha_df.empty:
                                            st.dataframe(alpha_df.sort_values('Similarity (%)', ascending=False))
                                            
                                            # Overall reliability similarity
                                            reliability_similarity = alpha_df['Similarity (%)'].mean()
                                            st.metric("Overall Reliability Similarity", f"{reliability_similarity:.2f}%")
                                            
                                            # Display clusters with reliability issues
                                            problem_clusters = alpha_df[alpha_df['Similarity (%)'] < 75]
                                            if not problem_clusters.empty:
                                                st.warning("The following clusters show significant reliability differences:")
                                                st.dataframe(problem_clusters[['Cluster', 'Original Alpha', 'Simulated Alpha', 'Similarity (%)']])
                                                
                                                # Suggestions for improvement
                                                st.info("💡 Suggestions: Use a larger dataset or try adjusting noise level or using different item weights extraction methods.")
                            
                            # Final combined score
                            overall_metrics = []
                            
                            if 'overall_similarity' in locals():
                                overall_metrics.append(('Statistical Descriptives', overall_similarity))
                            
                            if 'corr_similarity' in locals():
                                overall_metrics.append(('Correlation Structure', corr_similarity))
                            
                            if 'dist_similarity' in locals():
                                overall_metrics.append(('Distribution Similarity', dist_similarity))
                            
                            if 'reliability_similarity' in locals():
                                overall_metrics.append(('Reliability Metrics', reliability_similarity))
                            
                            if overall_metrics:
                                st.markdown("---")
                                st.subheader("Overall Similarity Assessment")
                                
                                final_score = sum(score for _, score in overall_metrics) / len(overall_metrics)
                                
                                # Create a gauge chart for final score
                                fig = go.Figure(go.Indicator(
                                    mode = "gauge+number",
                                    value = final_score,
                                    domain = {'x': [0, 1], 'y': [0, 1]},
                                    title = {'text': "Overall Simulation Quality"},
                                    gauge = {
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "royalblue"},
                                        'steps': [
                                            {'range': [0, 50], 'color': "red"},
                                            {'range': [50, 75], 'color': "orange"},
                                            {'range': [75, 90], 'color': "yellow"},
                                            {'range': [90, 100], 'color': "green"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "black", 'width': 4},
                                            'thickness': 0.75,
                                            'value': final_score
                                        }
                                    }
                                ))
                                fig.update_layout(height=300)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Individual score breakdown
                                score_df = pd.DataFrame(overall_metrics, columns=['Metric', 'Score (%)'])
                                fig = px.bar(
                                    score_df, 
                                    x='Score (%)', 
                                    y='Metric', 
                                    orientation='h',
                                    title="Component Similarity Scores",
                                    color='Score (%)',
                                    color_continuous_scale=px.colors.sequential.Viridis,
                                    range_color=[0, 100]
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Quality assessment
                                if final_score >= 90:
                                    st.success("🌟 Excellent simulation quality! The simulated data closely matches the original data across all metrics.")
                                elif final_score >= 80:
                                    st.success("✅ Good simulation quality. The simulated data captures most patterns in the original data.")
                                elif final_score >= 70:
                                    st.warning("⚠️ Fair simulation quality. The simulated data captures some patterns but has notable differences.")
                                else:
                                    st.error("❌ Poor simulation quality. Consider adjusting parameters to improve results.")
                                
                        except Exception as e:
                            st.error(f"Error performing statistical comparison: {str(e)}")
                    else:
                        st.info("Click the 'Compare Real vs Simulated Data' button above for a comprehensive statistical comparison between your original and simulated data.")
        
        # Tab 5: Reports
        with tab5:
            st.header("Analysis Reports")
            
            # Simple info about analysis method
            if analysis_method != "Python Only":
                st.info("Currently using analysis method: " + analysis_method)
                
            # Direct users to the right place for analysis information
            st.write("The detailed analysis results are available in the respective tabs:")
            st.markdown("""
            - **Data Preparation**: View item distributions and detect reverse-coded items
            - **Item Analysis**: See clustering results and reliability metrics
            - **Pattern Extraction**: Access item weights and factor analysis results
            - **Simulation**: Generate and validate simulated responses
            """)
            
            st.subheader("Generate Reports")
            st.write("Use the buttons below to export your analysis results:")
            
            # Generate HTML report
            if st.button("Generate Full Report"):
                with st.spinner("Generating report..."):
                    # Prepare simulation statistics to include in the report
                    sim_stats = {}
                    if st.session_state.sim_data is not None and 'show_stat_analysis' in st.session_state:
                        if 'overall_similarity' in globals():
                            sim_stats['descriptives'] = overall_similarity
                        
                        if 'corr_similarity' in globals():
                            sim_stats['correlations'] = corr_similarity
                        
                        if 'dist_similarity' in globals():
                            sim_stats['distributions'] = dist_similarity
                        
                        if 'reliability_similarity' in globals():
                            sim_stats['reliability'] = reliability_similarity
                        
                        # Calculate overall score if we have at least one metric
                        if sim_stats:
                            sim_stats['overall'] = sum(sim_stats.values()) / len(sim_stats)
                    
                    report_path = save_html_report({
                        'data': df,
                        'likert_items': st.session_state.likert_items,
                        'reverse_items': st.session_state.reverse_items,
                        'clusters': st.session_state.clusters,
                        'weights': st.session_state.weights,
                        'alphas': st.session_state.alphas,
                        'alpha_ci': st.session_state.alpha_ci,
                        'simulated': st.session_state.sim_data,
                        'sim_stats': sim_stats
                    })
                    
                    # Display the report
                    with open(report_path, 'r') as file:
                        html_content = file.read()
                    
                    # Create a download button for the report
                    st.download_button(
                        "Download HTML Report",
                        html_content,
                        file_name=f"likert_analysis_report_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html"
                    )
                    
                    # Display in iframe (limited functionality in Streamlit)
                    st.components.v1.html(html_content, height=600, scrolling=True)
            
            # Export all results to JSON
            if st.button("Export All Results"):
                # Prepare results
                # Prepare weights in a serializable format
                serializable_weights = {}
                for item, w_data in st.session_state.weights.items():
                    if isinstance(w_data, dict):
                        # Complex weights format
                        if w_data.get('is_distribution', False) and 'weights' in w_data:
                            # Distribution weights need to be converted to strings
                            weights_dict = {}
                            for k, v in w_data['weights'].items():
                                weights_dict[str(k)] = float(v)
                            serializable_weights[item] = {
                                'is_distribution': True,
                                'weights': weights_dict
                            }
                        else:
                            # Factor weights can be serialized directly
                            serializable_weights[item] = {
                                'is_distribution': False,
                                'weight': float(w_data.get('weight', 0.0))
                            }
                            if 'dist_weights' in w_data and w_data['dist_weights']:
                                dist_dict = {}
                                for k, v in w_data['dist_weights'].items():
                                    dist_dict[str(k)] = float(v)
                                serializable_weights[item]['dist_weights'] = dist_dict
                    else:
                        # Legacy format (just a float)
                        try:
                            serializable_weights[item] = float(w_data)
                        except:
                            serializable_weights[item] = 0.0
                
                # Prepare all results for export
                all_results = {
                    'timestamp': datetime.now().isoformat(),
                    'analysis_method': st.session_state.analysis_method,
                    'likert_items': st.session_state.likert_items,
                    'reverse_items': st.session_state.reverse_items,
                    'clusters': {str(k): v for k, v in st.session_state.clusters.items()},
                    'weights': serializable_weights,
                    'alphas': {str(k): float(v) for k, v in st.session_state.alphas.items()},
                    'alpha_ci': {str(k): [float(v[0]), float(v[1])] for k, v in st.session_state.alpha_ci.items()}
                }
                
                # Include simulation data and statistics if available
                if st.session_state.sim_data is not None:
                    # Add some basic information about the simulated data
                    all_results['simulated_count'] = len(st.session_state.sim_data)
                    
                    # Add simulation statistics if they have been calculated
                    sim_stats = {}
                    if 'show_stat_analysis' in st.session_state:
                        if 'overall_similarity' in globals():
                            sim_stats['descriptives'] = float(overall_similarity)
                        
                        if 'corr_similarity' in globals():
                            sim_stats['correlations'] = float(corr_similarity)
                        
                        if 'dist_similarity' in globals():
                            sim_stats['distributions'] = float(dist_similarity)
                        
                        if 'reliability_similarity' in globals():
                            sim_stats['reliability'] = float(reliability_similarity)
                        
                        # Calculate overall score if we have at least one metric
                        if sim_stats:
                            sim_stats['overall'] = float(sum(sim_stats.values()) / len(sim_stats))
                        
                        # Add to results
                        all_results['sim_stats'] = sim_stats
                
                # Convert to JSON
                results_json = json.dumps(all_results, indent=2)
                
                # Create download button
                st.download_button(
                    "Download Results as JSON",
                    results_json,
                    file_name=f"likert_analysis_results_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.exception(e)
else:
    # Show instructions when no file is uploaded
    st.info("Please upload a CSV or Excel file containing Likert scale survey data.")
    
    # Sample instruction guide
    with st.expander("How to prepare your data"):
        st.markdown("""
        ### Data Format Requirements
        
        - Your data should be in CSV or Excel format
        - Each row represents a respondent
        - Each column represents a survey item or demographic variable
        - Likert scale items should have integer values (e.g., 1-5, 1-7)
        - Missing values should be represented as empty cells or standard missing value codes (Better results if data is screened/cleaned first)
        
        ### Typical Data Structure
        
        | Respondent | Gender | Age | Item1 | Item2 | Item3 | ... |
        |------------|--------|-----|-------|-------|-------|-----|
        | 1          | F      | 25  | 4     | 5     | 3     | ... |
        | 2          | M      | 31  | 2     | 4     | 4     | ... |
        | ...        | ...    | ... | ...   | ...   | ...   | ... |
        
        ### Naming Conventions
        
        For best results, name your items with a common prefix to indicate which scale they belong to:
        
        - X1, X2, X3 (items for X scale)
        - Y1, Y2, Y3 (items for Y scale)
        - etc.
        """)

# Footer
st.markdown("---")
st.markdown(
    "Likert Scale Pattern Analysis Tool | "
    f"Version 1.0 | {datetime.now().year}"
)

# Removed Back to Top button as it's not compatible with Streamlit's iframe structure
