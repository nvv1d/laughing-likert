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
import re
from collections import defaultdict
from auth import init_auth_session_state, render_login_page, is_authenticated, update_last_activity, render_logout_button, get_current_username
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

# New helper functions for scale detection and biased simulation
def detect_scales_by_pattern(columns):
    """
    Automatically detect scales based on naming patterns like A1, A2, A3, B1, B2, B3, etc.
    Returns a dictionary with scale names as keys and lists of items as values.
    """
    scales = defaultdict(list)

    # Pattern 1: Letter + Number (A1, A2, B1, B2, etc.)
    pattern1 = re.compile(r'^([A-Za-z]+)(\d+)$')

    # Pattern 2: Word + Number (Scale1_1, Scale1_2, Scale2_1, etc.)
    pattern2 = re.compile(r'^([A-Za-z_]+?)(\d+)(?:_(\d+))?$')

    # Pattern 3: Common prefixes (att1, att2, sat1, sat2, etc.)
    pattern3 = re.compile(r'^([A-Za-z]+)(\d+)$')

    for col in columns:
        # Try pattern 1: A1, A2, B1, B2
        match1 = pattern1.match(col)
        if match1:
            prefix = match1.group(1).upper()
            number = int(match1.group(2))
            scales[f"Scale_{prefix}"].append((col, number))
            continue

        # Try pattern 2: Scale1_1, Scale1_2, Scale2_1
        match2 = pattern2.match(col)
        if match2:
            prefix = match2.group(1)
            if match2.group(3):  # Has underscore number
                scale_num = int(match2.group(2))
                item_num = int(match2.group(3))
                scales[f"Scale_{prefix}_{scale_num}"].append((col, item_num))
            else:
                number = int(match2.group(2))
                scales[f"Scale_{prefix}"].append((col, number))
            continue

    # Sort items within each scale by their numbers and return just the column names
    sorted_scales = {}
    for scale_name, items in scales.items():
        if len(items) >= 2:  # Only keep scales with at least 2 items
            sorted_items = sorted(items, key=lambda x: x[1])
            sorted_scales[scale_name] = [item[0] for item in sorted_items]

    return sorted_scales

def apply_bias_to_weights(weights, bias_type, bias_strength, bias_percentage):
    """
    Apply bias to existing weights to simulate high-achievers or low-achievers.
    
    Parameters:
    - weights: Original weights dictionary
    - bias_type: 'high' or 'low'
    - bias_strength: float 0-1, how strong the bias is
    - bias_percentage: float 0-1, what percentage of responses should be biased
    
    Returns:
    - Modified weights dictionary
    """
    biased_weights = weights.copy()

    for item, w_data in biased_weights.items():
        if isinstance(w_data, dict) and w_data.get('is_distribution', False):
            original_dist = w_data['weights'].copy()

            # Create biased distribution
            biased_dist = {}

            # Get all possible values and sort them
            values = sorted([int(k) for k in original_dist.keys()])

            if bias_type == 'high':
                # Shift probability mass toward higher values
                target_values = values[-2:]  # Top 2 values
            else:  # bias_type == 'low'
                # Shift probability mass toward lower values
                target_values = values[:2]  # Bottom 2 values

            # Calculate bias adjustment
            for val_str, prob in original_dist.items():
                val = int(val_str)

                if val in target_values:
                    # Increase probability for target values
                    bias_multiplier = 1 + (bias_strength * bias_percentage)
                    biased_dist[val_str] = prob * bias_multiplier
                else:
                    # Decrease probability for other values
                    bias_multiplier = 1 - (bias_strength * bias_percentage * 0.5)
                    biased_dist[val_str] = prob * bias_multiplier

            # Normalize to ensure probabilities sum to 1
            total_prob = sum(biased_dist.values())
            if total_prob > 0:
                biased_dist = {k: v/total_prob for k, v in biased_dist.items()}

            # Update the weights
            biased_weights[item] = {
                'is_distribution': True,
                'weights': biased_dist
            }

    return biased_weights

# Page configuration
st.set_page_config(
    page_title="Likert Scale Pattern Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize authentication
init_auth_session_state()

# Display login if not logged in
if not is_authenticated():
    render_login_page()
    st.stop()

# Update last activity when logged in
update_last_activity()

# Main Application (only shown if logged in)
st.title("Likert Scale Pattern Analysis")
st.markdown("""
This application helps you analyze Likert scale survey data, extract response patterns, 
and generate simulated data based on those patterns. Upload your data file to get started.
""")

# Sidebar for inputs and controls
with st.sidebar:
    st.write(f"👋 Welcome, {get_current_username()}!")
    render_logout_button()

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
            min_cats = st.slider("Minimum Categories", 2, 10, 3,
                                 help="Minimum number of categories to identify Likert items")
            max_cats = st.slider("Maximum Categories", min_cats, 15, 10,
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
            st.session_state.likert_items = []
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
        if 'detected_scales' not in st.session_state:
            st.session_state.detected_scales = {}

        # For debugging, add a button to clear the session state
        with st.sidebar:
            if st.button("Reset Analysis"):
                for key in ['likert_items', 'reverse_items', 'clusters', 'weights',
                            'alphas', 'alpha_ci', 'sim_data', 'n_factors_detected', 'detected_scales']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        # Tab 1: Data Preparation - ENHANCED VERSION
        with tab1:
            st.header("Data Preparation")

            # Show data summary first
            st.subheader("Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", numeric_cols)

            # Enhanced Likert item detection
            st.subheader("Likert Item Detection")

            # Add detection method selection
            detection_method = st.radio(
                "Detection Method",
                ["Automatic (Smart Detection)", "Manual Selection", "All Numeric Columns"],
                help="Choose how to identify Likert scale items"
            )

            if detection_method == "Automatic (Smart Detection)":
                col1, col2 = st.columns(2)
                with col1:
                    min_cats_display = st.slider("Minimum Categories", 2, 10, min_cats,
                                        key="min_cats_display",
                                        help="Minimum number of unique values to identify Likert items")
                with col2:
                    max_cats_display = st.slider("Maximum Categories", min_cats_display, 15, max_cats,
                                        key="max_cats_display",
                                        help="Maximum number of unique values to identify Likert items")

                # Re-identify Likert items with new parameters
                if st.button("Re-detect Likert Items") or len(st.session_state.likert_items) == 0:
                    with st.spinner("Analyzing columns for Likert scale characteristics..."):
                        detected_items = identify_likert_columns(df, min_cats_display, max_cats_display)
                        st.session_state.likert_items = detected_items

                # Show detection results with detailed breakdown
                st.subheader("Detection Results")

                # Analyze all numeric columns and show why some were excluded
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                detection_details = []

                for col in numeric_columns:
                    unique_vals = df[col].nunique()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    has_negatives = min_val < 0
                    is_detected = col in st.session_state.likert_items

                    # Determine why it was included/excluded
                    if is_detected:
                        reason = "✅ Detected as Likert item"
                    elif unique_vals < min_cats_display:
                        reason = f"❌ Too few categories ({unique_vals} < {min_cats_display})"
                    elif unique_vals > max_cats_display:
                        reason = f"❌ Too many categories ({unique_vals} > {max_cats_display})"
                    elif has_negatives:
                        reason = f"⚠️ Contains negative values (min: {min_val})"
                    else:
                        reason = "❌ Other exclusion criteria"

                    detection_details.append({
                        'Column': col,
                        'Unique Values': unique_vals,
                        'Range': f"{min_val} - {max_val}",
                        'Status': reason,
                        'Include': is_detected
                    })

                # Display the detection table
                detection_df = pd.DataFrame(detection_details)

                # Show summary
                detected_count = len(st.session_state.likert_items)
                excluded_count = len(numeric_columns) - detected_count

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Detected Items", detected_count, delta=f"+{detected_count} included")
                with col2:
                    st.metric("Excluded Items", excluded_count, delta=f"-{excluded_count} excluded")
                with col3:
                    detection_rate = (detected_count / len(numeric_columns) * 100) if numeric_columns else 0
                    st.metric("Detection Rate", f"{detection_rate:.1f}%")

                # Show detailed breakdown in expandable section
                with st.expander("View Detection Details", expanded=True):
                    # Add filter options
                    filter_option = st.selectbox("Filter by:", ["All columns", "Detected only", "Excluded only"])

                    if filter_option == "Detected only":
                        display_df = detection_df[detection_df['Include'] == True]
                    elif filter_option == "Excluded only":
                        display_df = detection_df[detection_df['Include'] == False]
                    else:
                        display_df = detection_df

                    st.dataframe(display_df, use_container_width=True)

                    # Show excluded items that might be recoverable
                    potentially_recoverable = detection_df[
                        (detection_df['Include'] == False) &
                        (detection_df['Unique Values'] >= 2) &
                        (~detection_df['Status'].str.contains('negative'))
                    ]

                    if len(potentially_recoverable) > 0:
                        st.warning(f"Found {len(potentially_recoverable)} potentially recoverable items that were excluded due to category count limits.")
                        recoverable_items = st.multiselect(
                            "Select items to include anyway:",
                            options=potentially_recoverable['Column'].tolist(),
                            help="These items were excluded due to category limits but might still be valid Likert items"
                        )

                        if recoverable_items:
                            # Add recovered items to the Likert items list
                            updated_items = list(set(st.session_state.likert_items + recoverable_items))
                            st.session_state.likert_items = updated_items
                            st.success(f"Added {len(recoverable_items)} recovered items. Total Likert items: {len(updated_items)}")
                            st.rerun()

            elif detection_method == "Manual Selection":
                # Let user manually select from all columns
                st.write("Select which columns represent Likert scale items:")

                # Get all numeric columns as candidates
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

                # Show column info to help with selection
                col_info = []
                for col in numeric_columns:
                    unique_vals = df[col].nunique()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    col_info.append(f"{col} (Range: {min_val}-{max_val}, {unique_vals} categories)")

                # Multi-select with current selection as default
                current_selection = st.session_state.get('likert_items', [])
                selected_items = st.multiselect(
                    "Choose Likert scale columns:",
                    options=numeric_columns,
                    default=[item for item in current_selection if item in numeric_columns],
                    format_func=lambda x: next((info for info in col_info if info.startswith(x)), x),
                    help="Select all columns that contain Likert scale responses"
                )

                st.session_state.likert_items = selected_items

                if selected_items:
                    st.success(f"Selected {len(selected_items)} Likert items")
                else:
                    st.warning("No items selected")

            else:  # All Numeric Columns
                # Use all numeric columns as Likert items
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                st.session_state.likert_items = numeric_columns

                st.info(f"Using all {len(numeric_columns)} numeric columns as Likert items")

                # Show the columns that will be used
                with st.expander("View all numeric columns", expanded=False):
                    col_details = []
                    for col in numeric_columns:
                        unique_vals = df[col].nunique()
                        min_val = df[col].min()
                        max_val = df[col].max()
                        col_details.append({
                            'Column': col,
                            'Categories': unique_vals,
                            'Min': min_val,
                            'Max': max_val,
                            'Range': f"{min_val} - {max_val}"
                        })

                    st.dataframe(pd.DataFrame(col_details))

            # Final confirmation and overview
            likert_items = st.session_state.likert_items

            if len(likert_items) == 0:
                st.error("No Likert scale items identified. Please adjust your detection settings or manually select items.")
            else:
                # Show final selection with option to fine-tune
                st.subheader("Final Likert Items Selection")
                st.success(f"✅ **{len(likert_items)} Likert items** ready for analysis")

                # Allow final adjustments
                with st.expander("Fine-tune selection (optional)", expanded=False):
                    adjusted_items = st.multiselect(
                        "Adjust final selection:",
                        options=df.select_dtypes(include=[np.number]).columns.tolist(),
                        default=likert_items,
                        help="Make final adjustments to your Likert items selection"
                    )

                    if adjusted_items != likert_items:
                        st.session_state.likert_items = adjusted_items
                        likert_items = adjusted_items
                        st.success(f"Updated selection: {len(adjusted_items)} items")
                        st.rerun()

                # NEW FEATURE: Automatic Scale Detection and Reliability Analysis
                st.markdown("---")
                st.subheader("🔍 Automatic Scale Detection & Reliability Analysis")

                # Detect scales based on naming patterns
                if st.button("🔎 Detect Scales by Naming Pattern"):
                    with st.spinner("Analyzing column names for scale patterns..."):
                        detected_scales = detect_scales_by_pattern(likert_items)
                        st.session_state.detected_scales = detected_scales

                        if detected_scales:
                            st.success(f"🎯 Detected {len(detected_scales)} scales with clear naming patterns!")
                        else:
                            st.warning("⚠️ No clear naming patterns detected. Items may not follow standard naming conventions (e.g., A1, A2, B1, B2).")

                # Show detected scales if available
                if st.session_state.detected_scales:
                    st.subheader("📋 Detected Scales")

                    # Calculate reliability for each detected scale
                    scale_reliability = {}
                    reliability_details = []

                    for scale_name, scale_items in st.session_state.detected_scales.items():
                        # Filter items that actually exist in the data
                        valid_items = [item for item in scale_items if item in df.columns]

                        if len(valid_items) >= 2:
                            try:
                                alpha = cronbach_alpha(df, valid_items)
                                alpha_ci = bootstrap_alpha(df, valid_items, n_bootstrap=100)

                                scale_reliability[scale_name] = {
                                    'items': valid_items,
                                    'n_items': len(valid_items),
                                    'alpha': alpha,
                                    'alpha_ci': alpha_ci,
                                    'quality': 'Excellent' if alpha >= 0.9 else
                                              'Good' if alpha >= 0.8 else
                                              'Acceptable' if alpha >= 0.7 else
                                              'Questionable' if alpha >= 0.6 else 'Poor'
                                }

                                reliability_details.append({
                                    'Scale': scale_name,
                                    'Items': len(valid_items),
                                    'Cronbach Alpha': f"{alpha:.3f}",
                                    '95% CI': f"[{alpha_ci[0]:.3f}, {alpha_ci[1]:.3f}]",
                                    'Quality': scale_reliability[scale_name]['quality'],
                                    'Item List': ', '.join(valid_items)
                                })
                            except Exception as e:
                                st.warning(f"Could not calculate reliability for scale {scale_name}: {str(e)}")

                    # Display reliability results in a nice format
                    if reliability_details:
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)

                        excellent_scales = len([s for s in scale_reliability.values() if s['alpha'] >= 0.9])
                        good_scales = len([s for s in scale_reliability.values() if 0.8 <= s['alpha'] < 0.9])
                        acceptable_scales = len([s for s in scale_reliability.values() if 0.7 <= s['alpha'] < 0.8])
                        poor_scales = len([s for s in scale_reliability.values() if s['alpha'] < 0.7])

                        with col1:
                            st.metric("Excellent (≥0.9)", excellent_scales, delta="✅")
                        with col2:
                            st.metric("Good (≥0.8)", good_scales, delta="👍")
                        with col3:
                            st.metric("Acceptable (≥0.7)", acceptable_scales, delta="⚠️")
                        with col4:
                            st.metric("Poor (<0.7)", poor_scales, delta="❌")

                        # Detailed results table
                        reliability_df = pd.DataFrame(reliability_details)
                        st.dataframe(reliability_df, use_container_width=True)
                        
                        # Option to export reliability results
                        if st.button("📥 Export Reliability Analysis"):
                            # Create comprehensive export data
                            export_data = {
                                'analysis_timestamp': datetime.now().isoformat(),
                                'total_scales': len(scale_reliability),
                                'scale_details': []
                            }

                            for scale_name, scale_info in scale_reliability.items():
                                item_stats = []
                                for item in scale_info['items']:
                                    other_items = [i for i in scale_info['items'] if i != item]
                                    other_total = df[other_items].sum(axis=1)
                                    corr = df[item].corr(other_total)

                                    item_stats.append({
                                        'item': item,
                                        'mean': float(df[item].mean()),
                                        'std': float(df[item].std()),
                                        'corrected_item_total_correlation': float(corr)
                                    })

                                export_data['scale_details'].append({
                                    'scale_name': scale_name,
                                    'n_items': scale_info['n_items'],
                                    'cronbach_alpha': float(scale_info['alpha']),
                                    'alpha_ci_lower': float(scale_info['alpha_ci'][0]),
                                    'alpha_ci_upper': float(scale_info['alpha_ci'][1]),
                                    'quality': scale_info['quality'],
                                    'items': scale_info['items'],
                                    'item_statistics': item_stats
                                })

                            # Convert to JSON and create download
                            export_json = json.dumps(export_data, indent=2)

                            st.download_button(
                                "Download Reliability Analysis (JSON)",
                                export_json,
                                file_name=f"scale_reliability_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )

                            # Also create a CSV summary
                            csv_data = []
                            for scale_name, scale_info in scale_reliability.items():
                                csv_data.append({
                                    'Scale_Name': scale_name,
                                    'N_Items': scale_info['n_items'],
                                    'Cronbach_Alpha': scale_info['alpha'],
                                    'CI_Lower': scale_info['alpha_ci'][0],
                                    'CI_Upper': scale_info['alpha_ci'][1],
                                    'Quality': scale_info['quality'],
                                    'Items': '; '.join(scale_info['items'])
                                })

                            csv_df = pd.DataFrame(csv_data)
                            csv_string = csv_df.to_csv(index=False)

                            st.download_button(
                                "Download Reliability Summary (CSV)",
                                csv_string,
                                file_name=f"scale_reliability_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                        # Expandable detailed analysis for each scale
                        st.subheader("📊 Detailed Scale Analysis")

                        for scale_name, scale_info in scale_reliability.items():
                            with st.expander(f"📈 {scale_name} (α = {scale_info['alpha']:.3f}, {scale_info['quality']})",
               expanded=bool(scale_info['alpha'] < 0.7)):  # Auto-expand poor scales

                                col1, col2 = st.columns([2, 1])

                                with col1:
                                    # Scale statistics
                                    st.write("**Scale Information:**")
                                    st.write(f"- **Items**: {', '.join(scale_info['items'])}")
                                    st.write(f"- **Number of items**: {scale_info['n_items']}")
                                    st.write(f"- **Cronbach's Alpha**: {scale_info['alpha']:.3f}")
                                    st.write(f"- **95% Confidence Interval**: [{scale_info['alpha_ci'][0]:.3f}, {scale_info['alpha_ci'][1]:.3f}]")
                                    st.write(f"- **Reliability Quality**: {scale_info['quality']}")

                                    # Item-total correlations
                                    st.write("**Item-Total Correlations:**")
                                    scale_data = df[scale_info['items']]
                                    scale_total = scale_data.sum(axis=1)

                                    correlations = []
                                    for item in scale_info['items']:
                                        # Corrected item-total correlation (excluding the item itself)
                                        other_items = [i for i in scale_info['items'] if i != item]
                                        other_total = df[other_items].sum(axis=1)
                                        corr = df[item].corr(other_total)
                                        correlations.append({
                                            'Item': item,
                                            'Corrected Item-Total r': f"{corr:.3f}"
                                        })

                                    corr_df = pd.DataFrame(correlations)
                                    st.dataframe(corr_df, use_container_width=True)

                                with col2:
                                    # Reliability interpretation
                                    st.write("**Interpretation:**")
                                    if scale_info['alpha'] >= 0.9:
                                        st.success("🌟 Excellent internal consistency")
                                    elif scale_info['alpha'] >= 0.8:
                                        st.success("✅ Good internal consistency")
                                    elif scale_info['alpha'] >= 0.7:
                                        st.warning("⚠️ Acceptable internal consistency")
                                    elif scale_info['alpha'] >= 0.6:
                                        st.warning("⚡ Questionable internal consistency")
                                    else:
                                        st.error("❌ Poor internal consistency")

                                    # Recommendations
                                    st.write("**Recommendations:**")
                                    if scale_info['alpha'] < 0.7:
                                        st.write("- Consider removing poor-performing items")
                                        st.write("- Check for reverse-coded items")
                                        st.write("- Examine item wording for clarity")
                                    elif scale_info['alpha'] < 0.8:
                                        st.write("- Scale is usable but could be improved")
                                        st.write("- Consider adding more items")
                                    else:
                                        st.write("- Scale shows good reliability")
                                        st.write("- Suitable for research use")

                                # Inter-item correlation matrix for this scale
                                if len(scale_info['items']) <= 10:  # Only show for smaller scales
                                    st.write("**Inter-item Correlation Matrix:**")
                                    corr_matrix = df[scale_info['items']].corr()

                                    fig = px.imshow(
                                        corr_matrix,
                                        title=f"Inter-item Correlations: {scale_name}",
                                        color_continuous_scale="RdBu_r",
                                        zmin=-1, zmax=1,
                                        aspect="auto"
                                    )
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No scales with sufficient items (≥2) found for reliability analysis.")

                else:
                    # Manual scale definition option
                    st.info("💡 **Tip**: If automatic detection didn't work, you can manually group your items into scales for reliability analysis.")

                    with st.expander("🔧 Manual Scale Definition", expanded=False):
                        st.write("Define scales manually by grouping related items:")

                        # Initialize manual scales in session state
                        if 'manual_scales' not in st.session_state:
                            st.session_state.manual_scales = {}

                        # Add new scale
                        new_scale_name = st.text_input("New Scale Name (e.g., 'Satisfaction', 'Attitude'):")

                        if new_scale_name:
                            scale_items = st.multiselect(
                                f"Select items for '{new_scale_name}':",
                                options=likert_items,
                                key=f"manual_scale_{new_scale_name}"
                            )

                            if len(scale_items) >= 2:
                                if st.button(f"Add '{new_scale_name}' Scale"):
                                    st.session_state.manual_scales[new_scale_name] = scale_items
                                    st.success(f"Added scale '{new_scale_name}' with {len(scale_items)} items")
                                    st.rerun()

                        # Show existing manual scales
                        if st.session_state.manual_scales:
                            st.write("**Defined Scales:**")
                            for scale_name, items in st.session_state.manual_scales.items():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"- **{scale_name}**: {', '.join(items)}")
                                with col2:
                                    if st.button(f"Remove", key=f"remove_{scale_name}"):
                                        del st.session_state.manual_scales[scale_name]
                                        st.rerun()

                            # Calculate reliability for manual scales
                            if st.button("Calculate Reliability for Manual Scales"):
                                manual_reliability = {}
                                for scale_name, scale_items in st.session_state.manual_scales.items():
                                    if len(scale_items) >= 2:
                                        try:
                                            alpha = cronbach_alpha(df, scale_items)
                                            alpha_ci = bootstrap_alpha(df, scale_items, n_bootstrap=100)
                                            manual_reliability[scale_name] = {
                                                'alpha': alpha,
                                                'alpha_ci': alpha_ci,
                                                'items': scale_items
                                            }
                                        except Exception as e:
                                            st.warning(f"Could not calculate reliability for {scale_name}: {str(e)}")

                                # Display results
                                if manual_reliability:
                                    st.subheader("Manual Scale Reliability Results")
                                    for scale_name, results in manual_reliability.items():
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric(f"{scale_name}", f"α = {results['alpha']:.3f}")
                                        with col2:
                                            st.write(f"95% CI: [{results['alpha_ci'][0]:.3f}, {results['alpha_ci'][1]:.3f}]")
                                        with col3:
                                            quality = ('Excellent' if results['alpha'] >= 0.9 else
                                                     'Good' if results['alpha'] >= 0.8 else
                                                     'Acceptable' if results['alpha'] >= 0.7 else
                                                     'Questionable' if results['alpha'] >= 0.6 else 'Poor')
                                            st.write(f"Quality: {quality}")

                # Show distribution of selected items
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
                    reverse_threshold_display = st.slider(
                        "Reverse Item Detection Threshold",
                        -1.0, 0.0, reverse_threshold, 0.05,
                        key="reverse_threshold_display",
                        help="Correlation threshold to identify reverse-coded items"
                    )

                with col2:
                    st.write("") # Placeholder for alignment
                    st.write("") # Placeholder for alignment
                    detect_button = st.button("Detect Reverse-Coded Items")

                # Create dedicated containers that will persist on the page
                reverse_count_container = st.empty()
                reverse_items_list = st.empty()

                if detect_button:
                    with st.spinner("Analyzing item correlations to detect reverse-coded items..."):
                        # Add more detailed analysis here
                        reverse_items = detect_reverse_items(df, likert_items, reverse_threshold_display)
                        st.session_state.reverse_items = reverse_items

                        # Get correlation details for reverse items
                        corr_details = {}
                        corr_matrix = df[likert_items].corr()

                        for item in reverse_items:
                            # Find items with highest negative correlation to this reverse item
                            negative_correlations = corr_matrix[item].sort_values().head(3)
                            corr_details[item] = negative_correlations.to_dict()

                        st.session_state.reverse_details = corr_details

                        # Show results immediately
                        if len(reverse_items) > 0:
                            reverse_count_container.success(f"📋 Detected {len(reverse_items)} reverse-coded items")
                            reverse_items_list.markdown(f"### Reverse-coded items:\n**{', '.join(reverse_items)}**")
                        else:
                            reverse_count_container.info("No reverse-coded items detected in this dataset")
                
                # Always show the results, even if not from current button press
                elif 'reverse_items' in st.session_state and st.session_state.reverse_items:
                    reverse_count_container.success(f"📋 Detected {len(st.session_state.reverse_items)} reverse-coded items")
                    reverse_items_list.markdown(f"### Reverse-coded items:\n**{', '.join(st.session_state.reverse_items)}**")
                
                # Detailed reverse items container (expandable analysis)
                if 'reverse_items' in st.session_state and st.session_state.reverse_items:
                    st.markdown("---")
                    st.subheader("Reverse-Coded Items Analysis")

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
                                        delta="Negative correlation"
                                    )

                            if len(st.session_state.reverse_items) > 0:
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write("##### Original Distribution")
                                    counts = df[item].value_counts().sort_index()
                                    fig = px.bar(
                                        x=counts.index,
                                        y=counts.values,
                                        title="Before Reverse Coding"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                                with col2:
                                    st.write("##### After Reverse Coding (Preview)")
                                    scale_min = int(df[likert_items].min().min())
                                    scale_max = int(df[likert_items].max().max())
                                    
                                    reversed_values = scale_min + scale_max - df[item]
                                    counts = reversed_values.value_counts().sort_index()
                                    
                                    fig = px.bar(
                                        x=counts.index,
                                        y=counts.values,
                                        title="After Reverse Coding"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)

                    if st.button("Apply Reverse Coding to All Items"):
                        scale_min = int(df[likert_items].min().min())
                        scale_max = int(df[likert_items].max().max())
                        df = reverse_code(df, st.session_state.reverse_items, scale_min, scale_max)
                        st.success(f"Reverse coding applied to {len(st.session_state.reverse_items)} items")
                        st.rerun()

                # Sampling adequacy tests
                st.subheader("Sampling Adequacy")
                with st.spinner("Checking sampling adequacy..."):
                    sampling = check_sampling(df, likert_items)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("KMO", f"{sampling['kmo']:.3f}",
                                  delta="Good" if sampling['kmo'] > 0.6 else "Poor")
                    with col2:
                        st.metric("Bartlett's p-value", f"{sampling['bartlett_p']:.3e}",
                                  delta="Significant" if sampling['bartlett_p'] < 0.05 else "Not Significant")
        
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
                
                col1, col2 = st.columns([2, 1])
                
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
                
                if st.button("Generate Network Graph"):
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

            if not st.session_state.get('clusters'):
                st.warning("Please cluster items in the Item Analysis tab first.")
            else:
                # Determine factors if not already done
                if 'n_factors_detected' not in st.session_state:
                    with st.spinner("Determining optimal number of factors..."):
                        # Use only the likert items that have been clustered
                        clustered_items = [item for sublist in st.session_state.clusters.values() for item in sublist]
                        if clustered_items:
                            n_factors_detected = determine_factors(df, clustered_items)
                            st.session_state.n_factors_detected = n_factors_detected
                        else:
                             st.session_state.n_factors_detected = 1 # Default if no clusters found

                st.info(f"🤖 Optimal number of factors suggested by analysis: **{st.session_state.n_factors_detected}**")

                # Extract weights
                if st.button("Extract Item Weights"):
                    with st.spinner("Extracting item weights... This may take a moment."):
                        weights = extract_weights(
                            df,
                            st.session_state.clusters,
                            n_factors if n_factors > 0 else st.session_state.n_factors_detected
                        )
                        st.session_state.weights = weights

                if st.session_state.get('weights'):
                    st.success("✅ Item weights extracted successfully!")

                    # Visualize weights by cluster
                    st.subheader("Item Weight Visualization")
                    for sc, items in st.session_state.clusters.items():
                        # Extract weights in a format suitable for visualization
                        vis_weights = {}
                        for item in items:
                            if item in st.session_state.weights:
                                w_data = st.session_state.weights[item]
                                if isinstance(w_data, dict):
                                    if w_data.get('is_distribution', False):
                                        # For distribution-based weights, calculate the mean for visualization
                                        if 'weights' in w_data and w_data['weights']:
                                            dist = w_data['weights']
                                            try:
                                                avg = sum(float(k) * v for k, v in dist.items())
                                                vis_weights[item] = avg
                                            except (ValueError, TypeError):
                                                vis_weights[item] = 0.5  # Fallback
                                        else:
                                             vis_weights[item] = 0.5 # Fallback
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
                                title=f"Item Weights (Cluster {sc})",
                                labels={'x': 'Item', 'y': 'Weight/Mean Value'}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # Create a more detailed weights table showing all information
                    st.subheader("Detailed Weights Table")
                    weights_rows = []
                    for item, w_data in st.session_state.weights.items():
                        row = {'Item': item}
                        if isinstance(w_data, dict):
                            if w_data.get('is_distribution', False):
                                # For distribution-based weights, show the dictionary as a string
                                weight_display = str(w_data.get('weights', {}))
                                row['Weight Type'] = 'Distribution'
                                # Truncate long strings for cleaner display
                                row['Weight'] = (weight_display[:70] + '...') if len(weight_display) > 70 else weight_display
                            else:
                                # For factor-based weights, format the float as a string
                                weight_val = w_data.get('weight', 0.0)
                                row['Weight Type'] = 'Factor Loading'
                                row['Weight'] = f"{weight_val:.4f}"
                        else:
                            # For legacy simple weights, format the float as a string
                            weight_val = w_data
                            row['Weight Type'] = 'Simple'
                            row['Weight'] = f"{weight_val:.4f}"
                        weights_rows.append(row)

                    # Display detailed weights table
                    if weights_rows:
                        weights_df = pd.DataFrame(weights_rows)
                        st.dataframe(weights_df, use_container_width=True)

                    # Download weights
                    st.subheader("Export Weights")
                    simple_weights = {}
                    for item, w_data in st.session_state.weights.items():
                        if isinstance(w_data, dict):
                            if w_data.get('is_distribution', False):
                                # For distribution, use the string representation of the weights dict
                                simple_weights[item] = str(w_data.get('weights', {}))
                            else:
                                # For factor analysis weights
                                simple_weights[item] = w_data.get('weight', 0.0)
                        else:
                            # Legacy format
                            simple_weights[item] = w_data

                    download_df = pd.DataFrame({
                        'Item': simple_weights.keys(),
                        'Weight': simple_weights.values()
                    })
                    weights_csv = download_df.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        label="📥 Download Weights as CSV",
                        data=weights_csv,
                        file_name=f"likert_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        # Tab 4: Simulation - ENHANCED WITH BIAS OPTIONS
        with tab4:
            st.header("Response Simulation")
            
            if not st.session_state.weights:
                st.warning("Please extract item weights in the Pattern Extraction tab first")
            else:
                st.subheader("Simulate New Responses")
                
                # Base simulation settings
                col1, col2 = st.columns(2)
                
                with col1:
                    noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1, 0.05, 
                                           help="Higher values create more variable responses")
                
                with col2:
                    num_simulations = st.number_input("Number of responses to simulate", 
                                                   min_value=10, 
                                                   max_value=10000, 
                                                   value=100,
                                                   step=10,
                                                   help="Enter the number of simulated responses you want to generate")
                
                # NEW FEATURE: Bias Options
                st.markdown("---")
                st.subheader("🎯 Response Bias Options")
                
                enable_bias = st.checkbox(
                    "Enable Response Bias", 
                    value=False,
                    help="Apply systematic bias to simulate specific response patterns"
                )
                
                if enable_bias:
                    st.info("🔧 Configure bias settings to simulate different respondent types (e.g., high-achievers, pessimists)")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        bias_type = st.selectbox(
                            "Bias Direction",
                            options=["high", "low"],
                            format_func=lambda x: "High Bias (optimistic/high-achievers)" if x == "high" else "Low Bias (pessimistic/critical)",
                            help="Direction of bias: high = toward maximum scale values, low = toward minimum scale values"
                        )
                    
                    with col2:
                        bias_strength = st.slider(
                            "Bias Strength",
                            min_value=0.1,
                            max_value=2.0,
                            value=0.5,
                            step=0.1,
                            help="How strong the bias is (higher = more extreme bias)"
                        )
                    
                    with col3:
                        bias_percentage = st.slider(
                            "Percentage Affected",
                            min_value=0.1,
                            max_value=1.0,
                            value=0.3,
                            step=0.05,
                            format_func=lambda x: f"{x*100:.0f}%",
                            help="What percentage of responses should be affected by bias"
                        )
                    
                    # Bias explanation and preview
                    with st.expander("🔍 Bias Configuration Preview", expanded=True):
                        if bias_type == "high":
                            st.success(f"**High Bias Configuration:**")
                            st.write(f"- **Effect**: {bias_percentage*100:.0f}% of responses will be biased toward higher values")
                            st.write(f"- **Strength**: {bias_strength:.1f}x increase in probability for top 2 scale values")
                            st.write(f"- **Use case**: Simulate high-achievers, optimistic respondents, or positive response bias")
                        else:
                            st.warning(f"**Low Bias Configuration:**")
                            st.write(f"- **Effect**: {bias_percentage*100:.0f}% of responses will be biased toward lower values")
                            st.write(f"- **Strength**: {bias_strength:.1f}x increase in probability for bottom 2 scale values")
                            st.write(f"- **Use case**: Simulate critical respondents, pessimistic views, or negative response bias")
                        
                        st.write(f"- **Unbiased responses**: {(1-bias_percentage)*100:.0f}% will follow original patterns")
                        
                        # Show example of how bias would affect a 5-point scale
                        st.write("**Example Effect on 5-point Scale (1-5):**")
                        example_original = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1}
                        
                        if bias_type == "high":
                            target_values = [4, 5]
                        else:
                            target_values = [1, 2]
                        
                        example_biased = {}
                        for val, prob in example_original.items():
                            if val in target_values:
                                example_biased[val] = prob * (1 + (bias_strength * bias_percentage))
                            else:
                                example_biased[val] = prob * (1 - (bias_strength * bias_percentage * 0.5))
                        
                        # Normalize
                        total_prob = sum(example_biased.values())
                        example_biased = {k: v/total_prob for k, v in example_biased.items()}
                        
                        comparison_df = pd.DataFrame({
                            'Scale Value': [1, 2, 3, 4, 5],
                            'Original': [example_original[i] for i in [1, 2, 3, 4, 5]],
                            'Biased': [example_biased[i] for i in [1, 2, 3, 4, 5]]
                        })
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=comparison_df['Scale Value'],
                            y=comparison_df['Original'],
                            name='Original Distribution',
                            marker_color='lightblue'
                        ))
                        fig.add_trace(go.Bar(
                            x=comparison_df['Scale Value'],
                            y=comparison_df['Biased'],
                            name='Biased Distribution',
                            marker_color='orange'
                        ))
                        fig.update_layout(
                            title="Example: Bias Effect on Response Distribution",
                            xaxis_title="Scale Value",
                            yaxis_title="Probability",
                            barmode='group',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Simulation button
                if st.button("🚀 Simulate Responses"):
                    with st.spinner(f"Simulating {num_simulations} responses..."):
                        # Make sure all clustered items are in the weights
                        updated_weights = st.session_state.weights.copy()
                        
                        # Check for missing items from clusters that need to be included
                        if st.session_state.clusters:
                            all_cluster_items = set(item for items in st.session_state.clusters.values() for item in items)
                            
                            # Add any missing items with default distribution weights
                            for item in all_cluster_items:
                                if item not in updated_weights and item in df.columns:
                                    try:
                                        counts = df[item].value_counts(normalize=True).sort_index()
                                        weight_dict = {str(val): prob for val, prob in counts.items()}
                                        updated_weights[item] = {
                                            'is_distribution': True,
                                            'weights': weight_dict
                                        }
                                    except Exception as e:
                                        st.warning(f"Could not create weights for {item}: {str(e)}")
                                        updated_weights[item] = {'is_distribution': False, 'weight': 0.5}
                        
                        # Apply bias if enabled
                        if enable_bias:
                            st.info(f"Applying {bias_type} bias (strength: {bias_strength}, affected: {bias_percentage*100:.0f}%)")
                            updated_weights = apply_bias_to_weights(
                                updated_weights, bias_type, bias_strength, bias_percentage
                            )
                        
                        # Now simulate with complete weights
                        sim_data = simulate_responses(updated_weights, num_simulations, noise_level)
                        
                        # Reorder columns to match original data order
                        original_order = st.session_state.likert_items
                        available_cols = [col for col in original_order if col in sim_data.columns]
                        
                        if available_cols:
                            sim_data = sim_data[available_cols]
                        
                        st.session_state.sim_data = sim_data
                        st.session_state.weights = updated_weights
                        st.success(f"🎯 Successfully generated {num_simulations} responses!")
                
                if st.session_state.sim_data is not None:
                    st.subheader("Simulated Data Preview")
                    with st.expander("View simulated data", expanded=True):
                        total_rows = len(st.session_state.sim_data)
                        num_rows = st.slider("Number of rows to display", 5, min(total_rows, 1000), min(10, total_rows), key="sim_rows_display")
                        st.dataframe(st.session_state.sim_data.head(num_rows))
                        st.subheader("Summary Statistics")
                        st.dataframe(st.session_state.sim_data.describe())
                    
                    # Compare original vs simulated data
                    st.subheader("Original vs Simulated Distributions")
                    compare_type = st.radio(
                        "Comparison type", 
                        ["Show one item in detail", "Show multiple items side by side"],
                        key="compare_type_radio"
                    )
                    
                    items = list(st.session_state.weights.keys())
                    
                    if compare_type == "Show one item in detail":
                        selected_item = st.selectbox("Select item to visualize", items)
                        if selected_item:
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(x=df[selected_item], name="Original", opacity=0.7))
                            fig.add_trace(go.Histogram(x=st.session_state.sim_data[selected_item], name="Simulated", opacity=0.7))
                            fig.update_layout(title=f"Distribution Comparison: {selected_item}", barmode="overlay")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        selected_items = st.multiselect("Select items to compare (max 6 recommended)", options=items, default=items[:min(3, len(items))])
                        if selected_items:
                            cols = min(2, len(selected_items))
                            rows = (len(selected_items) + 1) // 2
                            fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Item: {item}" for item in selected_items])
                            for i, item in enumerate(selected_items):
                                row = i // cols + 1
                                col = i % cols + 1
                                fig.add_trace(go.Histogram(x=df[item], name=f"Original {item}", showlegend=(i==0)), row=row, col=col)
                                fig.add_trace(go.Histogram(x=st.session_state.sim_data[item], name=f"Simulated {item}", showlegend=(i==0)), row=row, col=col)
                            fig.update_layout(title="Distribution Comparison: Multiple Items", height=300 * rows, barmode="overlay")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Download simulated data
                    sim_csv = st.session_state.sim_data.to_csv(index=False)
                    st.download_button("Download Simulated Data as CSV", sim_csv, file_name=f"simulated_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
                    
                    st.markdown("---")
                    st.subheader("Statistical Comparison: Real vs Simulated Data")
                    
                    if st.button("Compare Real vs Simulated Data"):
                        st.session_state.show_stat_analysis = True
                    
                    if st.session_state.get('show_stat_analysis', False):
                        try:
                            # Descriptive statistics comparison
                            with st.expander("Descriptive Statistics Comparison", expanded=True):
                                real_desc = df[st.session_state.likert_items].describe().T
                                sim_desc = st.session_state.sim_data[st.session_state.likert_items].describe().T
                                real_desc['var'] = df[st.session_state.likert_items].var()
                                sim_desc['var'] = st.session_state.sim_data[st.session_state.likert_items].var()
                                
                                stats_diff = {metric: abs(real_desc[metric] - sim_desc[metric]).mean() for metric in ['mean', 'std', 'var'] if metric in real_desc}
                                
                                st.subheader("Similarity Metrics")
                                similarity_df = pd.DataFrame({
                                    'Statistic': list(stats_diff.keys()),
                                    'Mean Absolute Difference': list(stats_diff.values()),
                                    'Similarity Score (%)': [max(0, 100 - 100 * diff) for diff in stats_diff.values()]
                                })
                                st.dataframe(similarity_df)
                                overall_similarity = similarity_df['Similarity Score (%)'].mean()
                                st.metric("Overall Statistical Similarity", f"{overall_similarity:.2f}%")

                            # Correlation structure comparison
                            with st.expander("Correlation Structure Comparison", expanded=True):
                                real_corr = df[st.session_state.likert_items].corr()
                                sim_corr = st.session_state.sim_data[st.session_state.likert_items].corr()
                                corr_diff = abs(real_corr - sim_corr)
                                
                                mean_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean()
                                corr_similarity = max(0, 100 - (mean_diff * 100))
                                st.metric("Correlation Structure Similarity", f"{corr_similarity:.2f}%")
                            
                            # Reliability comparison
                            if st.session_state.clusters:
                                with st.expander("Reliability Comparison", expanded=True):
                                    alpha_data = []
                                    valid_items = {sc: [item for item in items if item in st.session_state.sim_data.columns] for sc, items in st.session_state.clusters.items()}

                                    for sc, items in valid_items.items():
                                        if len(items) > 1:
                                            orig_alpha = st.session_state.alphas.get(sc, 0)
                                            sim_alpha = cronbach_alpha(st.session_state.sim_data, items)
                                            alpha_data.append({'Cluster': sc, 'Original Alpha': orig_alpha, 'Simulated Alpha': sim_alpha})
                                    
                                    alpha_df = pd.DataFrame(alpha_data)
                                    st.dataframe(alpha_df)
                                    
                        except Exception as e:
                            st.error(f"Error performing statistical comparison: {str(e)}")
        
        # Tab 5: Reports
        with tab5:
            st.header("Analysis Reports")
            st.write("Use the buttons below to export your analysis results.")

            # Generate HTML report
            if st.button("Generate Full Report"):
                with st.spinner("Generating report..."):
                    report_path = save_html_report({
                        'data': df,
                        'likert_items': st.session_state.likert_items,
                        'reverse_items': st.session_state.reverse_items,
                        'clusters': st.session_state.clusters,
                        'weights': st.session_state.weights,
                        'alphas': st.session_state.alphas,
                        'alpha_ci': st.session_state.alpha_ci,
                        'simulated': st.session_state.sim_data,
                        'detected_scales': st.session_state.get('detected_scales', {})
                    })
                    
                    with open(report_path, 'r') as file:
                        html_content = file.read()
                    
                    st.download_button(
                        "Download HTML Report",
                        html_content,
                        file_name=f"likert_analysis_report_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html"
                    )
                    st.components.v1.html(html_content, height=600, scrolling=True)

            # Export all results to JSON
            if st.button("Export All Results"):
                with st.spinner("Preparing JSON export..."):
                    serializable_weights = {}
                    for item, w_data in st.session_state.weights.items():
                        if isinstance(w_data, dict) and w_data.get('is_distribution', False):
                            serializable_weights[item] = {
                                'is_distribution': True,
                                'weights': {str(k): float(v) for k, v in w_data['weights'].items()}
                            }
                        else:
                            weight_val = w_data.get('weight', 0.0) if isinstance(w_data, dict) else w_data
                            serializable_weights[item] = {
                                'is_distribution': False,
                                'weight': float(weight_val)
                            }

                    # Prepare all results for JSON export
                    export_data = {
                        'analysis_timestamp': datetime.now().isoformat(),
                        'likert_items': st.session_state.likert_items,
                        'reverse_items': st.session_state.reverse_items,
                        'clusters': st.session_state.clusters,
                        'weights': serializable_weights,
                        'alphas': st.session_state.alphas,
                        'alpha_ci': st.session_state.alpha_ci,
                        'detected_scales': st.session_state.get('detected_scales', {})
                    }
                    
                    export_json = json.dumps(export_data, indent=4)
                    st.download_button(
                        "Download All Results (JSON)",
                        export_json,
                        file_name=f"full_analysis_results_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                    st.success("JSON export file is ready for download.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.exception(e)
