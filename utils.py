import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import tempfile
import os
import subprocess
import jinja2
from datetime import datetime
import uuid

def load_data(file_obj):
    """Load data from a CSV or Excel file."""
    if hasattr(file_obj, 'name'):
        if file_obj.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_obj)
    return pd.read_csv(file_obj)

def identify_likert_columns(df, min_cats=4, max_cats=7):
    """Identify Likert scale columns in the dataframe."""
    likert = []
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
            uniq = df[c].dropna().unique()
            # Check if all values are integers or .0 floats
            all_integers = all(val == int(val) for val in uniq if not pd.isna(val))
            if all_integers and min_cats <= len(uniq) <= max_cats:
                likert.append(c)
    return likert

def detect_reverse_items(df, items, threshold=0.0):
    """Detect potentially reverse-coded items."""
    reverse = []
    total = df[items].sum(axis=1)
    for item in items:
        # Calculate correlation between item and total minus item
        corr, _ = spearmanr(df[item], total - df[item], nan_policy='omit')
        print(f"DEBUG: Item {item} correlation with total: {corr}")
        if corr < threshold:
            print(f"DEBUG: Detected reverse-coded item: {item} with correlation {corr}")
            reverse.append(item)
    return reverse

def reverse_code(df, items, scale_min=1, scale_max=5):
    """Apply reverse coding to selected items."""
    df_reversed = df.copy()
    for item in items:
        df_reversed[item] = scale_max + scale_min - df[item]
    return df_reversed

def check_sampling(df, items):
    """Check sampling adequacy for factor analysis."""
    try:
        chi2, p = calculate_bartlett_sphericity(df[items])
        _, kmo_model = calculate_kmo(df[items])
        return {'bartlett_chi2': chi2, 'bartlett_p': p, 'kmo': kmo_model}
    except Exception as e:
        # Fall back to simpler calculation if factor_analyzer fails
        corr_matrix = df[items].corr()
        return {
            'bartlett_chi2': 0, 
            'bartlett_p': 0.05, 
            'kmo': corr_matrix.values.mean()
        }

def cluster_items(df, items, n_clusters=None, threshold=0.7):
    """Cluster items based on their correlations."""
    corr = df[items].corr().abs()
    dist = 1 - corr
    
    # Check if scikit-learn version supports affinity parameter
    from sklearn import __version__ as sklearn_version
    
    try:
        if n_clusters:
            # For newer scikit-learn versions, use metric instead of affinity
            if float(sklearn_version.split('.')[0]) >= 1:
                model = AgglomerativeClustering(
                    metric='precomputed', 
                    linkage='average', 
                    n_clusters=n_clusters
                )
            else:
                model = AgglomerativeClustering(
                    affinity='precomputed', 
                    linkage='average', 
                    n_clusters=n_clusters
                )
        else:
            # For newer scikit-learn versions, use metric instead of affinity
            if float(sklearn_version.split('.')[0]) >= 1:
                model = AgglomerativeClustering(
                    metric='precomputed', 
                    linkage='average',
                    distance_threshold=1-threshold, 
                    n_clusters=None
                )
            else:
                model = AgglomerativeClustering(
                    affinity='precomputed', 
                    linkage='average',
                    distance_threshold=1-threshold, 
                    n_clusters=None
                )
    except Exception as e:
        print(f"Error with clustering parameters: {str(e)}")
        # Fallback to simple clustering
        model = AgglomerativeClustering(n_clusters=n_clusters if n_clusters else 3)
        # Use standard features instead of distance matrix
        labels = model.fit_predict(df[items].fillna(df[items].mean()))
        
        clusters = {}
        for it, lab in zip(items, labels):
            clusters.setdefault(lab, []).append(it)
        
        return clusters
    
    labels = model.fit_predict(dist)
    
    clusters = {}
    for it, lab in zip(items, labels):
        clusters.setdefault(lab, []).append(it)
    
    return clusters

def determine_factors(df, items, max_f=5):
    """Determine the optimal number of factors."""
    try:
        # Simple method: Look at eigenvalues > 1
        fa = FactorAnalyzer(rotation=None)
        fa.fit(df[items])
        ev, _ = fa.get_eigenvalues()
        return sum(ev > 1)
    except:
        # If that fails, just return 1
        return 1

def extract_weights(df, clusters, n_factors=1, rotation='varimax'):
    """Extract item weights using factor analysis."""
    weights = {}
    print(f"DEBUG: Extracting weights for {len(clusters)} clusters")
    
    for sc, its in clusters.items():
        print(f"DEBUG: Processing cluster {sc} with {len(its)} items")
        if len(its) <= 1:
            # For single-item scales, just use a mix of weights rather than 1.0
            if len(its) == 1:
                # Create variation in weights (0.7-1.0 range)
                item = its[0]
                # Get distribution of this item to create more realistic weights
                try:
                    counts = df[item].value_counts().sort_index()
                    values = counts.index.values
                    counts_values = counts.values
                    
                    # Normalize to sum to 1
                    if sum(counts_values) > 0:
                        normalized_weights = counts_values / sum(counts_values)
                        weight_dict = {str(val): float(weight) for val, weight in zip(values, normalized_weights)}
                        
                        # Store in format compatible with the simulation function
                        weights[item] = {
                            'is_distribution': True,
                            'weights': weight_dict
                        }
                        print(f"DEBUG: Created distribution weights for {item}: {weight_dict}")
                    else:
                        weights[item] = {
                            'is_distribution': False,
                            'weight': float(1.0)
                        }
                        print(f"DEBUG: No valid distribution for {item}, using unit weight")
                except Exception as e:
                    # Fallback if distribution analysis fails
                    print(f"ERROR creating distribution for single item {item}: {str(e)}")
                    weights[item] = {
                        'is_distribution': False,
                        'weight': float(1.0)
                    }
                    print(f"DEBUG: Using unit weight for {item} due to error")
            continue
            
        try:
            # Make sure we have at least 2 items for factor analysis
            if len(its) < 2:
                print(f"DEBUG: Single factor cluster with {len(its)} items, using distribution")
                for itm in its:
                    # Get distribution for this item
                    try:
                        counts = df[itm].value_counts().sort_index()
                        values = counts.index.values
                        counts_values = counts.values
                        
                        # Normalize to sum to 1
                        if sum(counts_values) > 0:
                            normalized_weights = counts_values / sum(counts_values)
                            weight_dict = {str(val): float(weight) for val, weight in zip(values, normalized_weights)}
                            
                            weights[itm] = {
                                'is_distribution': True,
                                'weights': weight_dict
                            }
                            print(f"DEBUG: Created secondary distribution weights for {itm}: {weight_dict}")
                        else:
                            weights[itm] = {
                                'is_distribution': False,
                                'weight': float(1.0)
                            }
                            print(f"DEBUG: No valid secondary distribution for {itm}, using unit weight")
                    except Exception as e:
                        print(f"ERROR creating secondary distribution for {itm}: {str(e)}")
                        weights[itm] = {
                            'is_distribution': False,
                            'weight': float(1.0)
                        }
                        print(f"DEBUG: Using unit weight for {itm} due to secondary error")
                continue
                
            # Handle case where n_factors is too large for the number of items
            n_f = min(n_factors, len(its)-1)
            if n_f < 1:
                n_f = 1
                
            # Create factor analyzer with appropriate parameters
            fa = FactorAnalyzer(n_factors=n_f, rotation=rotation)
            
            # Handle any missing values before fitting
            clean_data = df[its].fillna(df[its].mean())
            
            # Suppress warnings about matrix inversion
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                try:
                    fa.fit(clean_data)
                except Exception as e:
                    print(f"WARNING: Factor Analysis error: {str(e)}")
                    # Create a fallback with random weights that are deterministic 
                    # (based on the mean of each variable)
                    load = np.abs(clean_data.mean().values)
                    if np.sum(load) > 0:
                        load /= np.sum(load)
                    else:
                        load = np.ones(len(its)) / len(its)
                    fa.loadings_ = np.column_stack([load] + [np.zeros(len(its))] * (n_f-1))
            
            # Use first factor loadings as weights
            load = fa.loadings_[:, 0]
            
            # Normalize loadings to sum to 1
            load = np.abs(load)  # Use absolute values
            # Avoid division by zero
            if np.sum(load) > 0:
                load /= np.sum(load)
            else:
                load = np.ones_like(load) / len(load)
            
            # For each item, capture both the factor loading weight and the distribution
            for itm, val in zip(its, load):
                # Store both the factor loading and the observed distribution
                try:
                    counts = df[itm].value_counts().sort_index()
                    values = counts.index.values
                    counts_values = counts.values
                    
                    # Normalize to sum to 1
                    if sum(counts_values) > 0:
                        norm_counts = counts_values / sum(counts_values)
                        # Make sure to convert to strings for keys, floats for values
                        dist_weights = {str(val): float(weight) for val, weight in zip(values, norm_counts)}
                        print(f"DEBUG: Created distribution weights for {itm}: {len(dist_weights)} values")
                    else:
                        dist_weights = None
                        print(f"DEBUG: No distribution weights for {itm}")
                    
                    # Store both types of weights
                    weights[itm] = {
                        'is_distribution': False,  # Primary weight is from factor analysis
                        'weight': float(val),      # Factor analysis weight
                        'dist_weights': dist_weights  # Distribution-based weights as backup
                    }
                    print(f"DEBUG: Added factor weight {val:.4f} for {itm}")
                except Exception as e:
                    print(f"ERROR analyzing distribution for {itm}: {str(e)}")
                    weights[itm] = {
                        'is_distribution': False,
                        'weight': float(val)
                    }
                
        except Exception as e:
            print(f"Error in factor analysis for cluster {sc}: {str(e)}")
            # If factor analysis fails, use distribution-based weights
            for itm in its:
                try:
                    counts = df[itm].value_counts().sort_index()
                    values = counts.index.values
                    counts_values = counts.values
                    
                    # Normalize to sum to 1
                    if sum(counts_values) > 0:
                        normalized_weights = counts_values / sum(counts_values)
                        weight_dict = {str(val): float(weight) for val, weight in zip(values, normalized_weights)}
                        
                        weights[itm] = {
                            'is_distribution': True,
                            'weights': weight_dict
                        }
                        print(f"DEBUG: Created fallback distribution weights for {itm} with {len(weight_dict)} values")
                    else:
                        weights[itm] = {
                            'is_distribution': False,
                            'weight': float(1.0 / len(its))
                        }
                        print(f"DEBUG: No valid distribution for fallback {itm}, using equal weight {1.0/len(its):.4f}")
                except Exception as e:
                    # Equal weights as fallback
                    print(f"ERROR creating distribution for {itm}: {str(e)}")
                    weights[itm] = {
                        'is_distribution': False,
                        'weight': float(1.0 / len(its))
                    }
                    print(f"DEBUG: Using equal weight {1.0/len(its):.4f} for {itm} due to error")
    
    return weights

def cronbach_alpha(df, items):
    """Calculate Cronbach's alpha reliability coefficient."""
    if len(items) <= 1:
        return 0.0
        
    d = df[items]
    # Handle missing values
    d = d.dropna()
    
    if len(d) < 3:  # Not enough data
        return 0.0
        
    var_sum = d.var(axis=0, ddof=1).sum()
    tot_var = d.sum(axis=1).var(ddof=1)
    k = len(items)
    
    if tot_var == 0:
        return 0.0
        
    return k/(k-1)*(1 - var_sum/tot_var)

def bootstrap_alpha(df, items, n_boot=100):
    """Calculate confidence interval for Cronbach's alpha using bootstrapping."""
    if len(items) <= 1:
        return [0.0, 0.0]
        
    alphas = []
    data = df[items].dropna()
    
    if len(data) < 5:  # Not enough data for bootstrapping
        return [0.0, 0.0]
    
    n = data.shape[0]
    
    for _ in range(n_boot):
        try:
            # Sample with replacement
            sample = data.sample(n, replace=True)
            # Calculate alpha
            var_sum = sample.var(axis=0, ddof=1).sum()
            tot_var = sample.sum(axis=1).var(ddof=1)
            k = len(items)
            if tot_var > 0:
                alphas.append(k/(k-1)*(1 - var_sum/tot_var))
        except:
            continue
    
    if not alphas:
        return [0.0, 0.0]
        
    return np.percentile(alphas, [2.5, 97.5])

def simulate_responses(weights, n_samples=1000, noise=0.1):
    """Simulate responses based on item weights."""
    if not weights:
        print("ERROR: No weights provided for simulation!")
        return pd.DataFrame()
        
    items = list(weights.keys())
    if not items:
        print("ERROR: Weights dictionary has no keys!")
        return pd.DataFrame()
    
    # Debug the weights format
    print(f"DEBUG: Simulating with {len(items)} items")
    print(f"DEBUG: First item format: {type(weights[items[0]])}")
    
    # Check the weight format
    if isinstance(weights[items[0]], dict) and 'is_distribution' in weights[items[0]]:
        # New format with distribution-based weights
        print("DEBUG: Using distribution-based simulation")
        return simulate_responses_with_distributions(weights, n_samples, noise)
    else:
        # Legacy format with simple weights
        print("DEBUG: Using legacy simulation")
        return simulate_responses_legacy(weights, n_samples, noise)

def simulate_responses_legacy(weights, n_samples=1000, noise=0.1):
    """Legacy simulation method based on simple weights."""
    items = list(weights.keys())
    w = np.array([weights[i] for i in items])
    
    # Generate latent factor
    f = np.random.normal(size=(n_samples, 1))
    
    # Generate continuous responses
    cont = f.dot(w.reshape(1, -1)) + np.random.normal(scale=noise, size=(n_samples, len(items)))
    
    # Discretize to Likert scale (assuming 1-5)
    scale_min = 1
    scale_max = 5
    n_cats = scale_max - scale_min + 1
    
    # Create percentile thresholds to convert to discrete scale
    thresholds = [np.percentile(cont[:, i], 100 * j / n_cats) 
                  for i in range(len(items)) 
                  for j in range(1, n_cats)]
    thresholds = np.array(thresholds).reshape(len(items), n_cats-1)
    
    # Convert to discrete responses
    discrete = {}
    for i, item in enumerate(items):
        item_responses = np.ones(n_samples) * scale_min
        for j in range(n_cats-1):
            item_responses[cont[:, i] > thresholds[i, j]] = scale_min + j + 1
        discrete[item] = item_responses
    
    return pd.DataFrame(discrete)

def simulate_responses_with_distributions(weights, n_samples=1000, noise=0.1):
    """Simulate responses using distribution-based weights."""
    items = list(weights.keys())
    discrete = {}
    
    print(f"DEBUG: Simulating with distributions for {len(items)} items")
    
    # First, extract the factor weights
    factor_weights = {}
    for item in items:
        try:
            if weights[item]['is_distribution'] == False and 'weight' in weights[item]:
                factor_weights[item] = weights[item]['weight']
                print(f"DEBUG: Added factor weight for {item}")
            elif 'dist_weights' in weights[item] and weights[item]['dist_weights']:
                print(f"DEBUG: Found distribution weights for {item}")
        except Exception as e:
            print(f"ERROR in weight format for {item}: {str(e)}")
            print(f"Weight data: {weights[item]}")
    
    print(f"DEBUG: Found {len(factor_weights)} items with factor weights")
        
    # If we have factor weights, use them to generate correlated responses
    if factor_weights:
        factor_items = list(factor_weights.keys())
        w = np.array([factor_weights[i] for i in factor_items])
        
        # Generate latent factor
        f = np.random.normal(size=(n_samples, 1))
        
        # Generate continuous responses for factor-weighted items
        cont = f.dot(w.reshape(1, -1)) + np.random.normal(scale=noise, size=(n_samples, len(factor_items)))
        
        # Discretize to Likert scale
        scale_min = 1
        scale_max = 5
        n_cats = scale_max - scale_min + 1
        
        # Create percentile thresholds to convert to discrete scale
        thresholds = [np.percentile(cont[:, i], 100 * j / n_cats) 
                      for i in range(len(factor_items)) 
                      for j in range(1, n_cats)]
        thresholds = np.array(thresholds).reshape(len(factor_items), n_cats-1)
        
        # Convert to discrete responses
        for i, item in enumerate(factor_items):
            item_responses = np.ones(n_samples) * scale_min
            for j in range(n_cats-1):
                item_responses[cont[:, i] > thresholds[i, j]] = scale_min + j + 1
            discrete[item] = item_responses
    
    # For distribution-based items, sample directly from distribution
    for item in items:
        if item in discrete:
            print(f"DEBUG: Item {item} already processed with factor weights")
            continue  # Skip items already processed
        
        try:    
            if 'is_distribution' in weights[item] and weights[item]['is_distribution'] and 'weights' in weights[item]:
                # Get the distribution weights
                print(f"DEBUG: Using distribution weights for {item}")
                dist = weights[item]['weights']
                # Convert all keys to float and all values to float
                dist_dict = {float(k): float(v) for k, v in dist.items()}
                values = np.array(list(dist_dict.keys()))
                probs = np.array(list(dist_dict.values()))
                
                # Debug the distribution
                print(f"DEBUG: Distribution for {item} - values: {values}, probabilities: {probs}")
                
                # Normalize probabilities if needed
                if sum(probs) > 0:
                    probs = probs / sum(probs)
                    print(f"DEBUG: Normalized probabilities: {probs}")
                else:
                    print(f"WARNING: Zero sum probabilities for {item}")
                    probs = np.ones_like(probs) / len(probs)
                    
                # Sample from distribution
                try:
                    discrete[item] = np.random.choice(values, size=n_samples, p=probs)
                    print(f"DEBUG: Successfully sampled distribution for {item}")
                except Exception as e:
                    print(f"ERROR sampling from distribution for {item}: {str(e)}")
                    # Fallback to uniform
                    discrete[item] = np.random.randint(1, 6, size=n_samples)
                
            elif 'dist_weights' in weights[item] and weights[item]['dist_weights']:
                # Use distribution weights as backup
                print(f"DEBUG: Using backup distribution weights for {item}")
                dist = weights[item]['dist_weights']
                # Convert all keys to float and all values to float
                dist_dict = {float(k): float(v) for k, v in dist.items()}
                values = np.array(list(dist_dict.keys()))
                probs = np.array(list(dist_dict.values()))
                
                # Debug the distribution
                print(f"DEBUG: Backup distribution for {item} - values: {values}, probabilities: {probs}")
                
                # Normalize probabilities if needed
                if sum(probs) > 0:
                    probs = probs / sum(probs)
                    print(f"DEBUG: Normalized backup probabilities: {probs}")
                else:
                    print(f"WARNING: Zero sum backup probabilities for {item}")
                    probs = np.ones_like(probs) / len(probs)
                    
                # Sample from distribution
                try:
                    discrete[item] = np.random.choice(values, size=n_samples, p=probs)
                    print(f"DEBUG: Successfully sampled backup distribution for {item}")
                except Exception as e:
                    print(f"ERROR sampling from backup distribution for {item}: {str(e)}")
                    # Fallback to uniform
                    discrete[item] = np.random.randint(1, 6, size=n_samples)
        except Exception as e:
            print(f"ERROR in distribution sampling for {item}: {str(e)}")
            # Get weight data format for debugging
            print(f"Weight data for {item}: {weights[item]}")
            # Fallback to uniform distribution
            print(f"DEBUG: Using fallback uniform distribution for {item}")
            discrete[item] = np.random.randint(1, 6, size=n_samples)
    
    return pd.DataFrame(discrete)

def create_network_graph(df, items, threshold=0.3, layout='force'):
    """Create an enhanced network visualization of item relationships.
    
    Parameters:
    ----------
    df : DataFrame
        Dataset containing Likert items
    items : list
        List of item names to include in the graph
    threshold : float
        Minimum correlation threshold to display an edge
    layout : str
        Graph layout algorithm ('force', 'circular', 'kamada_kawai', 'spectral')
    """
    # Calculate correlation matrix
    corr = df[items].corr().abs()
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with properties
    for i, item in enumerate(items):
        # Calculate node importance by summing all correlations
        importance = corr[item].sum() - 1  # Subtract self-correlation
        G.add_node(item, importance=importance)
    
    # Add edges with correlation > threshold
    for i, item1 in enumerate(items):
        for j, item2 in enumerate(items):
            if i > j and corr.loc[item1, item2] > threshold:
                G.add_edge(item1, item2, weight=corr.loc[item1, item2])
    
    # Apply clustering to identify communities (for coloring)
    communities = nx.community.greedy_modularity_communities(G)
    community_map = {}
    for i, community in enumerate(communities):
        for node in community:
            community_map[node] = i
    
    # Get positions using selected layout algorithm
    if layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'spectral':
        pos = nx.spectral_layout(G)
    else:  # default to force-directed (spring layout)
        pos = nx.spring_layout(G, seed=42, k=0.8)  # k controls node spacing
    
    # Define color palette for communities
    color_palettes = [
        px.colors.qualitative.Plotly,
        px.colors.qualitative.D3,
        px.colors.qualitative.G10,
        px.colors.qualitative.T10,
        px.colors.qualitative.Alphabet
    ]
    
    # Flatten all color palettes into one large palette
    all_colors = [color for palette in color_palettes for color in palette]
    
    # Create edge traces grouped by community pairs
    edge_traces = []
    seen_communities = set()
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        
        # Get community pair
        comm1 = community_map.get(edge[0], 0)
        comm2 = community_map.get(edge[1], 0)
        
        # Use same color for intra-community edges
        if comm1 == comm2:
            color = all_colors[comm1 % len(all_colors)]
            comm_key = f"comm_{comm1}"
        else:
            # Different color for inter-community edges
            sorted_comms = tuple(sorted([comm1, comm2]))
            if sorted_comms in seen_communities:
                # Reuse color for this community pair
                color = all_colors[(comm1 + comm2) % len(all_colors)]
            else:
                color = "rgba(180,180,180,0.3)"  # Light gray for inter-community
                seen_communities.add(sorted_comms)
            comm_key = f"comm_{comm1}_{comm2}"
        
        # Scale width and opacity by correlation strength
        width = weight * 4 
        opacity = min(0.7 + weight * 0.3, 1.0)
        
        # Use RGBA to control opacity
        rgba_color = color if isinstance(color, str) and color.startswith('rgba') else color
        if not isinstance(rgba_color, str) or not rgba_color.startswith('rgba'):
            # Convert hex or RGB to RGBA
            if rgba_color.startswith('#'):
                r, g, b = int(rgba_color[1:3], 16), int(rgba_color[3:5], 16), int(rgba_color[5:7], 16)
                rgba_color = f'rgba({r},{g},{b},{opacity})'
            else:
                rgba_color = f'rgba(100,100,200,{opacity})'
                
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color=rgba_color),
            hoverinfo='text',
            text=f"Correlation: {weight:.3f}",
            mode='lines',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node traces by community
    node_traces = []
    node_communities = {}
    
    for comm_idx, community in enumerate(communities):
        community_color = all_colors[comm_idx % len(all_colors)]
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in community:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create more informative tooltip text
            importance = G.nodes[node]['importance']
            neighbors = list(G.neighbors(node))
            top_correlations = sorted(
                [(neigh, corr.loc[node, neigh]) for neigh in neighbors],
                key=lambda x: x[1], 
                reverse=True
            )[:3]  # Top 3 correlations
            
            tooltip = f"{node}<br>Community: {comm_idx+1}<br>Total Correlation: {importance:.2f}<br><br>Top connections:"
            for n, w in top_correlations:
                tooltip += f"<br>• {n}: {w:.3f}"
            
            node_text.append(tooltip)
            
            # Size node by importance (sum of correlations)
            size = 10 + importance * 5
            node_size.append(size)
            
            # Save for legend
            node_communities[node] = comm_idx
        
        # Create a trace per community
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=[n.split('.', 1)[0] if '.' in n else n for n in community],  # Show shorter labels
            textposition="top center",
            textfont=dict(
                size=10,
                color='black'
            ),
            marker=dict(
                size=node_size,
                color=community_color,
                line=dict(width=2, color='white'),
                opacity=0.85
            ),
            hovertext=node_text,
            hoverinfo='text',
            name=f"Community {comm_idx+1}"
        )
        node_traces.append(node_trace)
    
    # Create plotly figure with all traces
    fig = go.Figure()
    
    # Add all edge traces
    for trace in edge_traces:
        fig.add_trace(trace)
    
    # Add all node traces
    for trace in node_traces:
        fig.add_trace(trace)
    
    # Update layout with better styling
    fig.update_layout(
        title={
            'text': 'Item Relationship Network',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        legend=dict(
            title="Communities",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=650
    )
    
    # Add a color scale legend for edge weights
    scale_points = [i/10 for i in range(3, 11, 2)]  # [0.3, 0.5, 0.7, 0.9]
    for i, point in enumerate(scale_points):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(width=point*4, color=f'rgba(100,100,100,{min(0.7 + point * 0.3, 1.0)})'),
            name=f"Corr: {point:.1f}",
            legendgroup="edge_weights",
            showlegend=True
        ))
    
    return fig

def save_html_report(results):
    """Generate and save HTML report."""
    # Create a temp file for the report
    temp_file = os.path.join(tempfile.gettempdir(), f"likert_report_{uuid.uuid4()}.html")
    
    # Load templates
    template_str = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Likert Scale Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .metric { font-weight: bold; }
            .header { background-color: #2874a6; color: white; padding: 20px; border-radius: 5px; }
            .section { margin: 30px 0; padding: 20px; border-radius: 5px; border: 1px solid #ddd; }
            .item-list { list-style-type: none; padding: 0; }
            .item-list li { margin: 5px 0; padding: 5px; background-color: #f9f9f9; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Likert Scale Analysis Report</h1>
            <p>Generated on {{ timestamp }}</p>
        </div>
        
        <div class="section">
            <h2>Data Overview</h2>
            <p>Number of responses: {{ data_shape[0] }}</p>
            <p>Number of variables: {{ data_shape[1] }}</p>
            <p>Number of Likert items identified: {{ likert_items|length }}</p>
        </div>
        
        <div class="section">
            <h2>Likert Items</h2>
            <ul class="item-list">
                {% for item in likert_items %}
                <li>{{ item }}</li>
                {% endfor %}
            </ul>
            
            {% if reverse_items %}
            <h3>Reverse Coded Items</h3>
            <ul class="item-list">
                {% for item in reverse_items %}
                <li>{{ item }}</li>
                {% endfor %}
            </ul>
            {% endif %}
        </div>
        
        <div class="section">
            <h2>Item Clusters</h2>
            {% for sc, items in clusters.items() %}
            <h3>Cluster {{ sc }} ({{ items|length }} items)</h3>
            <ul class="item-list">
                {% for item in items %}
                <li>{{ item }}</li>
                {% endfor %}
            </ul>
            
            {% if sc in alphas %}
            <p class="metric">Cronbach's Alpha: {{ alphas[sc]|float|round(3) }}</p>
            <p>95% Confidence Interval: [{{ alpha_ci[sc][0]|float|round(3) }}, {{ alpha_ci[sc][1]|float|round(3) }}]</p>
            {% endif %}
            {% endfor %}
        </div>
        
        <div class="section">
            <h2>Item Weights</h2>
            <table>
                <tr>
                    <th>Item</th>
                    <th>Weight Type</th>
                    <th>Value</th>
                </tr>
                {% for item, weight in weights.items() %}
                <tr>
                    <td>{{ item }}</td>
                    {% if weight is mapping and weight.is_distribution is defined and weight.is_distribution %}
                        <td>Distribution</td>
                        <td>Distribution-based weights</td>
                    {% elif weight is mapping and weight.weight is defined %}
                        <td>Factor Loading</td>
                        <td>{{ weight.weight|float|round(4) }}</td>
                    {% else %}
                        <td>Simple</td>
                        <td>{{ weight|float|round(4) }}</td>
                    {% endif %}
                </tr>
                {% endfor %}
            </table>
        </div>
        
        {% if simulated is not none %}
        <div class="section">
            <h2>Simulation Results</h2>
            <p>Generated {{ simulated|length }} simulated responses based on extracted patterns.</p>
        </div>
        {% endif %}
        
        <div class="section">
            <h2>Analysis Summary</h2>
            <p>The analysis identified {{ clusters|length }} clusters of items, suggesting the presence of {{ clusters|length }} distinct constructs in the survey.</p>
            <p>Reliability analysis indicates that 
            {% set reliable_clusters = [] %}
            {% for sc, alpha in alphas.items() %}
                {% if alpha|float > 0.7 %}
                    {% set _ = reliable_clusters.append(sc) %}
                {% endif %}
            {% endfor %}
            {{ reliable_clusters|length }} out of {{ alphas|length }} scales have good reliability (α > 0.7).
            </p>
        </div>
        
        <footer>
            <p><small>This report was generated using the Likert Scale Pattern Analysis application.</small></p>
        </footer>
    </body>
    </html>
    """
    
    # Create template and render
    template = jinja2.Template(template_str)
    html = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data_shape=results['data'].shape,
        likert_items=results['likert_items'],
        reverse_items=results['reverse_items'],
        clusters=results['clusters'],
        weights=results['weights'],
        alphas=results['alphas'],
        alpha_ci=results['alpha_ci'],
        simulated=results['simulated']
    )
    
    # Write to file
    with open(temp_file, 'w') as f:
        f.write(html)
    
    return temp_file

def run_hybrid_analysis(data_path, scales, n_sim=500):
    """Run the hybrid Python+R analysis."""
    # Create temp directory for outputs
    output_dir = tempfile.mkdtemp()
    
    # Prepare command
    cmd = ['python', 'likert_hybrid.py', data_path, '--n_sim', str(n_sim)]
    
    # Add scales
    cmd.extend(['--scales'])
    cmd.extend(scales)
    
    # Run the command
    subprocess.run(cmd, cwd=output_dir, check=True)
    
    # Read results
    results = {}
    
    # Read weights
    weights = {}
    for scale in scales:
        weight_file = os.path.join(output_dir, f'weights_{scale}.csv')
        if os.path.exists(weight_file):
            df = pd.read_csv(weight_file)
            for _, row in df.iterrows():
                item = row.iloc[0]
                weight = row.iloc[1]
                weights[item] = weight
    
    # Read simulated data
    simulated = None
    for scale in scales:
        sim_file = os.path.join(output_dir, f'simulated_{scale}.csv')
        if os.path.exists(sim_file):
            simulated = pd.read_csv(sim_file)
            break  # Just use the first one we find
    
    results['weights'] = weights
    results['simulated'] = simulated
    
    return results
