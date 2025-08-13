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
        if corr < threshold:
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
    
    from sklearn import __version__ as sklearn_version
    
    try:
        if n_clusters:
            if float(sklearn_version.split('.')[0]) >= 1:
                model = AgglomerativeClustering(metric='precomputed', linkage='average', n_clusters=n_clusters)
            else:
                model = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=n_clusters)
        else:
            if float(sklearn_version.split('.')[0]) >= 1:
                model = AgglomerativeClustering(metric='precomputed', linkage='average', distance_threshold=1-threshold, n_clusters=None)
            else:
                model = AgglomerativeClustering(affinity='precomputed', linkage='average', distance_threshold=1-threshold, n_clusters=None)
    except Exception as e:
        model = AgglomerativeClustering(n_clusters=n_clusters if n_clusters else 3)
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
        fa = FactorAnalyzer(rotation=None)
        fa.fit(df[items])
        ev, _ = fa.get_eigenvalues()
        return sum(ev > 1)
    except:
        return 1

def extract_weights(df, clusters, n_factors=1, rotation='varimax'):
    """Extract item weights using factor analysis."""
    weights = {}
    
    for sc, its in clusters.items():
        if len(its) <= 1:
            if len(its) == 1:
                item = its[0]
                try:
                    counts = df[item].value_counts().sort_index()
                    values = counts.index.values
                    counts_values = counts.values
                    
                    if sum(counts_values) > 0:
                        normalized_weights = counts_values / sum(counts_values)
                        weight_dict = {str(val): float(weight) for val, weight in zip(values, normalized_weights)}
                        weights[item] = {'is_distribution': True, 'weights': weight_dict}
                    else:
                        weights[item] = {'is_distribution': False, 'weight': float(1.0)}
                except Exception as e:
                    weights[item] = {'is_distribution': False, 'weight': float(1.0)}
            continue
            
        try:
            if len(its) < 2:
                for itm in its:
                    try:
                        counts = df[itm].value_counts().sort_index()
                        values = counts.index.values
                        counts_values = counts.values
                        
                        if sum(counts_values) > 0:
                            normalized_weights = counts_values / sum(counts_values)
                            weight_dict = {str(val): float(weight) for val, weight in zip(values, normalized_weights)}
                            weights[itm] = {'is_distribution': True, 'weights': weight_dict}
                        else:
                            weights[itm] = {'is_distribution': False, 'weight': float(1.0)}
                    except Exception as e:
                        weights[itm] = {'is_distribution': False, 'weight': float(1.0)}
                continue
                
            n_f = min(n_factors, len(its)-1)
            if n_f < 1: n_f = 1
                
            fa = FactorAnalyzer(n_factors=n_f, rotation=rotation)
            clean_data = df[its].fillna(df[its].mean())
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                try:
                    fa.fit(clean_data)
                except Exception as e:
                    load = np.abs(clean_data.mean().values)
                    if np.sum(load) > 0:
                        load /= np.sum(load)
                    else:
                        load = np.ones(len(its)) / len(its)
                    fa.loadings_ = np.column_stack([load] + [np.zeros(len(its))] * (n_f-1))
            
            load = fa.loadings_[:, 0]
            load = np.abs(load)
            if np.sum(load) > 0:
                load /= np.sum(load)
            else:
                load = np.ones_like(load) / len(load)
            
            for itm, val in zip(its, load):
                try:
                    counts = df[itm].value_counts().sort_index()
                    values = counts.index.values
                    counts_values = counts.values
                    
                    if sum(counts_values) > 0:
                        norm_counts = counts_values / sum(counts_values)
                        dist_weights = {str(val): float(weight) for val, weight in zip(values, norm_counts)}
                    else:
                        dist_weights = None
                    
                    weights[itm] = {'is_distribution': False, 'weight': float(val), 'dist_weights': dist_weights}
                except Exception as e:
                    weights[itm] = {'is_distribution': False, 'weight': float(val)}
                
        except Exception as e:
            for itm in its:
                try:
                    counts = df[itm].value_counts().sort_index()
                    values = counts.index.values
                    counts_values = counts.values
                    
                    if sum(counts_values) > 0:
                        normalized_weights = counts_values / sum(counts_values)
                        weight_dict = {str(val): float(weight) for val, weight in zip(values, normalized_weights)}
                        weights[itm] = {'is_distribution': True, 'weights': weight_dict}
                    else:
                        weights[itm] = {'is_distribution': False, 'weight': float(1.0 / len(its))}
                except Exception as e:
                    weights[itm] = {'is_distribution': False, 'weight': float(1.0 / len(its))}
    
    return weights

def cronbach_alpha(df, items):
    """Calculate Cronbach's alpha reliability coefficient."""
    if len(items) <= 1:
        return 0.0
        
    d = df[items].dropna()
    if len(d) < 3: return 0.0
        
    var_sum = d.var(axis=0, ddof=1).sum()
    tot_var = d.sum(axis=1).var(ddof=1)
    k = len(items)
    
    if tot_var == 0: return 0.0
    return k/(k-1)*(1 - var_sum/tot_var)

def bootstrap_alpha(df, items, n_bootstrap=100):
    """Calculate confidence interval for Cronbach's alpha using bootstrapping."""
    if len(items) <= 1:
        return [0.0, 0.0]
        
    alphas = []
    data = df[items].dropna()
    
    if len(data) < 5: return [0.0, 0.0]
    
    n = data.shape[0]
    
    for _ in range(n_bootstrap):
        try:
            sample = data.sample(n, replace=True)
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
    if not weights: return pd.DataFrame()
    items = list(weights.keys())
    if not items: return pd.DataFrame()
    
    if isinstance(weights[items[0]], dict) and 'is_distribution' in weights[items[0]]:
        return simulate_responses_with_distributions(weights, n_samples, noise)
    else:
        return simulate_responses_legacy(weights, n_samples, noise)

def simulate_responses_legacy(weights, n_samples=1000, noise=0.1):
    """Legacy simulation method based on simple weights."""
    items = list(weights.keys())
    w = np.array([weights[i] for i in items])
    f = np.random.normal(size=(n_samples, 1))
    cont = f.dot(w.reshape(1, -1)) + np.random.normal(scale=noise, size=(n_samples, len(items)))
    
    scale_min, scale_max, n_cats = 1, 5, 5
    thresholds = np.array([np.percentile(cont[:, i], 100 * j / n_cats) for i in range(len(items)) for j in range(1, n_cats)]).reshape(len(items), n_cats-1)
    
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
    
    factor_weights = {item: data['weight'] for item, data in weights.items() if not data.get('is_distribution')}
        
    if factor_weights:
        factor_items = list(factor_weights.keys())
        w = np.array([factor_weights[i] for i in factor_items])
        f = np.random.normal(size=(n_samples, 1))
        cont = f.dot(w.reshape(1, -1)) + np.random.normal(scale=noise, size=(n_samples, len(factor_items)))
        
        scale_min, scale_max, n_cats = 1, 5, 5
        thresholds = np.array([np.percentile(cont[:, i], 100 * j / n_cats) for i in range(len(factor_items)) for j in range(1, n_cats)]).reshape(len(factor_items), n_cats-1)
        
        for i, item in enumerate(factor_items):
            item_responses = np.ones(n_samples) * scale_min
            for j in range(n_cats-1):
                item_responses[cont[:, i] > thresholds[i, j]] = scale_min + j + 1
            discrete[item] = item_responses
    
    for item in items:
        if item in discrete: continue
        
        try:
            dist_data = None
            if weights[item].get('is_distribution') and 'weights' in weights[item]:
                dist_data = weights[item]['weights']
            elif 'dist_weights' in weights[item] and weights[item]['dist_weights']:
                dist_data = weights[item]['dist_weights']
            
            if dist_data:
                dist_dict = {float(k): float(v) for k, v in dist_data.items()}
                values, probs = np.array(list(dist_dict.keys())), np.array(list(dist_dict.values()))
                
                if sum(probs) > 0:
                    probs /= sum(probs)
                else:
                    probs = np.ones_like(probs) / len(probs)
                
                discrete[item] = np.random.choice(values, size=n_samples, p=probs)
            else:
                 discrete[item] = np.random.randint(1, 6, size=n_samples)
        except Exception as e:
            discrete[item] = np.random.randint(1, 6, size=n_samples)
    
    return pd.DataFrame(discrete)

def create_network_graph(df, items, threshold=0.3, layout='force'):
    """Create an enhanced network visualization of item relationships."""
    corr = df[items].corr().abs()
    G = nx.Graph()
    
    for item in items:
        G.add_node(item, importance=corr[item].sum() - 1)
    
    for i, item1 in enumerate(items):
        for j, item2 in enumerate(items):
            if i > j and corr.loc[item1, item2] > threshold:
                G.add_edge(item1, item2, weight=corr.loc[item1, item2])
    
    communities = nx.community.greedy_modularity_communities(G)
    community_map = {node: i for i, comm in enumerate(communities) for node in comm}
    
    layouts = {
        'circular': nx.circular_layout(G),
        'kamada_kawai': nx.kamada_kawai_layout(G),
        'spectral': nx.spectral_layout(G),
        'force': nx.spring_layout(G, seed=42, k=0.8)
    }
    pos = layouts.get(layout, layouts['force'])
    
    all_colors = [color for palette in [px.colors.qualitative.Plotly, px.colors.qualitative.D3, px.colors.qualitative.G10] for color in palette]
    
    edge_traces, node_traces = [], []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = edge[2]['weight']
        edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], line=dict(width=weight*4, color='rgba(180,180,180,0.5)'), hoverinfo='text', text=f"Correlation: {weight:.3f}", mode='lines'))
    
    for i, community in enumerate(communities):
        node_x, node_y, node_text, node_size = [], [], [], []
        for node in community:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            importance = G.nodes[node]['importance']
            node_text.append(f"{node}<br>Community: {i+1}<br>Importance: {importance:.2f}")
            node_size.append(10 + importance * 5)
        
        node_traces.append(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[n.split('.', 1)[0] for n in community], textposition="top center", textfont=dict(size=10, color='black'), marker=dict(size=node_size, color=all_colors[i % len(all_colors)]), hovertext=node_text, hoverinfo='text', name=f"Community {i+1}"))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(title='Item Relationship Network', showlegend=True, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor='white', height=650)
    
    return fig

def save_html_report(results):
    """Generate and save HTML report."""
    temp_file = os.path.join(tempfile.gettempdir(), f"likert_report_{uuid.uuid4()}.html")
    template_str = """
    <!DOCTYPE html><html><head><title>Likert Scale Analysis Report</title>
    <style>body{font-family:Arial,sans-serif;max-width:1200px;margin:0 auto;padding:20px}h1,h2,h3{color:#333}table{border-collapse:collapse;width:100%;margin:20px 0}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background-color:#f2f2f2}.header{background-color:#2874a6;color:#fff;padding:20px;border-radius:5px}.section{margin:30px 0;padding:20px;border:1px solid #ddd;border-radius:5px}</style>
    </head><body><div class="header"><h1>Likert Scale Analysis Report</h1><p>Generated on {{ timestamp }}</p></div>
    <div class="section"><h2>Data Overview</h2><p>Responses: {{ data_shape[0] }} | Variables: {{ data_shape[1] }} | Likert Items: {{ likert_items|length }}</p></div>
    <div class="section"><h2>Item Clusters & Reliability</h2>
    {% for sc, items in clusters.items() %}<h3>Cluster {{ sc }} ({{ items|length }} items)</h3><ul>{% for item in items %}<li>{{ item }}</li>{% endfor %}</ul>
    {% if sc in alphas %}<p><b>Cronbach's Alpha:</b> {{ alphas[sc]|float|round(3) }} | <b>95% CI:</b> [{{ alpha_ci[sc][0]|float|round(3) }}, {{ alpha_ci[sc][1]|float|round(3) }}]</p>{% endif %}{% endfor %}</div>
    <div class="section"><h2>Item Weights</h2><table><tr><th>Item</th><th>Weight Type</th><th>Value</th></tr>
    {% for item, data in weights.items() %}<tr><td>{{ item }}</td><td>{{ 'Distribution' if data.is_distribution else 'Factor Loading' }}</td><td>{{ data.weight|float|round(4) if not data.is_distribution else 'N/A' }}</td></tr>{% endfor %}</table></div>
    {% if simulated is not none %}<div class="section"><h2>Simulation</h2><p>Generated {{ simulated|length }} simulated responses.</p></div>{% endif %}
    </body></html>"""
    
    template = jinja2.Template(template_str)
    
    # Process weights for the template
    processed_weights = {}
    for item, data in results['weights'].items():
        is_dist = data.get('is_distribution', False)
        weight_val = data.get('weight', None)
        processed_weights[item] = {'is_distribution': is_dist, 'weight': weight_val}

    html = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data_shape=results['data'].shape,
        likert_items=results['likert_items'],
        clusters=results['clusters'],
        weights=processed_weights,
        alphas=results['alphas'],
        alpha_ci=results['alpha_ci'],
        simulated=results['simulated']
    )
    
    with open(temp_file, 'w') as f:
        f.write(html)
    return temp_file

def run_hybrid_analysis(data_path, scales, n_sim=500):
    """Run the hybrid Python+R analysis."""
    output_dir = tempfile.mkdtemp()
    cmd = ['python', 'likert_hybrid.py', data_path, '--n_sim', str(n_sim), '--scales'] + scales
    subprocess.run(cmd, cwd=output_dir, check=True)
    
    weights = {}
    for scale in scales:
        weight_file = os.path.join(output_dir, f'weights_{scale}.csv')
        if os.path.exists(weight_file):
            df = pd.read_csv(weight_file)
            for _, row in df.iterrows():
                weights[row.iloc[0]] = row.iloc[1]
    
    simulated = None
    for scale in scales:
        sim_file = os.path.join(output_dir, f'simulated_{scale}.csv')
        if os.path.exists(sim_file):
            simulated = pd.read_csv(sim_file)
            break
    
    return {'weights': weights, 'simulated': simulated}
