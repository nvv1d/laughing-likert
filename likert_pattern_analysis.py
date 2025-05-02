import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from scipy.stats import spearmanr
from statsmodels.imputation.mice import MICEData
from sklearn.linear_model import LogisticRegression
from semopy import Model, Optimizer
from jinja2 import Environment, FileSystemLoader
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
from plotly.offline import plot
import networkx as nx
import pingouin as pg
import warnings
import os
import subprocess

# Activate pandas <-> R data frame conversion
pandas2ri.activate()

# Load R's mirt package for IRT (if available)
try:
    mirt = importr('mirt')
except Exception:
    warnings.warn('R package mirt not found; IRT via rpy2 disabled')

def load_data(filepath):
    if filepath.endswith(('.xls', '.xlsx')):
        return pd.read_excel(filepath)
    return pd.read_csv(filepath)

def identify_likert_columns(df, min_cats=4, max_cats=7):
    likert = []
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            uniq = df[c].dropna().unique()
            if min_cats <= len(uniq) <= max_cats:
                likert.append(c)
    return likert

def impute_missing(df):
    mice = MICEData(df)
    for _ in range(5):
        mice.update_all()
    return mice.data

def detect_reverse_items(df, items, threshold=-0.2):
    reverse = []
    total = df[items].sum(axis=1)
    for item in items:
        corr, _ = spearmanr(df[item], total - df[item])
        if corr < threshold:
            reverse.append(item)
    return reverse

def reverse_code(df, items, scale_min=1, scale_max=5):
    return df.assign(**{item: scale_max + scale_min - df[item] for item in items})

def check_sampling(df, items):
    chi2, p = calculate_bartlett_sphericity(df[items])
    _, kmo_model = calculate_kmo(df[items])
    return {'bartlett_chi2': chi2, 'bartlett_p': p, 'kmo': kmo_model}

def bootstrap_alpha(df, items, n_boot=1000):
    alphas = []
    data = df[items]
    n = data.shape[0]
    for _ in range(n_boot):
        sample = data.sample(n, replace=True)
        var_sum = sample.var(axis=0, ddof=1).sum()
        tot_var = sample.sum(axis=1).var(ddof=1)
        k = len(items)
        alphas.append(k/(k-1)*(1 - var_sum/tot_var))
    return np.percentile(alphas, [2.5, 97.5])

def cluster_items(df, items, n_clusters=None, threshold=0.7):
    corr = df[items].corr().abs()
    dist = 1 - corr
    if n_clusters:
        model = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=n_clusters)
    else:
        model = AgglomerativeClustering(affinity='precomputed', linkage='average',
                                        distance_threshold=1-threshold, n_clusters=None)
    labels = model.fit_predict(dist)
    clusters = {}
    for it, lab in zip(items, labels):
        clusters.setdefault(lab, []).append(it)
    return clusters

def determine_factors(df, items, max_f=5):
    fa = FactorAnalyzer(n_factors=max_f, rotation=None)
    fa.fit(df[items])
    ev, _ = fa.get_eigenvalues()
    rand = np.median([np.linalg.eigvals(
        np.corrcoef(np.random.permutation(df[items].values).T)
    ) for _ in range(100)], axis=0)
    return int((ev > rand).sum())

def extract_weights(df, clusters, n_factors=1, rotation='varimax'):
    weights = {}
    for sc, its in clusters.items():
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation)
        fa.fit(df[its])
        load = fa.loadings_[:, 0]
        load /= np.sum(np.abs(load))
        for itm, val in zip(its, load):
            weights[itm] = val
    return weights

def cronbach_alpha(df, items):
    d = df[items]
    var_sum = d.var(axis=0, ddof=1).sum()
    tot = d.sum(axis=1).var(ddof=1)
    k = len(items)
    return k/(k-1)*(1 - var_sum/tot)

def irt_rpy2(df, items):
    try:
        r_df = pandas2ri.py2rpy(df[items])
        mod = mirt.mirt(r_df, 1, itemtype='graded')
        return mod
    except:
        return None

def simulate_responses(weights, n_samples=1000, noise=0.1):
    items = list(weights)
    w = np.array([weights[i] for i in items])
    f = np.random.normal(size=(n_samples, 1))
    cont = f.dot(w.reshape(1, -1)) + np.random.normal(scale=noise, size=(n_samples, len(items)))
    thr = np.percentile(cont, [20, 40, 60, 80], axis=0)
    return pd.DataFrame({itm: np.digitize(cont[:, i], thr[:, i]) + 1
                         for i, itm in enumerate(items)})

def confirmatory_sem(df, model_desc):
    m = Model(model_desc)
    opt = Optimizer(m)
    opt.optimize(df)
    return m.inspect()

def generate_report(results, template_dir='templates', out_file='report.html'):
    env = Environment(loader=FileSystemLoader(template_dir))
    tpl = env.get_template('report_template.html')
    html = tpl.render(**results)
    with open(out_file, 'w') as f:
        f.write(html)
    print(f'Report generated at {out_file}')

def export_pdf(html_file, pdf_file='report.pdf'):
    subprocess.run(['wkhtmltopdf', html_file, pdf_file], check=True)
    print(f'PDF generated at {pdf_file}')

def export_excel(results, excel_file='analysis_output.xlsx'):
    with pd.ExcelWriter(excel_file) as writer:
        pd.DataFrame([results['sampling']]).to_excel(writer, sheet_name='Sampling', index=False)
        for sc, its in results['clusters'].items():
            pd.Series(results['weights'], name='Weight').loc[its].to_excel(writer, sheet_name=f'Scale_{sc}')
        pd.DataFrame(results['alphas'], index=['alpha']).to_excel(writer, sheet_name='Reliability')
    print(f'Excel output at {excel_file}')

def generate_dashboard(df, clusters, weights):
    plots = {}
    for sc, its in clusters.items():
        fig = px.bar(x=its, y=[weights[i] for i in its], title=f'Scale {sc} Loadings')
        plots[f'scale_{sc}_loadings'] = fig
        df['score'] = df[its].sum(axis=1)
        fig2 = px.histogram(df, x='score', nbins=20, title=f'Score Dist {sc}')
        plots[f'score_{sc}_dist'] = fig2
    with open('dashboard.html', 'w') as f:
        for fig in plots.values():
            f.write(plot(fig, include_plotlyjs='cdn'))

def main(filepath, group=None, sem_model=None, n_clusters=None, n_factors=None,
         bifactor_dims=2, n_sim=1000, do_pdf=False, do_excel=False, do_dash=False):
    df = load_data(filepath)
    df_imp = impute_missing(df)
    likert = identify_likert_columns(df_imp)
    rev = detect_reverse_items(df_imp, likert)
    df_clean = reverse_code(df_imp, rev)

    sampling = check_sampling(df_clean, likert)
    clusters = cluster_items(df_clean, likert, n_clusters)
    n_fact = n_factors or determine_factors(df_clean, likert)
    weights = extract_weights(df_clean, clusters, n_fact)

    alphas = {sc: cronbach_alpha(df_clean, its) for sc, its in clusters.items()}
    ci = {sc: bootstrap_alpha(df_clean, its) for sc, its in clusters.items()}

    irt_model = irt_rpy2(df_clean, likert)
    sim_classic = simulate_responses(weights, n_sim)

    sem_results = confirmatory_sem(df_clean, sem_model) if sem_model else None

    results = {
        'sampling': sampling,
        'clusters': clusters,
        'n_factors': n_fact,
        'weights': weights,
        'alphas': alphas,
        'alpha_ci': ci,
        'irt_model': irt_model,
        'sem_results': sem_results
    }

    generate_report(results)
    if do_pdf:
        export_pdf('report.html')
    if do_excel:
        export_excel(results)
    if do_dash:
        generate_dashboard(df_clean, clusters, weights)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('--group')
    parser.add_argument('--sem_model')
    parser.add_argument('--n_clusters', type=int)
    parser.add_argument('--n_factors', type=int)
    parser.add_argument('--bifactor_dims', type=int, default=2)
    parser.add_argument('--n_sim', type=int, default=1000)
    parser.add_argument('--do_pdf', action='store_true')
    parser.add_argument('--do_excel', action='store_true')
    parser.add_argument('--do_dash', action='store_true')
    args = parser.parse_args()
    main(**vars(args))
