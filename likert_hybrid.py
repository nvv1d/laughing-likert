import pandas as pd
import subprocess
from rpy2.robjects import pandas2ri
pandas2ri.activate()

# Path to your R pipeline script
R_SCRIPT = 'likert_analysis.R'

def load_and_clean(filepath):
    df = pd.read_csv(filepath)
    # Insert your Python cleaning/imputation/reverse-code routines here if needed
    return df

def export_for_r(df, out_csv='temp_clean.csv'):
    df.to_csv(out_csv, index=False)
    return out_csv

def run_r_pipeline(clean_csv, group=None, n_sim=1000):
    cmd = ['Rscript', R_SCRIPT, clean_csv]
    if group:
        cmd.append(group)
    cmd.append(str(n_sim))
    subprocess.run(cmd, check=True)

def import_r_results(scales):
    weights = {}
    sims = {}
    for sc in scales:
        weights[sc] = pd.read_csv(f'weights_{sc}.csv', index_col=0)
        sims[sc]    = pd.read_csv(f'simulated_{sc}.csv')
    return weights, sims

def generate_report_py(weights, sims):
    import plotly.express as px
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader('templates'))
    tpl = env.get_template('py_report_template.html')
    html = tpl.render(weights=weights, sims=sims)
    with open('py_report.html', 'w') as f:
        f.write(html)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Input CSV survey data')
    parser.add_argument('--group', help='Grouping variable for DIF', default=None)
    parser.add_argument('--n_sim', type=int, default=500)
    parser.add_argument('--scales', nargs='+', required=True,
                        help='List of scale prefixes, e.g. TCR WE EMP ALT')
    args = parser.parse_args()

    # 1. Load & clean
    df_clean = load_and_clean(args.data)

    # 2. Export for R
    tmp_csv = export_for_r(df_clean)

    # 3. Run R psychometrics
    run_r_pipeline(tmp_csv, args.group, args.n_sim)

    # 4. Import R outputs
    weights, sims = import_r_results(args.scales)

    # 5. Generate Python report
    generate_report_py(weights, sims)

    print('Hybrid pipeline complete. Reports generated.')
