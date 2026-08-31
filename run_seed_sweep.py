import os
import subprocess
import pandas as pd
import numpy as np
import time
import argparse

def check_collapse(all_df):
    target = np.log(10)
    if 'Test_Loss' not in all_df.columns:
        return False
        
    losses = all_df['Test_Loss'].dropna().values
    consec = 0
    for loss in losses:
        if abs(loss - target) <= 0.0005:
            consec += 1
            if consec >= 3:
                return True
        else:
            consec = 0
    return False

def run_experiment(script_name, seed):
    print(f"\n================ Running {script_name} with seed {seed} ================")
    cmd = f"python {script_name} --attack sign_flip --epochs 100 --seed {seed}"
    subprocess.run(cmd, shell=True, check=True)

def analyze_run(sac_file, all_file):
    df_sac = pd.read_csv(sac_file)
    df_all = pd.read_csv(all_file)
    
    # Filter for sign_flip
    sf_sac = df_sac[df_sac['Attack'] == 'sign_flip']
    sf_all = df_all[df_all['Attack'] == 'sign_flip']
    
    sf_sac['Entropy_Coeff'] = pd.to_numeric(sf_sac['Entropy_Coeff'], errors='coerce')
    sf_all['Test_Accuracy'] = pd.to_numeric(sf_all['Test_Accuracy'], errors='coerce')
    
    ec = sf_sac['Entropy_Coeff']
    entropy_delta = ec.iloc[-1] - ec.iloc[0]
    
    final_acc = sf_all['Test_Accuracy'].iloc[-1]
    
    last10 = sf_all.tail(10)['Test_Accuracy']
    last10_mean = last10.mean()
    last10_std = last10.std()
    
    collapsed = check_collapse(df_all)
    
    return {
        'final_acc': final_acc,
        'last10_mean': last10_mean,
        'last10_std': last10_std,
        'entropy_delta': entropy_delta,
        'collapsed': collapsed
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', required=True, help="List of seeds to run")
    parser.add_argument('--label', type=str, required=True, help="Label for this sweep run")
    args = parser.parse_args()

    seeds = args.seeds

    results = {
        'Baseline': [],
        'Ablation B (Target Entropy = -1.0)': []
    }

    # Run Baseline
    for seed in seeds:
        run_experiment('federated_main_sac.py', seed)
        res = analyze_run('results/sac_results_mnist_LUP_SAC_0.5.csv', 'results/all_results_mnist_LUP_SAC_0.5.csv')
        res['seed'] = seed
        results['Baseline'].append(res)

    # Run Ablation B
    for seed in seeds:
        run_experiment('federated_main_sac_ablation_b.py', seed)
        res = analyze_run('results/ablation_b_sac_results_mnist_LUP_SAC_0.5.csv', 'results/ablation_b_all_results_mnist_LUP_SAC_0.5.csv')
        res['seed'] = seed
        results['Ablation B (Target Entropy = -1.0)'].append(res)

    print(f"\n\n================ FINAL RESULTS FOR {args.label} ================")
    for condition, runs in results.items():
        final_accs = [r['final_acc'] for r in runs]
        last10_means = [r['last10_mean'] for r in runs]
        last10_stds = [r['last10_std'] for r in runs]
        
        print(f"--- {condition} ---")
        for r in runs:
            col_str = "YES" if r['collapsed'] else "NO"
            print(f"  Seed {r['seed']} | Final Acc: {r['final_acc']:.2f}% | L10 Mean: {r['last10_mean']:.2f}% | L10 Std: {r['last10_std']:.2f} | Collapsed: {col_str} | Entropy Delta: {r['entropy_delta']:.6f}")
        
        print(f"\n  [AGGREGATE]")
        print(f"  Final Accuracy:      {np.mean(final_accs):.2f}% ± {np.std(final_accs):.2f}")
        print(f"  Last-10 Mean Acc:    {np.mean(last10_means):.2f}% ± {np.std(last10_means):.2f}")
        print(f"  Last-10 Std Acc:     {np.mean(last10_stds):.2f} ± {np.std(last10_stds):.2f}\n")

if __name__ == '__main__':
    main()
