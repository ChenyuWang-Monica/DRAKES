import pandas as pd


methods_name_dict = {
    # 'cfg': 'original_cfg_10.0',
    'ours': 'original_new_10_0.5',
    'pretrain': 'original_old_10_0.5',
    'zeroalpha': 'original_zero_alpha_10_0.5',
    'smc': 'smc_old_10_0.03',
    'tds': 'tds_old_1000.0_0.03',
    'cg': 'cg_old_1000.0_0.5',
}

# Load the evaluation results
for method, filename in methods_name_dict.items():
    df_list = []
    for i in range(3):
        eval_results = pd.read_csv(f'./eval_results/{filename}_{i}_results_summary.csv', index_col=0)
        df_list.append(eval_results)
    df = pd.concat(df_list, axis=1)
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    print(method)
    print(df)
