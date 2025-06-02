## The draw_plots.py is generated with ai. (seems to work)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import os

def generate_box_plots(task, file_list, x_labels, y_label="f-score", title="", significance_pairs=None, results_dir="./results"):
    # Load data from all files
    data = []
    all_values = []
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    for file_path in file_list:
        df = pd.read_csv(file_path)
        column_name = f"{task.upper()}_f1"
        values = df[column_name].values
        data.append(values)
        all_values.extend(values)
    
    # Check consistent number of rows
    n_rows = len(data[0])
    for i, values in enumerate(data[1:], 1):
        if len(values) != n_rows:
            raise ValueError(f"Inconsistent number of folds: file 0 has {n_rows}, file {i} has {len(values)}")
    
    # Create box plot
    plt.figure(figsize=(10, 6))
    
    # Generate different colors for each box
    colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
    
    box_plot = plt.boxplot(data, labels=x_labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel(y_label)
    plt.xlabel('model')
    if title:
        plt.title(title)
    
    # Add significance annotations if specified
    if significance_pairs:
        y_max = max(all_values)
        y_range = max(all_values) - min(all_values)
        
        for i, (idx1, idx2, sig_label) in enumerate(significance_pairs):
            y_pos = y_max + y_range * 0.05 * (i + 1)
            
            # Draw horizontal line
            plt.plot([idx1 + 1, idx2 + 1], [y_pos, y_pos], 'k-', linewidth=1)
            
            # Draw vertical lines
            plt.plot([idx1 + 1, idx1 + 1], [y_pos - y_range * 0.01, y_pos], 'k-', linewidth=1)
            plt.plot([idx2 + 1, idx2 + 1], [y_pos - y_range * 0.01, y_pos], 'k-', linewidth=1)
            
            # Add significance label
            plt.text((idx1 + idx2) / 2 + 1, y_pos + y_range * 0.01, sig_label, 
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'box_plot.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate and save mean and variance
    mean_var_data = []
    for i, (values, label) in enumerate(zip(data, x_labels)):
        mean_var_data.append({
            'model': label,
            'mean': np.mean(values),
            'variance': np.var(values, ddof=1)
        })
    
    mean_var_df = pd.DataFrame(mean_var_data)
    mean_var_df.to_csv(os.path.join(results_dir, 'box_plot_mean_var.csv'), index=False)
    # print(f"Mean and variance saved to {mean_var_path}")
    
    # Calculate p-values between all pairs
    p_value_data = []
    n_models = len(data)
    
    for i in range(n_models):
        for j in range(i + 1, n_models):
            # Perform t-test between pairs
            t_stat, p_value = stats.ttest_ind(data[i], data[j])
            p_value_data.append({
                'model1': x_labels[i],
                'model2': x_labels[j],
                'p_value': p_value,
                't_statistic': t_stat
            })
    
    p_value_df = pd.DataFrame(p_value_data)
    p_value_df.to_csv(os.path.join(results_dir, 'box_plot_p_values.csv'), index=False)
    
    print(f"Done with {title}. Saved to {results_dir}")
    return mean_var_df, p_value_df

significance_pairs = [(0, 1, 'hi'), (0, 2, 'hello'), (1, 2, 'you can put custom text here. cool')]

for lge in ["en", "fr", "zh"]:
    file_list = [
        f"./results/results_token_classification/{lge}_red_{lge}_wiki_sp_5e4_cv.csv",
        f"./results/results_token_classification/{lge}_red_{lge}_spoken_sp_5e4_cv.csv",
        f"./results/results_token_classification/{lge}_red_{lge}_mixed_sp_5e4_cv.csv",
    ]

    generate_box_plots(
        task="red",
        file_list=file_list,
        x_labels=["wiki", "spoken", "mixed"],
        title=f"{lge} red models",
        significance_pairs=significance_pairs,
        results_dir=f"./results/box_plot_{lge}"
    )