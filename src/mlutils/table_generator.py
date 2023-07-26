import pandas as pd
import glob
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--format_bold", action="store_true", default=False)
args = parser.parse_args()

project = "adni-brats-finetune"
print(f"Generating table for project {project}...")
# Define the directory containing the CSV files
data_dir = os.path.join('/dhc/home/tim.riedel/bachelor-thesis/jobs/results/', project)
save_dir = data_dir

mask_ratios = ["0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "Baseline"]
train_fracs = ["0.1", "0.25", "0.5", "1.0"]


# Get a list of all CSV files in the directory
csv_files = glob.glob(data_dir + 'train_fraction_*.csv')


def format_bold(val, is_max=False):
    if is_max and args.format_bold:
        return f"\\underline{{\\textbf{{{str(val)}}}}}"
    return str(val)


# Function to generate the latex table
def generate_latex_table():
    header = """\\begin{table}[]
    \\resizebox{\\textwidth}{!}{%
    \\begin{tabular}{clcccclcccclcccclcccc}
    \\hline
    Train Frac. &  & \\multicolumn{4}{c}{0.1} &  & \\multicolumn{4}{c}{0.25} &  & \\multicolumn{4}{c}{0.5} &  & \\multicolumn{4}{c}{1.0} \\\\ \\cline{1-1} \\cline{3-6} \\cline{8-11} \\cline{13-16} 
    Mask Ratio  &  & Avg    & TC     & ET     & WT     &  & Avg    & TC     & ET     & WT     &  & Avg    & TC     & ET     & WT     &  & Avg    & TC     & ET     & WT     \\\\ \\hline"""
    

    footer = """\\hline
    \\end{tabular}}
    \\end{table}
    """

    first_col = """
    {}         """
    tf_row_format = """&  & {} & {} & {} & {} """

    footer = """\\hline
    \\end{tabular}}
    \\caption{Results TODO}
    \\label{tab:results}\n\\end{table}"""
    
    table_rows = """"""
    for mr in mask_ratios:
        row_format = first_col.format(mr)
        for tf in train_fracs:
            file = os.path.join(data_dir, f"train_fraction_{tf}.csv")
            try:
                tf_df = pd.read_csv(file)
                row = tf_df.loc[tf_df['Mask Ratio'] == mr]
            except:
                tf_df = None
                row = pd.DataFrame()

            if row.empty:
                pass
                row_format += tf_row_format.format("", "", "", "")
            elif tf_df is not None:
                avg = format_bold(row['Average'].values[0], row['Average'].values[0] == tf_df['Average'].max())
                tc = format_bold(row['TC'].values[0], row['TC'].values[0] == tf_df['TC'].max())
                et = format_bold(row['ET'].values[0], row['ET'].values[0] == tf_df['ET'].max())
                wt = format_bold(row['WT'].values[0], row['WT'].values[0] == tf_df['WT'].max())

                row_format += tf_row_format.format(avg, tc, et, wt)
            else:
                raise Exception("File was not loaded.")
        table_rows += row_format
        table_rows += """\\\\"""

    return header + table_rows + footer

latex_table = generate_latex_table()

# Write the latex table to a .tex file
with open(os.path.join(save_dir, 'table.tex'), 'w') as f:
    f.write(latex_table)
    print("Table saved.")
