import pandas as pd
import os

def save_csv(filename, eval_results, model_dir, eval_data_dir):
	print("Saving csv file...")
	model = model_dir.split('/')[-2] if model_dir[-1] != '/' else model_dir.split('/')[-3]
	dataset = eval_data_dir.split('/')[-1] if eval_data_dir[-1] != '/' else eval_data_dir.split('/')[-2]
	columns = [(dataset, model)]
	df_eval = pd.DataFrame.from_dict(eval_results, orient='index', columns=columns)
	df_eval.columns = pd.MultiIndex.from_tuples(df_eval.columns, names=['dataset', 'model'])
	if os.path.exists(filename):
		df_csv = pd.read_csv(filename,  header=[0,1], skipinitialspace=True, tupleize_cols=True, index_col=0)
		df_csv = df_csv.join(df_eval)
		#df_csv.columns = pd.MultiIndex.from_tuples(sorted(list(df_csv)), names=['dataset', 'model'])
	else:
		df_csv = df_eval
	df_csv = df_csv.sort_index(axis=1, level=0)
	df_csv.to_csv(filename)

	print("csv file saved!")
