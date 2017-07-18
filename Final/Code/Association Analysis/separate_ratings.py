import pandas as pd
import numpy as np

def get_green_red():
	encoded_df=pd.read_csv('../../ProcessedData/encoded_capital_mode.csv')
	red_df=encoded_df[encoded_df['rating']=='red'].sample(frac=1).reset_index()
	print "#red=", len(red_df)

	green_df=encoded_df[encoded_df['rating']=='green'].sample(frac=1).reset_index()
	print "#green=",len(green_df)

	return green_df, red_df


if __name__ == '__main__':
	gr, rd= get_green_red()
	gr.to_csv('green.csv', encoding='utf-8', index=False)
	rd.to_csv('red.csv', encoding='utf-8', index=False)


