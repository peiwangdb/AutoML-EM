def ProduceTable(df_m, df_a, df_b):
	left_id_list = df_m["ltable_id"].tolist()
	right_id_list = df_m["rtable_id"].tolist()
	dict_new_data = {}
	
	left_list = df_a.columns.tolist()
	for list_name in left_list:
		if list_name != "id":
			dict_new_data["ltable_" + list_name] = []
			for each_id in left_id_list:
				dict_new_data["ltable_" + list_name].append(df_a.iloc[each_id][list_name])
	
	right_list = df_b.columns.tolist()
	for list_name in right_list:
		if list_name != "id":
			dict_new_data["rtable_" + list_name] = []
			for each_id in right_id_list:
				dict_new_data["rtable_" + list_name].append(df_b.iloc[each_id][list_name])

	for k, v in dict_new_data.items():
		df_m[k] = v
	
	return df_m

