from pandas_profiling import ProfileReport


def data_profiling_from(data_eda, output_file_path, output_file_name):
    prof = ProfileReport(data_eda)
    prof.to_file(output_file=output_file_path+'/'+output_file_name)