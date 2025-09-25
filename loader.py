# Existing CSV loads for persons J, M, D, etc.
import pandas as pd
import pickle



def load_data():
    df_j = pd.read_csv('valid_results_long_j.csv')
    df_m = pd.read_csv('valid_results_long_m.csv')
    df_d = pd.read_csv('valid_results_d.csv')
    df_ma = pd.read_csv('valid_results_MA.csv')
    df_v = pd.read_csv('valid_results_v.csv')
    df_t = pd.read_csv('valid_results_t.csv')
    df_003 = pd.read_csv('valid_results_003.csv')
    df_004 = pd.read_csv('valid_results_004.csv')
    df_005 = pd.read_csv('valid_results_005.csv')
    df_006 = pd.read_csv('valid_results_006.csv')
    df_007 = pd.read_csv('valid_results_007.csv')
    df_008 = pd.read_csv('valid_results_008.csv')
    df_009 = pd.read_csv('valid_results_009.csv')
    df_010 = pd.read_csv('valid_results_010.csv')

    # --- Added for people 011 through 016 ---
    df_011 = pd.read_csv('valid_results_011.csv')
    df_012 = pd.read_csv('valid_results_012.csv')
    df_013 = pd.read_csv('valid_results_013.csv')
    df_014 = pd.read_csv('valid_results_014.csv')
    df_015 = pd.read_csv('valid_results_015.csv')
    df_016 = pd.read_csv('valid_results_016.csv')

    # Assign person identifiers
    df_j['person'] = 'J'
    df_m['person'] = 'M'
    df_d['person'] = 'D'
    df_ma['person'] = 'MA'
    df_v['person'] = 'V'
    df_t['person'] = 'T'
    df_003['person'] = '003'
    df_004['person'] = '004'
    df_005['person'] = '005'
    df_006['person'] = '006'
    df_007['person'] = '007'
    df_008['person'] = '008'
    df_009['person'] = '009'
    df_010['person'] = '010'

    # --- Added for people 011 through 016 ---
    df_011['person'] = '011'
    df_012['person'] = '012'
    df_013['person'] = '013'
    df_014['person'] = '014'
    df_015['person'] = '015'
    df_016['person'] = '016'

    # Concatenate all dataframes
    df = pd.concat([df_j, df_m, df_d, df_ma, df_v, df_t, 
                    df_003, df_004, df_005, df_006, df_007, df_008, df_009, df_010,
                    df_011, df_012, df_013, df_014, df_015, df_016], ignore_index=True)

    classes = []

    # Assign class labels based on person ID
    for i in range(len(df)):
        if df['person'][i] == 'J':
            classes.extend([0, 0, 0])
        elif df['person'][i] == 'M':
            classes.extend([1, 1, 1])
        elif df['person'][i] == 'D':
            classes.extend([2, 2, 2])
        elif df['person'][i] == 'MA':
            classes.extend([3, 3, 3])
        elif df['person'][i] == 'V':
            classes.extend([4, 4, 4])
        elif df['person'][i] == 'T':
            classes.extend([5, 5, 5])
        elif df['person'][i] == '003':
            classes.extend([6, 6, 6])
        elif df['person'][i] == '004':
            classes.extend([7, 7, 7])
        elif df['person'][i] == '005':
            classes.extend([8, 8, 8])
        elif df['person'][i] == '006':
            classes.extend([9, 9, 9])
        elif df['person'][i] == '007':
            classes.extend([10, 10, 10])
        elif df['person'][i] == '008':
            classes.extend([11, 11, 11])
        elif df['person'][i] == '009':
            classes.extend([12, 12, 12])
        elif df['person'][i] == '010':
            classes.extend([13, 13, 13])
        # --- Added for people 011 through 016 ---
        elif df['person'][i] == '011':
            classes.extend([14, 14, 14])
        elif df['person'][i] == '012':
            classes.extend([15, 15, 15])
        elif df['person'][i] == '013':
            classes.extend([16, 16, 16])
        elif df['person'][i] == '014':
            classes.extend([17, 17, 17])
        elif df['person'][i] == '015':
            classes.extend([18, 18, 18])
        elif df['person'][i] == '016':
            classes.extend([19, 19, 19])

    # Load additional feature data from pickle files
    with open('dataframes_work_new_long_j.pkl', 'rb') as file:
        loaded_df_list_j = pickle.load(file)
    with open('dataframes_work_new_long_m.pkl', 'rb') as file:
        loaded_df_list_m = pickle.load(file)
    with open('dataframes_d.pkl', 'rb') as file:
        loaded_df_list_d = pickle.load(file)
    with open('dataframes_MA.pkl', 'rb') as file:
        loaded_df_list_ma = pickle.load(file)
    with open('dataframes_v.pkl', 'rb') as file:
        loaded_df_list_v = pickle.load(file)
    with open('dataframes_t.pkl', 'rb') as file:
        loaded_df_list_t = pickle.load(file)
    with open('dataframes_003.pkl', 'rb') as file:
        loaded_df_list_003 = pickle.load(file)
    with open('dataframes_004.pkl', 'rb') as file:
        loaded_df_list_004 = pickle.load(file)
    with open('dataframes_005.pkl', 'rb') as file:
        loaded_df_list_005 = pickle.load(file)
    with open('dataframes_006.pkl', 'rb') as file:
        loaded_df_list_006 = pickle.load(file)
    with open('dataframes_007.pkl', 'rb') as file:
        loaded_df_list_007 = pickle.load(file)
    with open('dataframes_008.pkl', 'rb') as file:
        loaded_df_list_008 = pickle.load(file)
    with open('dataframes_009.pkl', 'rb') as file:
        loaded_df_list_009 = pickle.load(file)
    with open('dataframes_010.pkl', 'rb') as file:
        loaded_df_list_010 = pickle.load(file)

    # --- Added for people 011 through 016 ---
    with open('dataframes_011.pkl', 'rb') as file:
        loaded_df_list_011 = pickle.load(file)
    with open('dataframes_012.pkl', 'rb') as file:
        loaded_df_list_012 = pickle.load(file)
    with open('dataframes_013.pkl', 'rb') as file:
        loaded_df_list_013 = pickle.load(file)
    with open('dataframes_014.pkl', 'rb') as file:
        loaded_df_list_014 = pickle.load(file)
    with open('dataframes_015.pkl', 'rb') as file:
        loaded_df_list_015 = pickle.load(file)
    with open('dataframes_016.pkl', 'rb') as file:
        loaded_df_list_016 = pickle.load(file)

    # Concatenate all loaded dataframe lists
    loaded_df_list = (loaded_df_list_j + loaded_df_list_m + loaded_df_list_d +
                    loaded_df_list_ma + loaded_df_list_v + loaded_df_list_t +
                    loaded_df_list_003 + loaded_df_list_004 + loaded_df_list_005 +
                    loaded_df_list_006 + loaded_df_list_007 + loaded_df_list_008 +
                    loaded_df_list_009 + loaded_df_list_010 +
                    loaded_df_list_011 + loaded_df_list_012 + loaded_df_list_013 +
                    loaded_df_list_014 + loaded_df_list_015 + loaded_df_list_016)
    
    return df, classes, loaded_df_list