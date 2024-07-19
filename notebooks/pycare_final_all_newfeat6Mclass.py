import pandas as pd
#from functions import *
import ipywidgets as widgets
from ipywidgets import interact, fixed, interact_manual
# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score
from pycaret.classification import *



clusters = [-1.0, 0.0, 1.0]
mods = ['lr', 'ridge', 'et', 'rf', 'lightgbm', 'knn', 'gbr', 'huber', 'ada', 'omp']
# mods = [ 'lightgbm']
#mods = ['ada']


for md in mods:
        

        #data0 = pd.read_excel('data/raw/Fundos2023maisretorno.xlsx', index_col=0, skiprows=3)

        dta0_01 = pd.read_csv('data/raw/jan2023.csv')
        dta0_02 = pd.read_csv('data/raw/fev2023.csv')
        dta0_03 = pd.read_csv('data/raw/mar2023.csv')
        dta0_04 = pd.read_csv('data/raw/abr2023.csv')
        dta0_05 = pd.read_csv('data/raw/mai2023.csv')
        dta0_06 = pd.read_csv('data/raw/jun2023.csv')
        dta0_07 = pd.read_csv('data/raw/jul2023.csv')
        dta0_08 = pd.read_csv('data/raw/ago2023.csv')
        dta0_09 = pd.read_csv('data/raw/set2023.csv')
        dta0_10 = pd.read_csv('data/raw/out2023.csv')
        dta0_11 = pd.read_csv('data/raw/nov2023.csv')
        data0_0 = pd.read_csv('data/raw/dez2023.csv')
        data0_1 = pd.read_csv('data/raw/jan2024.csv')
        data0_2 = pd.read_csv('data/raw/fev2024.csv')
        data0_3 = pd.read_csv('data/raw/mar2024.csv')
        data0_4 = pd.read_csv('data/raw/abr2024.csv')
        data0_5 = pd.read_csv('data/raw/mai2024.csv')

        feat_drop = ['Nome do fundo', 'CNPJ do fundo', 'cnpj', 'Classe N1',
                'Patrimônio líquido', 'Cotistas', 'Valor da cota', 'Variação da Cota',
                'Drawdown máximo']
        
        cpj01 = dta0_01['CNPJ do fundo'].loc[ (dta0_01['Cotistas'] > 100) & (dta0_01['Classe N1'] == 'Ações')].values
        cpj02 = dta0_02['CNPJ do fundo'].loc[ (dta0_02['Cotistas'] > 100) & (dta0_02['Classe N1'] == 'Ações')].values
        cpj03 = dta0_03['CNPJ do fundo'].loc[ (dta0_03['Cotistas'] > 100) & (dta0_03['Classe N1'] == 'Ações')].values
        cpj04 = dta0_04['CNPJ do fundo'].loc[ (dta0_04['Cotistas'] > 100) & (dta0_04['Classe N1'] == 'Ações')].values
        cpj05 = dta0_05['CNPJ do fundo'].loc[ (dta0_05['Cotistas'] > 100) & (dta0_05['Classe N1'] == 'Ações')].values     
        cpj06 = dta0_06['CNPJ do fundo'].loc[ (dta0_06['Cotistas'] > 100) & (dta0_06['Classe N1'] == 'Ações')].values     
        cpj07 = dta0_07['CNPJ do fundo'].loc[ (dta0_07['Cotistas'] > 100) & (dta0_07['Classe N1'] == 'Ações')].values        
        cpj08 = dta0_08['CNPJ do fundo'].loc[ (dta0_08['Cotistas'] > 100) & (dta0_08['Classe N1'] == 'Ações')].values        
        cpj09 = dta0_09['CNPJ do fundo'].loc[ (dta0_09['Cotistas'] > 100) & (dta0_09['Classe N1'] == 'Ações')].values        
        cpj10 = dta0_10['CNPJ do fundo'].loc[ (dta0_10['Cotistas'] > 100) & (dta0_10['Classe N1'] == 'Ações')].values        
        cpj11 = dta0_11['CNPJ do fundo'].loc[ (dta0_11['Cotistas'] > 100) & (dta0_11['Classe N1'] == 'Ações')].values
        cnpj0 = data0_0['CNPJ do fundo'].loc[ (data0_0['Cotistas'] > 100) & (data0_0['Classe N1'] == 'Ações')].values
        cnpj1 = data0_1['CNPJ do fundo'].loc[ (data0_1['Cotistas'] > 100) & (data0_1['Classe N1'] == 'Ações')].values
        cnpj2 = data0_2['CNPJ do fundo'].loc[ (data0_2['Cotistas'] > 100) & (data0_2['Classe N1'] == 'Ações')].values
        cnpj3 = data0_3['CNPJ do fundo'].loc[ (data0_3['Cotistas'] > 100) & (data0_3['Classe N1'] == 'Ações')].values
        cnpj4 = data0_4['CNPJ do fundo'].loc[ (data0_4['Cotistas'] > 100) & (data0_4['Classe N1'] == 'Ações')].values
        cnpj5 = data0_5['CNPJ do fundo'].loc[ (data0_5['Cotistas'] > 100) & (data0_5['Classe N1'] == 'Ações')].values

        set1 = set(cpj01)
        set2 = set(cpj02)
        matches01 = list(set1.intersection(set2))
        dta01  = dta0_01.loc[dta0_01['CNPJ do fundo'].isin(matches01)]
        rent1M = dta0_02.loc[dta0_02['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        dta01 = dta01.merge(rent1M, how='inner', on='CNPJ do fundo')
        
        set1 = set(cpj02)
        set2 = set(cpj03)
        matches01 = list(set1.intersection(set2))
        dta02  = dta0_02.loc[dta0_02['CNPJ do fundo'].isin(matches01)]
        rent1M = dta0_03.loc[dta0_03['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        dta02 = dta02.merge(rent1M, how='inner', on='CNPJ do fundo')
        
        set1 = set(cpj03)
        set2 = set(cpj04)
        matches01 = list(set1.intersection(set2))
        dta03  = dta0_03.loc[dta0_03['CNPJ do fundo'].isin(matches01)]
        rent1M = dta0_04.loc[dta0_04['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        dta03 = dta03.merge(rent1M, how='inner', on='CNPJ do fundo')
        
        set1 = set(cpj04)
        set2 = set(cpj05)
        matches01 = list(set1.intersection(set2))
        dta04  = dta0_04.loc[dta0_04['CNPJ do fundo'].isin(matches01)]
        rent1M = dta0_05.loc[dta0_05['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        dta04 = dta04.merge(rent1M, how='inner', on='CNPJ do fundo')


        set1 = set(cpj05)
        set2 = set(cpj06)
        matches01 = list(set1.intersection(set2))
        dta05  = dta0_05.loc[dta0_05['CNPJ do fundo'].isin(matches01)]
        rent1M = dta0_06.loc[dta0_06['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        dta05 = dta05.merge(rent1M, how='inner', on='CNPJ do fundo')
        
        set1 = set(cpj06)
        set2 = set(cpj07)
        matches01 = list(set1.intersection(set2))
        dta06  = dta0_06.loc[dta0_06['CNPJ do fundo'].isin(matches01)]
        rent1M = dta0_07.loc[dta0_07['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        dta06 = dta06.merge(rent1M, how='inner', on='CNPJ do fundo')
        
        set1 = set(cpj07)
        set2 = set(cpj08)
        matches01 = list(set1.intersection(set2))
        dta07  = dta0_07.loc[dta0_07['CNPJ do fundo'].isin(matches01)]
        rent1M = dta0_08.loc[dta0_08['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        dta07 = dta07.merge(rent1M, how='inner', on='CNPJ do fundo')
        
        set1 = set(cpj08)
        set2 = set(cpj09)
        matches01 = list(set1.intersection(set2))
        dta08  = dta0_08.loc[dta0_08['CNPJ do fundo'].isin(matches01)]
        rent1M = dta0_09.loc[dta0_09['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        dta08 = dta08.merge(rent1M, how='inner', on='CNPJ do fundo')
        
        set1 = set(cpj09)
        set2 = set(cpj10)
        matches01 = list(set1.intersection(set2))
        dta09  = dta0_09.loc[dta0_09['CNPJ do fundo'].isin(matches01)]
        rent1M = dta0_10.loc[dta0_10['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        dta09 = dta09.merge(rent1M, how='inner', on='CNPJ do fundo')
                        
        set1 = set(cpj10)
        set2 = set(cpj11)
        matches01 = list(set1.intersection(set2))
        dta10  = dta0_10.loc[dta0_10['CNPJ do fundo'].isin(matches01)]
        rent1M = dta0_11.loc[dta0_11['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        dta10 = dta10.merge(rent1M, how='inner', on='CNPJ do fundo')
        
        set1 = set(cpj11)
        set2 = set(cnpj0)
        matches01 = list(set1.intersection(set2))
        dta11  = dta0_11.loc[dta0_11['CNPJ do fundo'].isin(matches01)]
        rent1M = data0_0.loc[data0_0['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        dta11 = dta11.merge(rent1M, how='inner', on='CNPJ do fundo')
        
        set1 = set(cnpj0)
        set2 = set(cnpj1)
        matches01 = list(set1.intersection(set2))
        data0  = data0_0.loc[data0_0['CNPJ do fundo'].isin(matches01)]
        rent1M = data0_1.loc[data0_1['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        data0 = data0.merge(rent1M, how='inner', on='CNPJ do fundo')


        set1 = set(cnpj1)
        set2 = set(cnpj2)
        matches01 = list(set1.intersection(set2))
        data1  = data0_1.loc[data0_1['CNPJ do fundo'].isin(matches01)]
        rent1M = data0_2.loc[data0_2['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        data1 = data1.merge(rent1M, how='inner', on='CNPJ do fundo')


        set1 = set(cnpj2)
        set2 = set(cnpj3)
        matches01 = list(set1.intersection(set2))
        data2  = data0_2.loc[data0_2['CNPJ do fundo'].isin(matches01)]
        rent1M = data0_3.loc[data0_3['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        data2 = data2.merge(rent1M, how='inner', on='CNPJ do fundo')

        data = pd.concat([ dta01, dta02, dta03, dta04, dta05, dta06, dta07, dta08, dta09, dta10, dta11, data0, data1, data2], ignore_index=True)
        #data = pd.concat([ dta01, dta02, dta03, dta04, dta05, dta06, dta07, dta08, dta09, dta10, dta11, data0, data1, data2], ignore_index=True)
        data.dropna(axis=0, inplace=True)
        data.drop(columns=feat_drop, inplace=True)
        data.reset_index(drop=True, inplace=True)
        
        data['Cluster'] = 0
        data['Cluster'].loc[ data['Rentabilidade 6 meses'] < (data['Rentabilidade 6 meses'].mean() - 0.5*data['Rentabilidade 6 meses'].std())] = -1.0
        data['Cluster'].loc[ data['Rentabilidade 6 meses'] > (data['Rentabilidade 6 meses'].mean() + 0.5*data['Rentabilidade 6 meses'].std())] =  1.0
        
        data.drop(columns='Rentabilidade 6M', inplace=True)
        # data = data.loc[data['Cluster']==cluster]
        # data.drop(columns='Cluster', inplace=True)
        
        
        s = setup(data, target = 'Cluster', session_id=123, transformation = True, transformation_method='quantile', train_size=0.8)
        #s = setup(data, target = 'Rentabilidade 6M', session_id=123, feature_selection=True, transformation = True, transformation_method='quantile', train_size=0.8)
        #s = setup(data, target = 'Rentabilidade 6M', session_id=123, feature_selection=True, transformation = True, transformation_method='quantile')
        #s = setup(data, target = 'Rentabilidade 6M', session_id=123, transformation = True, transformation_method='quantile')
        #s = setup(data, target = 'Rentabilidade 6M', session_id=123)


        best = compare_models(include=[md])


        tuned_best = best

        final_model = finalize_model(tuned_best)


        set1 = set(cnpj3)
        set2 = set(cnpj4)
        matches01 = list(set1.intersection(set2))
        data3  = data0_3.loc[data0_3['CNPJ do fundo'].isin(matches01)]
        rent1M = data0_4.loc[data0_4['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        data3 = data3.merge(rent1M, how='inner', on='CNPJ do fundo')
        data3.dropna(axis=0, inplace=True)
        
        data_teste = data3
        
        data_teste.drop(columns=feat_drop, inplace=True)
        data_teste.reset_index(drop=True, inplace=True)
        
        data_teste['Cluster'] = 0
        data_teste['Cluster'].loc[ data_teste['Rentabilidade 6 meses'] < (data_teste['Rentabilidade 6 meses'].mean() - 0.5*data_teste['Rentabilidade 6 meses'].std())] = -1.0
        data_teste['Cluster'].loc[ data_teste['Rentabilidade 6 meses'] > (data_teste['Rentabilidade 6 meses'].mean() + 0.5*data_teste['Rentabilidade 6 meses'].std())] =  1.0
        
        # data_teste = data_teste.loc[data_teste['Cluster']==cluster]
        # data_teste.drop(columns='Cluster', inplace=True)
        

        # previsão em dados não vistos
        unseen_predictions = predict_model(final_model, data=data_teste)
        #unseen_predictions.head()

        # Evaluating the model
        mse1 = mean_squared_error(unseen_predictions['Rentabilidade 6M'], unseen_predictions['prediction_label'])    
        r21 = r2_score(unseen_predictions['Rentabilidade 6M'], unseen_predictions['prediction_label'])



        set1 = set(cnpj4)
        set2 = set(cnpj5)
        matches01 = list(set1.intersection(set2))
        data4  = data0_4.loc[data0_4['CNPJ do fundo'].isin(matches01)]
        rent1M = data0_5.loc[data0_5['CNPJ do fundo'].isin(matches01)][['CNPJ do fundo','Rentabilidade 6 meses']]
        rent1M['Rentabilidade 6M'] = rent1M['Rentabilidade 6 meses']
        rent1M.drop(columns='Rentabilidade 6 meses', inplace=True)
        data4 = data4.merge(rent1M, how='inner', on='CNPJ do fundo')
        data4.dropna(axis=0, inplace=True)
        
        data_teste = data4
        
        data_teste.drop(columns=feat_drop, inplace=True)
        data_teste.reset_index(drop=True, inplace=True)
        
        data_teste['Cluster'] = 0
        data_teste['Cluster'].loc[ data_teste['Rentabilidade 6 meses'] < (data_teste['Rentabilidade 6 meses'].mean() - 0.5*data_teste['Rentabilidade 6 meses'].std())] = -1.0
        data_teste['Cluster'].loc[ data_teste['Rentabilidade 6 meses'] > (data_teste['Rentabilidade 6 meses'].mean() + 0.5*data_teste['Rentabilidade 6 meses'].std())] =  1.0
        
        # data_teste = data_teste.loc[data_teste['Cluster']==cluster]
        # data_teste.drop(columns='Cluster', inplace=True)
        

        # previsão em dados não vistos
        unseen_predictions = predict_model(final_model, data=data_teste)
        #unseen_predictions.head()

        mse2 = mean_squared_error(unseen_predictions['Rentabilidade 6M'], unseen_predictions['prediction_label'])    
        r22 = r2_score(unseen_predictions['Rentabilidade 6M'], unseen_predictions['prediction_label'])

        print('\n                          1M       2M       3M')
        print(f'Mean Squared Error: {mse1: 8.4f} {mse2: 8.4f}')
        print(f'R-squared.........: {r21: 8.4f} {r22: 8.4f}\n\n')