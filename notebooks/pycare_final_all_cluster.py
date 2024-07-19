import pandas as pd
#from functions import *
import ipywidgets as widgets
from ipywidgets import interact, fixed, interact_manual
# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score
from pycaret.regression import *


clusters = [0]

mods = ['lr', 'ridge', 'et', 'rf', 'lightgbm', 'knn', 'gbr', 'huber', 'ada', 'omp']

for cluster in clusters:
        
        print(f'\n CLUSTER NUMBER: {cluster: .0f}')

        for md in mods:
                features = ['Rentabilidade 1M','Beta No mês', 'Beta 3 meses', 'Beta 6 meses',
                        'Beta 12 meses', 'Beta 24 meses',   
                        ]

                data0 = pd.read_csv('data/processed/kmeans_cluster_3M.csv', index_col=0)
                # data = data0.loc[data0['Cluster']==cluster]   
                # data.drop(columns='Cluster', inplace=True)

                data=data0
                #data.drop(columns='Cluster', inplace=True)
                
                s = setup(data, target = 'Rentabilidade 1M', session_id=123, transformation = True, transformation_method='quantile')
                #s = setup(data, target = 'Rentabilidade 1M', session_id=123)


                best = compare_models(include=[md])


                tuned_best = best

                final_model = finalize_model(tuned_best)


                data_teste0 = pd.read_csv('data/processed/kmeans_cluster_1M.csv', index_col=0)
                # data_teste = data_teste0.loc[data0['Cluster']==cluster]
                data_teste = data_teste0

                # previsão em dados não vistos
                unseen_predictions = predict_model(final_model, data=data_teste)
                #unseen_predictions.head()

                # Evaluating the model
                mse1 = mean_squared_error(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])    
                r21 = r2_score(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])

                data_teste0 = pd.read_csv('data/processed/kmeans_cluster_2M.csv', index_col=0)
                # data_teste = data_teste0.loc[data0['Cluster']==cluster]
                data_teste = data_teste0

                # previsão em dados não vistos
                unseen_predictions = predict_model(final_model, data=data_teste)
                #unseen_predictions.head()

                mse2 = mean_squared_error(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])    
                r22 = r2_score(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])


                data_teste0 = pd.read_csv('data/processed/kmeans_cluster_3M.csv', index_col=0)
                # data_teste = data_teste0.loc[data0['Cluster']==cluster]
                data_teste = data_teste0
                
                # previsão em dados não vistos
                unseen_predictions = predict_model(final_model, data=data_teste)
                #unseen_predictions.head()

                # Evaluating the model
                mse3 = mean_squared_error(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])    
                r23 = r2_score(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])

                print('\n                          1M       2M       3M')
                print(f'Mean Squared Error: {mse1: 8.4f} {mse2: 8.4f} {mse3: 8.4f}')
                print(f'R-squared.........: {r21: 8.4f} {r22: 8.4f} {r23: 8.4f} \n\n')