import pandas as pd
#from functions import *
import ipywidgets as widgets
from ipywidgets import interact, fixed, interact_manual
# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score
from pycaret.regression import *





mods = ['lr', 'ridge', 'et', 'rf', 'lightgbm', 'knn', 'gbr', 'huber', 'ada', 'omp']
for md in mods:
    data0 = load_mais_retorno_data(1)

    features = ['Rentabilidade 1M','Beta No mês', 'Beta 3 meses', 'Beta 6 meses',
            'Beta 12 meses', 'Beta 24 meses',   
            ]

    data = data0
#     data['Rentabilidade 1M'] = np.log( data['Rentabilidade 1M'] )
#     data['Rentabilidade No mês'] = np.log( data['Rentabilidade No mês'])
#     data['Rentabilidade 3 meses'] = np.log( data['Rentabilidade 3 meses'])
#     data['Rentabilidade 6 meses'] = np.log( data['Rentabilidade 6 meses'])
#     data['Rentabilidade 12 meses'] = np.log( data['Rentabilidade 12 meses'])
#     data['Rentabilidade 24 meses'] = np.log( data['Rentabilidade 24 meses'])
    data.dropna(inplace=True)


    s = setup(data, target = 'Rentabilidade 1M', session_id=123, transformation = True, transformation_method='quantile')


    best = compare_models(include=[md])


    tuned_best = best

    final_model = finalize_model(tuned_best)


    data_teste0 = load_mais_retorno_data_test(1)

    data_teste = data_teste0


#     data_teste['Rentabilidade 1M'] = np.log( data_teste['Rentabilidade 1M'] )
#     data_teste['Rentabilidade No mês'] = np.log( data_teste['Rentabilidade No mês'])
#     data_teste['Rentabilidade 3 meses'] = np.log( data_teste['Rentabilidade 3 meses'])
#     data_teste['Rentabilidade 6 meses'] = np.log( data_teste['Rentabilidade 6 meses'])
#     data_teste['Rentabilidade 12 meses'] = np.log( data_teste['Rentabilidade 12 meses'])
#     data_teste['Rentabilidade 24 meses'] = np.log( data_teste['Rentabilidade 24 meses'])
    data_teste.dropna(inplace=True)

    # previsão em dados não vistos
    unseen_predictions = predict_model(final_model, data=data_teste)
    #unseen_predictions.head()

    # df_final = unseen_predictions.copy()
    # df_final['diferença'] = abs( df_final['prediction_label'] - df_final['Rentabilidade 1M'] )
    # (df_final.corr().iloc[-1]).sort_values(ascending=True)


    # Evaluating the model
    mse = mean_squared_error(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])
    print(f'Mean Squared Error: {mse}')
    
    r2 = r2_score(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])
    print(f'R-squared: {r2}')





    data_teste0 = load_mais_retorno_data_test(2)
    data_teste = data_teste0
#     data_teste['Rentabilidade 1M'] = np.log( data_teste['Rentabilidade 1M'] )
#     data_teste['Rentabilidade No mês'] = np.log( data_teste['Rentabilidade No mês'])
#     data_teste['Rentabilidade 3 meses'] = np.log( data_teste['Rentabilidade 3 meses'])
#     data_teste['Rentabilidade 6 meses'] = np.log( data_teste['Rentabilidade 6 meses'])
#     data_teste['Rentabilidade 12 meses'] = np.log( data_teste['Rentabilidade 12 meses'])
#     data_teste['Rentabilidade 24 meses'] = np.log( data_teste['Rentabilidade 24 meses'])
    data_teste.dropna(inplace=True)

    # previsão em dados não vistos
    unseen_predictions = predict_model(final_model, data=data_teste)
    #unseen_predictions.head()

    # df_final = unseen_predictions.copy()
    # df_final['diferença'] = abs( df_final['prediction_label'] - df_final['Rentabilidade 1M'] )
    # (df_final.corr().iloc[-1]).sort_values(ascending=True)
    # Evaluating the model
    mse = mean_squared_error(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])
    print(f'Mean Squared Error: {mse}')
    
    r2 = r2_score(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])
    print(f'R-squared: {r2}')




    data_teste0 = load_mais_retorno_data_test(3)

    data_teste = data_teste0

#     data_teste['Rentabilidade 1M'] = np.log( data_teste['Rentabilidade 1M'] )
#     data_teste['Rentabilidade No mês'] = np.log( data_teste['Rentabilidade No mês'])
#     data_teste['Rentabilidade 3 meses'] = np.log( data_teste['Rentabilidade 3 meses'])
#     data_teste['Rentabilidade 6 meses'] = np.log( data_teste['Rentabilidade 6 meses'])
#     data_teste['Rentabilidade 12 meses'] = np.log( data_teste['Rentabilidade 12 meses'])
#     data_teste['Rentabilidade 24 meses'] = np.log( data_teste['Rentabilidade 24 meses'])
    data_teste.dropna(inplace=True)

    # previsão em dados não vistos
    unseen_predictions = predict_model(final_model, data=data_teste)
    #unseen_predictions.head()

    # df_final = unseen_predictions.copy()
    # df_final['diferença'] = abs( df_final['prediction_label'] - df_final['Rentabilidade 1M'] )
    # (df_final.corr().iloc[-1]).sort_values(ascending=True)
    # Evaluating the model
    mse = mean_squared_error(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])
    print(f'Mean Squared Error: {mse}')
    
    r2 = r2_score(unseen_predictions['Rentabilidade 1M'], unseen_predictions['prediction_label'])
    print(f'R-squared: {r2}')