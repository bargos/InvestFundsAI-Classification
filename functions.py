import pandas as pd
from itertools import combinations
import numpy as np



def load_mais_retorno_data(test_date):
    features = ['Rentabilidade 1M','Rentabilidade No mês', 'Rentabilidade 3 meses',
        'Rentabilidade 6 meses', 'Rentabilidade 12 meses',
        'Rentabilidade 24 meses', 'Volatilidade No mês', 'Volatilidade 3 meses',
        'Volatilidade 6 meses', 'Volatilidade 12 meses',
        'Volatilidade 24 meses', 'Beta No mês', 'Beta 3 meses', 'Beta 6 meses',
        'Beta 12 meses', 'Beta 24 meses', 'Tracking Error No mês',
        'Tracking Error 3 meses', 'Tracking Error 6 meses',
        'Tracking Error 12 meses', 'Tracking Error 24 meses',
        'Índice de sharpe No mês', 'Índice de sharpe 3 meses',
        'Índice de sharpe 6 meses', 'Índice de sharpe 12 meses',
        'Índice de sharpe 24 meses', 'Índice de sortino No mês',
        'Índice de sortino 3 meses', 'Índice de sortino 6 meses',
        'Índice de sortino 12 meses', 'Índice de sortino 24 meses',
        'Information Ratio No mês', 'Information Ratio 3 meses',
        'Information Ratio 6 meses', 'Information Ratio 12 meses',
        'Information Ratio 24 meses', 'Índice de Treynor No mês',
        'Índice de Treynor 3 meses', 'Índice de Treynor 6 meses',
        'Índice de Treynor 12 meses', 'Índice de Treynor 24 meses']

    mais_data  = pd.read_csv(f'data/processed/dados_122023.csv')
    mais_data['CNPJ'] = mais_data['CNPJ do fundo']
    mais_data.head(2)

    fundos_ranking = pd.read_csv('data/raw/fundos_ranking_24M_202201_202312.csv', sep=';', encoding = 'ISO-8859-1')
    fundos_ranking.head(2)

    fundos_ranking['CNPJ'] = fundos_ranking['CNPJ_FUNDO']
    fundos_ranking['CNPJ'] = pd.to_numeric( ((fundos_ranking['CNPJ'].apply(lambda x: x.replace('.',''))).apply(lambda x: x.replace('/',''))).apply(lambda x: x.replace('-','')) )
    #fundos_ranking.index = fundos_ranking['CNPJ']
    #fundos_ranking.drop(inplace=True, columns='CNPJ_FUNDO2')
    #fundos_ranking.index.names = ['CNPJ_FUNDO']
    fundos_ranking.head()

    result = pd.merge(fundos_ranking, mais_data, on="CNPJ")

    mais_data_1M  = pd.read_csv(f'data/processed/dados_0{test_date+1}2024.csv', usecols=['CNPJ do fundo', 'Rentabilidade No mês'])
    mais_data_1M['CNPJ'] = mais_data_1M['CNPJ do fundo']
    mais_data_1M.drop(inplace=True, columns='CNPJ do fundo')

    mais_data_1M['Rentabilidade 1M'] = mais_data_1M['Rentabilidade No mês']
    mais_data_1M.drop(inplace=True, columns='Rentabilidade No mês')
    mais_data_1M.head()

    data_all = pd.merge(result, mais_data_1M, on="CNPJ")
    result.head(2)

    # features = ['Rentabilidade 1M','Rentabilidade No mês', 'Rentabilidade 3 meses',
    #         'Rentabilidade 6 meses', 'Rentabilidade 12 meses',
    #         'Rentabilidade 24 meses', 'Volatilidade No mês', 'Volatilidade 3 meses',
    #         'Volatilidade 6 meses', 'Volatilidade 12 meses',
    #         'Volatilidade 24 meses',
    #        'Índice de sharpe No mês', 'Índice de sharpe 3 meses',
    #        'Índice de Treynor No mês',
    #        'Índice de Treynor 3 meses']


    data_fundos = data_all[ features ]
    #data_fundos = data_all
    
    
    num_ativos = 4
    num_fundos = data_fundos.shape[0]
    portfolios = list( combinations(range(0, num_fundos), num_ativos) )

    # número de portfólio (número de pesos aleatórios gerados)
    num_portfolios = len( portfolios )

    # cria um array para manter os resultados
    results = np.zeros((num_portfolios, data_fundos.shape[1] ))

    # cria o for loop para as simulações
    for i in range(num_portfolios):
        results[i,:] = data_fundos.iloc[ list(portfolios[i]) ].mean().values
        
    # Converte o resultado para um df
    data = pd.DataFrame(results, index=portfolios, columns = data_fundos.columns)
    
    return data


def load_mais_retorno_data_test(test_date):
    features = ['Rentabilidade 1M','Rentabilidade No mês', 'Rentabilidade 3 meses',
        'Rentabilidade 6 meses', 'Rentabilidade 12 meses',
        'Rentabilidade 24 meses', 'Volatilidade No mês', 'Volatilidade 3 meses',
        'Volatilidade 6 meses', 'Volatilidade 12 meses',
        'Volatilidade 24 meses', 'Beta No mês', 'Beta 3 meses', 'Beta 6 meses',
        'Beta 12 meses', 'Beta 24 meses', 'Tracking Error No mês',
        'Tracking Error 3 meses', 'Tracking Error 6 meses',
        'Tracking Error 12 meses', 'Tracking Error 24 meses',
        'Índice de sharpe No mês', 'Índice de sharpe 3 meses',
        'Índice de sharpe 6 meses', 'Índice de sharpe 12 meses',
        'Índice de sharpe 24 meses', 'Índice de sortino No mês',
        'Índice de sortino 3 meses', 'Índice de sortino 6 meses',
        'Índice de sortino 12 meses', 'Índice de sortino 24 meses',
        'Information Ratio No mês', 'Information Ratio 3 meses',
        'Information Ratio 6 meses', 'Information Ratio 12 meses',
        'Information Ratio 24 meses', 'Índice de Treynor No mês',
        'Índice de Treynor 3 meses', 'Índice de Treynor 6 meses',
        'Índice de Treynor 12 meses', 'Índice de Treynor 24 meses']

    mais_data_teste  = pd.read_csv(f'data/processed/dados_0{test_date}2024.csv')
    mais_data_teste['CNPJ'] = mais_data_teste['CNPJ do fundo']
    fundos_ranking = pd.read_csv('data/raw/fundos_ranking_24M_202201_202312.csv', sep=';', encoding = 'ISO-8859-1')

    fundos_ranking['CNPJ'] = fundos_ranking['CNPJ_FUNDO']
    fundos_ranking['CNPJ'] = pd.to_numeric( ((fundos_ranking['CNPJ'].apply(lambda x: x.replace('.',''))).apply(lambda x: x.replace('/',''))).apply(lambda x: x.replace('-','')) )
    #fundos_ranking.index = fundos_ranking['CNPJ']
    #fundos_ranking.drop(inplace=True, columns='CNPJ_FUNDO2')
    #fundos_ranking.index.names = ['CNPJ_FUNDO']
    result2 = pd.merge(fundos_ranking, mais_data_teste, on="CNPJ")

    mais_data_1M_teste  = pd.read_csv(f'data/processed/dados_0{test_date+1}2024.csv', usecols=['CNPJ do fundo', 'Rentabilidade No mês'])
    mais_data_1M_teste['CNPJ'] = mais_data_1M_teste['CNPJ do fundo']
    mais_data_1M_teste.drop(inplace=True, columns='CNPJ do fundo')

    mais_data_1M_teste['Rentabilidade 1M'] = mais_data_1M_teste['Rentabilidade No mês']
    mais_data_1M_teste.drop(inplace=True, columns='Rentabilidade No mês')

    data_all2 = pd.merge(result2, mais_data_1M_teste, on="CNPJ")

    data_fundos_teste = data_all2[ features ]

    num_ativos = 4
    num_fundos = data_fundos_teste.shape[0]
    portfolios = list( combinations(range(0, num_fundos), num_ativos) )

    # número de portfólio (número de pesos aleatórios gerados)
    num_portfolios = len( portfolios )

    # cria um array para manter os resultados
    results_teste = np.zeros((num_portfolios, data_fundos_teste.shape[1] ))

    # cria o for loop para as simulações
    for i in range(num_portfolios):
        results_teste[i,:] = data_fundos_teste.iloc[ list(portfolios[i]) ].mean().values
        
    # Converte o resultado para um df
    data_teste = pd.DataFrame(results_teste, index=portfolios, columns = data_fundos_teste.columns)
    data_teste.head()
    
    return data_teste
