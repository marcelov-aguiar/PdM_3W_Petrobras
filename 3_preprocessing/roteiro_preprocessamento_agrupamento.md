- Apenas pastas cujos nomes sejam 1, 2, 5, 6, 7 e 8 devem ser usadas; Tem períodos normais e períodos com anomalia.
- Fazer processamento por instância (arquivo csv)


- Remover colunas com mais de 10% de dados nulos (essas colunas já foram mapeadas no notebook de análise exploratória)
    - Se a coluna classe for removida, descartar a instância
        - Colocar um log com isso
- Remover colunas com baixa variância (essas colunas já foram mapeadas na análise exploratória)

- Implementar polynomialfeatures
- implementar função para atrasar variável em k janelas
- implementar função para calcular estatísticas de uma janela k (não será usado no agrupamento)
- implementar scaler nos dados
- implementar PCA junto com a explicabilidade em cada componente
- implementar K-MEANS
- implementar DBSCAN com GridSearch (usar nova métrica para avaliar e não a silhueta) -> deixar opção para usar silhueta

- Salvar imagem as duas componentes com o resultado real
- Salvar imagem as duas componentes com o agrupamento do algoritmo
- Salvar as métricas de agrupamento (silhouette e davies_bouldin)
- Deixar flag quem foi o melhor baseado no critério do davies_bouldin
- Salvar métrica de acurácia, precision, recall e F1 Score do agrupamento (Isso precisa ser investigado como fazer pq a priori não sei que classe é qual classe -> a primeiro é sempre zero!!!)


- Casos de teste para agrupamento
    - Poly 2; Atraso T-TPT; 4 componentes 
    - Poly 2; Sem atraso T-TPT; 4 componentes 
    - Sem transformações (Só com PCA)

-------------------------------------------
# Modelo conceitual (Passos):
- Leitura dos dados
    - Apenas pastas cujos nomes sejam 1, 2, 5, 6, 7 e 8 devem ser usadas; Tem períodos normais e períodos com anomalia.
    - Fazer processamento por instância (arquivo csv)
    - Fixar índice como timestamp
    - Trocar indices 2, 102, ... por 1???

- Pré-processamento (limpeza dos dados e transformação deles)
    - Remover colunas com mais de 10% de dados nulos (essas colunas já foram mapeadas no notebook de análise exploratória)
    - Se a coluna classe for removida, descartar a instância
        - Colocar um log com isso
    - Remover colunas com baixa variância (essas colunas já foram mapeadas na análise exploratória)

    - Implementar polynomialfeatures
    - implementar função para atrasar variável em k janelas: T-TPT
    - implementar função para calcular estatísticas de uma janela k (não será usado no agrupamento)
    - implementar scaler nos dados


- Treinamento dos modelo -> Separa isso em casos de teste (classes) ou fazer um painel de controle
    - implementar PCA junto com a explicabilidade em cada componente
    - implementar K-MEANS
    - implementar DBSCAN com GridSearch (usar nova métrica para avaliar e não a silhueta)

- Avaliação dos resultados
    - Calcular métrica de acurácia, precision, recall e **F1 Score** do agrupamento (armazer em um dataframe)
        - Ver como vai calcular, talvez precise fazer um pré-processamento
    - calcular métricas de agrupamento (silhouette e davies_bouldin) -> armazenar em um dataframe
    - Criar gráfico com as duas componentes (divisão real das classes e a divisão prevista pelo modelo) -> criar uma fig

- Salvar resultados (usar MLFlow para salvar imagens e as métricas)
    - Salvar as métricas
        - Criar um experimento por falha. Cada arquivo vai ter três casos de teste -> ver como MLFlow vai tratar esses casos (usar nested=True exemplo do Nunes)
    - Salvar as imagens

# Modelo lógico (classes e funções)

df_csv_files = search_file_csv(raw_path,)

for csv_file in df_csv_files["full_path"]:
    df_data = read_data(csv_file)
    try:
        report_multi_dataset(df_data, csv_file)
        modeling(df_data, csv_file)
    except ClassNull:
        #trata o erro se a coluna class for nula
        logger_to_csv(csv_file)

def search_file_csv(raw_path,
                    type_file="real",
                    abnormal_classes_codes=List[int])
     # retorna data apenas arquivos 1, 2, 5, 6, 7 e 8
     # dataframe tem o full_path e o path só com a pasta 1, 2, etc..
     # também também tem o número do poço se o dado for real.

def ClassNull(Exception):
    pass

def report_multi_dataset(df_data, csv_file):
    # faz um relatório do dataset que está indo para modelagem

def logger_to_csv():
    # salva em um arquivo csv quando a coluna classe for nula
    # coluna com o nome do arquivo e outra informando se coluna classe é nula

def read_data(csv_file: str)
    # faz a leitura do arquivo csv e converte indice para timestamp

def modeling(df_data: pd.DataFrame)

    df_preprocessing = preprocessing_transform(df_data)
    report_multi_dataset(df_data, csv_file) -> Faz o report após o pré-processamento

    metric_1 = case_1(df_preprocessing)
    # salva MlFlow

    metric_2 = case_2()
    # salva MlFlow

    metric_3 = case_3()
    # salva MlFlow

    case_better = better(metric_1, metric_2, metric_3)
    #salva tudo no MLFlow


def preprocessing_transform(df_data):
    # processamento comum entre todos os casos de teste
    # vai remover as colunas levantadas no notebook -> gravar essas colunas em alguma classe
    # remove colunas com mais de 10% de dados nulos
        # gera um erro se a coluna "class" for nula
    # remove colunas com baixa variância
    # remove indices duplicados
    #remove demais dados nulos -> dropna()

def case_1(df_preprocessing) -> davies_bouldin
    # não faz nenhuma traformação nos dados; apenas PCA
    # aplica PCA 4
    # cria gráficos
    # faz grid  com DBSCAN -> deixar opção de não fazer ele!!!
    # treina DBSCAN
    # salva as métricas
    # salva os gráficos
    pre_processing_metric() -> altera o target do DBSCAN. O primeiro é zero e os mais são 1
    calc_metric()
    return métricas

def case_2(df_preprocessing) -> davies_bouldin
    # faz poly grau 2
    # aplica PCA 4
    # cria gráficos
    # faz grid  com DBSCAN -> deixar opção de não fazer ele!!!
    # treina DBSCAN
    # salva as métricas
    # salva os gráficos
    pre_processing_metric() -> altera o target do DBSCAN. O primeiro é zero e os mais são 1
    calc_metric()
    return métricas

def case_3(df_preprocessing) -> davies_bouldin
    # faz poly grau 2
    # aplica atraso T-TPT
    # aplica PCA 4
    # cria gráficos
    # faz grid  com DBSCAN -> deixar opção de não fazer ele!!!
    # treina DBSCAN
    # salva as métricas
    # salva os gráficos
    pre_processing_metric() -> altera o target do DBSCAN. O primeiro é zero e os mais são 1
    calc_metric()
    return métricas

def better(...) -> metric
    min(...)