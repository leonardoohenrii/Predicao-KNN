#### Bibliotecas 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#### Dataset
dataset_cancer = pd.read_csv("D:/green_lake/Projetos/cancer/dataset/Breast_cancer_data.csv")

#### Formatação Dataset
### Renomear colunas 
dataset_cancer.rename(columns = {"mean_radius" : "media_radioterapia",
                                 "mean_texture" : "media_textura",
                                 "mean_perimeter" : "media_perimetro",
                                 "mean_area" : "media_area",
                                 "mean_smoothness" : "media_suavidade",
                                 "diagnosis" : "diagnostico"}, inplace = True)

#### ML KNN 
dataset_cancer_diagnostico_a = dataset_cancer.drop("diagnostico", axis = 1).values # Passa o campo diagnostico para como matriz para uma nova variável
dataset_cancer_diagnostico_b = dataset_cancer["diagnostico"].values 

dataset_cancer_a_treino, dataset_cancer_a_teste, dataset_cancer_b_treino, dataset_cancer_b_teste = train_test_split(dataset_cancer_diagnostico_a, dataset_cancer_diagnostico_b,
                                                                                                                    train_size = 0.25, random_state = 42)                                  

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(dataset_cancer_a_treino,dataset_cancer_b_treino)

predicao_treino = knn.predict(dataset_cancer_a_treino)
predicao_teste = knn.predict(dataset_cancer_a_teste)

matrix_confusao = confusion_matrix(dataset_cancer_b_teste, predicao_teste)
sns.heatmap(matrix_confusao, annot = True)

