import re
import pandas as pd
import numpy as np
import nltk #version 3.6.5
from nltk.corpus import stopwords
nltk.download('stopwords')
#importar librerías para visualización
import seaborn as sns #version. 0.11.2
import matplotlib.pyplot as plt #version. 3.4.
from sklearn.metrics import classification_report

def text_preprocessor(text):
    swe   = stopwords.words('english')
    text = text.lower()
    text = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', '', text).strip()
    text = re.sub(r'http\S+', '', text).strip()
    text = [w for w in text.split() if w not in swe]
    text = ' '.join(text)
    text = re.sub(r'[^\w\s]', '', text).strip()
    return text


def graficos_seniment_preproces(df, df_var, lugar_proces):
    """
    Gráfica la distribución de las variables 'sentiment' dentro de un dataframe.

        Parametros:
            df: DataFrame a graficar
            df_var: Data Frame con el nombre de la variable. Ej: df['nombre_de_la_variable']
            lugar_proces: Lugar en que ocurre el grafico en el preproceso, ejemplo: antes, despúes.
        
        Retorno:
            Gráfica la distribución de las variable 'sentiment'.
    """
    # Distribución de los sentimientos en twitter
    plt.figure(figsize=(12,6))
    ax = sns.countplot(x='sentiment', data=df, order=df_var.value_counts().index)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2., height + .5,
            '{:1.2f}'.format(height/len(df_var)), ha="center")
    plt.title(f'Distribución variable sentiment {lugar_proces} del preproceso')
    plt.show()


def graficar_frec_palabras (df,df_y, df_x):
    """
        Gráfica la distribución de las palabras más repetidas dentro de un dataframe.

            Parametros:
                df: DataFrame a graficar
                df_y: Data Frame con las palabras
                df_x: Data Frame con el número de frecuencias de las palabras
            
            Retorno:
                Gráfica la frecuencia de las palabras.
    """
    #seteamos el tamaño del gráfico que crearemos
    plt.figure(figsize=(20, 30))

    #creamos el gráfico de las palabras por frecuencia
    ax = sns.barplot(y=df_y,
                     x=df_x, data=df)
    plt.title(f'Frecuencias por palabras')
    #agregamos los totales al final de la barra
    for p in ax.patches:
        total = f'{int(p.get_width()):,}'.replace(',','.')
        x = p.get_x() + p.get_width() + 0.06
        y = p.get_y() + p.get_height()/2
        ax.annotate(total, (x, y))
    plt.show()



def compare_classifiers(estimators, X_test, y_test, n_cols=2):
    
    """
    Compara en forma gráfica las métricas de clasificación a partir de una lista de 
    tuplas con los modelos (nombre_modelo, modelo_entrendo) 
    """

    rows = np.ceil(len(estimators)/n_cols)
    height = 2 * rows
    width = n_cols * 5
    fig = plt.figure(figsize=(width, height))

    colors = ['dodgerblue', 'tomato', 'purple', 'orange']

    for n, clf in enumerate(estimators):

        y_hat = clf[1].predict(X_test)
        # Si las prediciones son probabilidades, binarizar
        if y_hat.dtype == 'float32':
            y_hat = [1 if i >= .5 else 0 for i in y_hat]

        dc = classification_report(y_test, y_hat, output_dict=True)

        plt.subplot(rows, n_cols, n + 1)

        for i, j in enumerate(['0', '1', 'macro avg']):

            tmp = {'0': {'marker': 'x', 'label': f'Class: {j}'},
                   '1': {'marker': 'x', 'label': f'Class: {j}'},
                   'macro avg': {'marker': 'o', 'label': 'Avg'}}

            plt.plot(dc[j]['precision'], [1], marker=tmp[j]['marker'], color=colors[i])
            plt.plot(dc[j]['recall'], [2], marker=tmp[j]['marker'], color=colors[i])
            plt.plot(dc[j]['f1-score'], [3], marker=tmp[j]['marker'],color=colors[i], label=tmp[j]['label'])
            plt.axvline(x=.5, ls='--')

        plt.yticks([1.0, 2.0, 3.0], ['Precision', 'Recall', 'f1-Score'])
        plt.title(clf[0])
        plt.xlim((0.1, 1.0))

        if (n + 1) % 2 == 0:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
    fig.tight_layout()
    
    return