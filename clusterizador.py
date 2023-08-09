import os
import shutil
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Baixando as stop words do NLTK
nltk.download('stopwords')
nltk.download('punkt')

def read_files(folder_path, file_extension):
    """
    Gera caminhos de arquivos a partir de uma pasta com uma extensão específica.

    Args:
        folder_path (str): Caminho da pasta contendo os arquivos.
        file_extension (str): Extensão dos arquivos desejados.

    Yields:
        str: Caminho de um arquivo encontrado na pasta com a extensão especificada.
    """
    for file_name in tqdm(os.listdir(folder_path), desc='Lendo os arquivos .pdf na pasta'):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith(file_extension):
            with open(file_path, 'rb') as file:
                yield file_path        
        os.system('cls')

def pdf_text_extractor(folder_path):
    """
    Extrai texto de arquivos PDF presentes em uma pasta.

    Args:
        folder_path (str): Caminho da pasta contendo os arquivos PDF.

    Yields:
        list: Lista de textos extraídos de cada página de cada arquivo PDF.
    """
    documents = []
    for file_name in tqdm(os.listdir(folder_path),desc ="Armazenando os textos na lista Documentos"):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file_path)
                text = ''
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                documents.append(text)                
                os.system('cls')
    yield documents

def cluster_texts(documents, num_clusters=5):
    """
    Realiza clusterização de textos.

    Args:
        documents (list): Lista de textos a serem agrupados.
        num_clusters (int, optional): Número de agrupamentos desejados. O valor padrão é 5.

    Returns:
        numpy.ndarray: Array com rótulos de cluster para cada texto.
    """
    vectorizer = TfidfVectorizer(stop_words='english')  # 'english' stop words
    X = vectorizer.fit_transform(documents)
    
    # Check if vocabulary is empty
    if not vectorizer.vocabulary_:
        raise ValueError("Todos os documentos contém apenas stop words. Verifique seus documentos")
    
    svd = TruncatedSVD(n_components=num_clusters - 1)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    
    return kmeans.labels_

def plot_clusters(X, labels):
    """
    Gera um gráfico de dispersão dos clusters.

    Args:
        X (numpy.ndarray): Array de dados.
        labels (list): Lista de rótulos de cluster.
    """
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i in range(len(labels)):
        plt.scatter(X[i, 0], X[i, 1], c=colors[labels[i]], marker='o')
    
    plt.title('Clustered Documents')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

if __name__ == "__main__":
    # Pasta contendo os arquivos PDF
    folder_path = "W:/PUBLICA/Janus/12377 - PC-PP/APROVACAO/sem_movimentacao"
    # Pasta para onde os arquivos separados serão copiados
    dest_folder_path = "D:/projetos/clusteriza_textos"
    
    # carrega e processa os documentos
    documents = pdf_text_extractor(folder_path)
    
    # filtra os arquivos pdf
    pdf_files = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.pdf')]
    
    # Verifica se o número de arquivos PDF e o número de documentos extraídos é igual
    if len(pdf_files) != len(documents):
        print("Erro: Número de arquivos pdf e documentos extraídos não é igual")
        exit(1)
    
    # Tokenize e remove stopwords
    stop_words = stopwords.words('portuguese')
    
    cleaned_documents = []
    for doc in tqdm(documents, desc='Limpando Documentos'):
        words = word_tokenize(doc, language='portuguese')
        filtered_words = [word for word in words if word.lower() not in stop_words]
        cleaned_doc = ' '.join(filtered_words)
        cleaned_documents.append(cleaned_doc)
        os.system('cls')
    
    # Define o número de agrupamentos
    num_clusters = 4
    
    # Realiza o agrupamento
    try:
        labels = cluster_texts(cleaned_documents, num_clusters)
    except ValueError as e:
        print(e)
        exit(1)
    
    # Cria um dataFrame para armazenar a informação
    # data = {'FileName': pdf_files, 'Text': cleaned_documents, 'Cluster': labels}
    data = {'FileName': pdf_files,'Cluster': labels}
    df = pd.DataFrame(data)
    
    # Redução de dimensionalidade usando TruncatedSVD para plotagem
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(cleaned_documents)
    svd = TruncatedSVD(n_components=2)
    X = svd.fit_transform(X)
    
    # Plotagem dos agrupamentos
    plot_clusters(X, labels)

    decisao = input("O agrupamento está adequado? (s/n)")
    if decisao == 's':
        # Criar diretório para as pastas de clusters
        output_folder = os.path.join(dest_folder_path, 'docs_separados')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Criar pastas separadas para cada cluster
        for cluster_id in tqdm(range(num_clusters), desc="Criando pastas"):
            cluster_folder = os.path.join(output_folder, f'Cluster{cluster_id}')
            if not os.path.exists(cluster_folder):
                os.makedirs(cluster_folder)
            os.system('cls')

        # Mover os documentos para suas respectivas pastas de cluster
        for i, pdf_file in tqdm(enumerate(pdf_files),desc="Movendo documentos para as pastas dos clusters"):
            source_file_path = os.path.join(folder_path, pdf_file)
            cluster_id = labels[i]
            cluster_folder = os.path.join(output_folder, f'Cluster{cluster_id}')
            destination_file_path = os.path.join(cluster_folder, pdf_file)
            shutil.copy(source_file_path, destination_file_path)
            os.system('cls')

        print("Documentos copiados para as pastas de cluster.")

        # Salva o dataframe para o arquivo .csv
        csv_filename = "clustered_texts.csv"
        df.to_csv(os.path.join(dest_folder_path, csv_filename), index=False, sep=";")
        
    else:
        print('Nenhum arquivo foi separado.')
