# Clusterização de Textos em Documentos PDF
Este é um projeto Python para realizar a clusterização de textos extraídos de arquivos PDF. O objetivo é agrupar documentos com conteúdo textual semelhante, permitindo uma análise mais eficiente e organização dos dados. O projeto utiliza várias bibliotecas Python para processamento de texto e aprendizado de máquina.

## Dependências
Para executar este projeto, você precisará ter as seguintes bibliotecas Python instaladas:

- **PyPDF2**: Uma biblioteca para lidar com arquivos PDF.
- **numpy**: Biblioteca para computação numérica.
- **sklearn**: Biblioteca de aprendizado de máquina que inclui diversas ferramentas para pré-processamento e modelagem.
- **matplotlib**: Biblioteca para criação de gráficos.
- **tqdm**: Biblioteca para exibir barras de progresso em loops.
- **pandas**: Biblioteca para análise de dados.
- **nltk**: Biblioteca para processamento de linguagem natural.


## Funcionalidades

O projeto consiste nos seguintes passos principais:

1. **Extração de Textos de PDFs**: A função **pdf_text_extractor** é responsável por percorrer uma pasta de arquivos PDF e extrair o texto de cada página de cada PDF. Os textos extraídos são armazenados em uma lista.
2. **Pré-processamento de Texto**: Os textos extraídos são pré-processados para remover stopwords (palavras comuns que não contribuem significativamente para o significado do texto) e realizar tokenização. O resultado é uma lista de textos pré-processados.
3. **Agrupamento de Textos**: Os textos pré-processados são convertidos em uma matriz TF-IDF (Term Frequency-Inverse Document Frequency) e, em seguida, reduzidos em dimensão usando SVD (Decomposição em Valores Singulares Truncada). O algoritmo K-Means é aplicado para agrupar os textos em um número definido de clusters.
4. **Visualização de Clusters**: A função **plot_clusters** é utilizada para visualizar os clusters formados, projetando os dados em um gráfico bidimensional.
5. **Organização dos Documentos**: Após a clusterização, os documentos PDF são organizados em pastas separadas com base nos clusters aos quais foram atribuídos.
6. **Exportação de Dados**: As informações sobre os arquivos PDF, os clusters atribuídos e outros detalhes são armazenados em um DataFrame do pandas e exportados para um arquivo CSV.

# Execução
Antes de executar o código, certifique-se de ajustar os caminhos das pastas **folder_path** e **dest_folder_path** de acordo com a sua estrutura de diretórios. Além disso, as dependências necessárias precisam ser instaladas previamente.

Após executar o código, você será apresentado com um gráfico de dispersão dos clusters. Você pode avaliar se a clusterização é adequada e, se sim, os documentos serão organizados em pastas de acordo com os clusters formados.

Lembre-se de que este projeto é um exemplo básico de clusterização de textos em documentos PDF. Dependendo da natureza dos documentos e do objetivo do projeto, você pode precisar ajustar e expandir o código para obter melhores resultados.
