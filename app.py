import os
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import dotenv

''' COMO TREINAR O TEXTO BASEADO NO MODELO DE I.A UTILIZADO?
Gustavo Ferreira
Como treinar texto contextual baseado em um modelo de inteligência artificial?

Para treinar texto de contexto com base em um modelo de inteligência artificial (IA), normalmente você seguiria estas etapas:

1.Coleta e preparação de dados: reúna um grande e diversificado conjunto de dados de textos contextuais que sejam relevantes para o 
problema que você está tentando resolver. Este conjunto de dados deve ser representativo dos tipos de contextos que você espera encontrar 
em cenários do mundo real. Pré-processe os dados limpando-os e transformando-os em um formato que possa ser usado pelo modelo de IA. 
Isso pode envolver tarefas como tokenização, lematização e remoção de palavras irrelevantes.

2.Seleção de modelo: escolha uma arquitetura de modelo de IA apropriada para sua tarefa. Para tarefas de processamento de linguagem natural (PNL), 
as escolhas comuns incluem transformadores, redes neurais recorrentes (RNNs) e redes neurais convolucionais (CNNs).

    2.1 Transformadores: 
        • Encoder --> Decoder
        • utliza  rede neural (Feed Forward Neural Network)
        • otimiza a compreensão do significado das palavras dentro de uma frase com contexto (sequencias/frases anteriores)
    2.2 Redes Neurais Recorrentes (RNNs):
    2.3 Redes Neurais Convolucionais (CNNs):
        • Analisam o significado das palavras se baseando em estruturas convolucionais
        • Recebem token e inicial e produz token final
        • Tokens iniciais passam pelo processo de embeddings:
            - representações de valores ou objetos, como texto, imagens e áudio, que foram projetados para serem consumidos por modelos de aprendizado de máquina e algoritmos de pesquisa semântica)
            - possibilitam que os computadores entendam as relações entre palavras e outros objetos
            - Tecnicamente, os embeddings são vetores criados por modelos de aprendizado de máquina com a finalidade de capturar dados significativos sobre cada objeto.
            - são usados para representar o significado de palavras, frases e textos em um espaço de dimensões muito menores do que o espaço original.
        • Gate Linear Units

3.Treinando o modelo: treine o modelo selecionado no conjunto de dados preparado usando um algoritmo de aprendizado de máquina adequado. 
Isso pode envolver tarefas como extração de recursos, ajuste de hiperparâmetros e otimização de modelo. O objetivo é treinar o modelo para aprender 
padrões e relações dentro dos textos contextuais que podem ser usados para fazer previsões precisas ou gerar respostas relevantes.

4. Avliação e Validação: Avalie o desempenho do modelo treinado em um conjunto de dados de validação separado para garantir que ele esteja 
generalizando bem para dados novos e invisíveis. Isso pode envolver tarefas como cálculo de métricas como exatidão, precisão, recall e pontuação F1.

5. Implantação e aprendizado contínuo: depois que o modelo for treinado e validado, ele poderá ser implantado para uso em um aplicativo do mundo real. 
Contudo, o trabalho não para por aqui. Os modelos de IA são frequentemente melhorados ao longo do tempo através de um processo denominado aprendizagem 
contínua, onde são continuamente expostos a novos dados e retreinados para melhorar o seu desempenho. Isso pode envolver tarefas como aprendizagem incremental, 
aprendizagem ativa e aprendizagem por transferência.

No contexto do código fornecido, a API ChatGPT da OpenAI já está treinada em um grande conjunto de dados de textos de contexto e é capaz de gerar 
respostas relevantes às consultas dos usuários. O snippet de código fornecido demonstra como usar essa API para gerar respostas às perguntas dos 
usuários no contexto de um documento específico.
'''


# Configuração da API do OpenAI
dotenv.load_dotenv(dotenv.find_dotenv())
API_KEY = os.getenv("API_KEY")
openai.api_key = API_KEY

# Configuração do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

''' Função extrair_textos_dos_pdfs()

'''
# Função para converter múltiplos PDFs em texto puro
# Função para extrair texto puro de um PDF
def extrair_texto_do_pdf(caminho_pdf):
    texto = ""
    with open(caminho_pdf, 'rb') as arquivo:
        leitor_pdf = PyPDF2.PdfReader(arquivo)
        for pagina in leitor_pdf.pages:
            texto += pagina.extract_text()
    return texto

''' Função preprocessar_texto()

'''
# Pré-processamento de texto
def preprocessar_texto(texto):
    # Tokenização
    tokens = word_tokenize(texto.lower())
    # Remoção de stopwords, pontuações e tokens não alfanuméricos
    stopwords_en = set(stopwords.words('portuguese'))
    tokens_filtrados = [token for token in tokens if token.isalnum() and token not in stopwords_en]
    # Lematização
    lemmatizer = WordNetLemmatizer()
    tokens_lemmatizados = [lemmatizer.lemmatize(token) for token in tokens_filtrados]
    if not tokens_lemmatizados:
        # Se não houver nenhuma palavra alfanumérica após o pré-processamento, manter todas as palavras
        tokens_lemmatizados = [token for token in tokens if token.isalnum()]
    return ' '.join(tokens_lemmatizados)

''' Função construir_indice()

• Função que constrói um índice do conteúdo de um conjunto de documentos. A função usa a classe TfidfVectorizer do 
Scikit-learn para criar uma matriz de frequência de termo-frequência inversa de documento (TF-IDF), que é uma 
representação numérica do conteúdo de cada documento. A função então retorna o vetorizador TF-IDF e a matriz TF-IDF.

• Neste caso, a função está sendo utilizada para indexar um conjunto de documentos que contêm políticas de 
governança corporativa. A matriz TF-IDF será usada para encontrar o documento mais relevante para responder 
à pergunta de um usuário, com base na similaridade da pergunta com o conteúdo dos documentos indexados.

'''
# Construção do índice invertido
# Utilizar técnicas de PLN para indexar o conteúdo dos documentos, facilitando a recuperação de informações relevantes.
def construir_indice(documentos):
    tfidf_vectorizer = TfidfVectorizer()
    matriz_tfidf = tfidf_vectorizer.fit_transform(documentos)
    return tfidf_vectorizer, matriz_tfidf

''' Função buscar_documentos()
• Função que busca o documento mais relevante para responder uma pergunta com base em um conjunto de documentos indexados. 
Ele usa a classe TfidfVectorizer do Scikit-learn para criar uma matriz de frequência de documento inversa de frequência 
de termo (TF-IDF), que é uma representação numérica do conteúdo de cada documento. A função então retorna o vetorizador 
TF-IDF e a matriz TF-IDF.

• Neste caso, a função está sendo utilizada para indexar um conjunto de documentos de políticas de governança corporativa 
e encontrar o documento mais relevante para responder a um usuário com base na semelhança da pergunta do usuário com o 
conteúdo dos documentos indexados. A matriz TF-IDF será utilizada para encontrar o documento mais relevante para responder
à pergunta do usuário com base na ###semelhança de cosseno entre a pergunta e o conteúdo dos documentos.
'''
# Busca de documentos relevantes - UTILIZANDO EMBEDDINGS - (VETORIZAÇÃO DAS PALAVRAS DA PERGUNTA)
# Aproxima o vetor gerado da pergunta com os vetores do documento referência para retornar a resposta 
def buscar_documentos(pergunta, documentos, tfidf_vectorizer, matriz_tfidf):
    pergunta = preprocessar_texto(pergunta)
    pergunta_vetor = tfidf_vectorizer.transform([pergunta]) 
    similaridades = cosine_similarity(pergunta_vetor, matriz_tfidf)
    indice_doc_relevantes = similaridades.argmax()
    return documentos[indice_doc_relevantes]

''' Função gerar_resposta()
A função gerar_resposta() gera respostas usando a API ChatGPT da OpenAI. 
A função recebe duas entradas: a pergunta e o contexto. O contexto é o texto no qual o modelo é treinado, 
portanto deve ser relevante para a questão. 
O código usa a API OpenAI para gerar uma resposta fazendo uma solicitação à API com a pergunta e 
o contexto como prompts. A resposta é então devolvida ao usuário.
'''
# Geração de respostas com o ChatGPT
def gerar_resposta(pergunta, contexto):
    prompt = f"Contexto: {contexto}\n\nPergunta: {pergunta}\n\nResposta do modelo:\n"
    resposta = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=300,
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    resposta_texto = resposta.choices[0]['text'].strip()
    # Limpar contexto após cada pergunta
    contexto = ""
    # Remover parênteses e aspas
    resposta_texto = resposta_texto.strip('()"')
    return resposta_texto

''' Função principal()

''' 
# Função principal
def principal():
    # 1. Pré-processamento de Dados
    caminhos = [r'C:\Users\gustavo.ferreira\Desktop\back-end-ia-chatBot\Política de Alçadas - Algar S.A-V2-01jan2023.pdf',r'C:\Users\gustavo.ferreira\Desktop\back-end-ia-chatBot\Política de Alçadas - Algar Tech-V3-01jan_2023.pdf', r'C:\Users\gustavo.ferreira\Desktop\back-end-ia-chatBot\Política Corporativa de Remuneração-V1-01jan2023.pdf' ] # Substitua pelo caminho do diretório onde os PDFs estão localizados
    textos = [extrair_texto_do_pdf(caminho) for caminho in caminhos]
    
    # Pré-processamento dos textos dos documentos
    textos_preprocessados = [preprocessar_texto(texto) for texto in textos if texto.strip()]
    
    # 2. Indexação de Conteúdo
    tfidf_vectorizer, matriz_tfidf = construir_indice(textos_preprocessados)

    contexto = ""  # Inicializa o contexto como uma string vazia

    while True:
        # 3. Compreensão de Perguntas
        pergunta = input("Faça uma pergunta sobre os normativos corporativos (ou digite 'sair' para encerrar): ")
        if pergunta.lower() == 'sair':
            print("Até logo!")
            break

        # Adiciona o contexto à pergunta
        pergunta_com_contexto = f"{pergunta} {contexto}"

        # 4. Extração de Palavras-Chave
        palavras_chave = word_tokenize(pergunta_com_contexto.lower())

        # 5. Busca e Recuperação de Informações
        doc_relevante = buscar_documentos(pergunta_com_contexto, textos_preprocessados, tfidf_vectorizer, matriz_tfidf)

        # 6. Geração de Respostas
        resposta = gerar_resposta(pergunta_com_contexto, doc_relevante)
        print("Resposta:", resposta)

        # Atualiza o contexto com a última pergunta e resposta
        contexto += f" Pergunta: {pergunta} Resposta: {resposta}"

if __name__ == "__main__":
    principal()