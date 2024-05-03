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

# Configuração da API do OpenAI
dotenv.load_dotenv(dotenv.find_dotenv())
API_KEY = os.getenv("API_KEY")
openai.api_key = API_KEY

# Configuração do NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

# Função para listar todos os arquivos PDF em um diretório
def listar_pdfs_em_diretorio(diretorio):
    return [os.path.join(diretorio, arquivo) for arquivo in os.listdir(diretorio) if arquivo.lower().endswith('.pdf')]

# Função para converter múltiplos PDFs em texto puro
def extrair_textos_dos_pdfs(diretorio):
    lista_caminhos_pdf = listar_pdfs_em_diretorio(diretorio)
    textos = []
    for caminho_pdf in lista_caminhos_pdf:
        texto = ""
        with open(caminho_pdf, 'rb') as arquivo:
            leitor_pdf = PyPDF2.PdfReader(arquivo)
            for pagina in leitor_pdf.pages:
                texto += pagina.extract_text()
        textos.append(texto)
    return textos

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

# Construção do índice invertido
def construir_indice(documentos):
    tfidf_vectorizer = TfidfVectorizer()
    matriz_tfidf = tfidf_vectorizer.fit_transform(documentos)
    return tfidf_vectorizer, matriz_tfidf

# Busca de documentos relevantes
def buscar_documentos(pergunta, documentos, tfidf_vectorizer, matriz_tfidf):
    pergunta = preprocessar_texto(pergunta)
    pergunta_vetor = tfidf_vectorizer.transform([pergunta]) 
    similaridades = cosine_similarity(pergunta_vetor, matriz_tfidf)
    indice_doc_relevantes = similaridades.argmax()
    return documentos[indice_doc_relevantes]

# Geração de respostas com o ChatGPT
def gerar_resposta(pergunta, contexto):
    prompt = f"Contexto: {contexto}\n\nPergunta: {pergunta}\n\nResposta do modelo:\n"
    resposta = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=300,
        temperature=0.5,  # Ajuste conforme a necessidade para mais ou menos criatividade
        top_p=1.0,  # Ajuste para controlar a diversidade da resposta
        frequency_penalty=0.5,  # Ajuste para desencorajar a repetição de palavras
        presence_penalty=0.5  # Ajuste para encorajar a introdução de novos conceitos
    )
    resposta_texto = resposta.choices[0]['text'].strip()
    # Limpar contexto após cada pergunta
    contexto = ""
    # Remover parênteses e aspas
    resposta_texto = resposta_texto.strip('()"')
    return resposta_texto

# Função principal
def principal():
    # 1. Pré-processamento de Dados
    diretorio_pdfs = r'C:\Users\gustavo.ferreira\Desktop\IA_chatBot\Políticas'  # Substitua pelo caminho do diretório onde os PDFs estão localizados
    textos = extrair_textos_dos_pdfs(diretorio_pdfs)
    
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
