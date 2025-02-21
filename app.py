import streamlit as st
import requests
import json
from datetime import datetime
import os
import ast
from typing import List, Dict
import zipfile
import io
import sqlite3
import tiktoken

# Database initialization
DB_NAME = "chat_db.sqlite"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT UNIQUE,
                  messages TEXT,
                  created_at DATETIME)''')
    conn.commit()
    conn.close()

init_db()  # Initialize database on startup

def list_conversations() -> list[str]:
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT name FROM conversations ORDER BY created_at DESC")
        conversations = [row[0] for row in c.fetchall()]
        conn.close()
        return conversations
    except Exception as e:
        st.error(f"Erro ao listar conversas: {str(e)}")
        return []

def load_conversation(conversation_name: str) -> list[dict]:
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT messages FROM conversations WHERE name = ?", (conversation_name,))
        result = c.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        else:
            st.warning(f"A conversa '{conversation_name}' não foi encontrada.")
            return []
    except Exception as e:
        st.error(f"Erro ao carregar conversa: {str(e)}")
        return []

def save_conversation(conversation_name: str, messages: list[dict]):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # Use UPSERT to update or insert
        c.execute('''INSERT OR REPLACE INTO conversations 
                     (name, messages, created_at) 
                     VALUES (?, ?, ?)''',
                  (conversation_name, 
                   json.dumps(messages), 
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        
        conn.commit()
        conn.close()
        st.success(f"Conversa salva como: {conversation_name}")
    except Exception as e:
        st.error(f"Erro ao salvar conversa: {str(e)}")

def count_tokens(text: str, model_name: str = "mistral") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except KeyError:
        return len(text.split())

class OpenRouterChat:
    def __init__(self, api_key: str, model: str, site_url: str, site_name: str):
        self.api_key = api_key
        self.model = model
        self.site_url = site_url
        self.site_name = site_name
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name,
        }
        self.context_dir = "contexts"
        self.knowledge_base_path = "knowledge_base.txt"
        self.suggestions_path = "suggestions.txt"
        os.makedirs(self.context_dir, exist_ok=True)

    def load_knowledge_base(self) -> str:
        try:
            with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def save_knowledge_base(self, content: str):
        with open(self.knowledge_base_path, "w", encoding="utf-8") as f:
            f.write(content)

    def load_suggestions(self) -> str:
        try:
            with open(self.suggestions_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def save_suggestions(self, content: str):
        with open(self.suggestions_path, "w", encoding="utf-8") as f:
            f.write(content)

    def generate(self, messages: list[dict]) -> str:
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

def analyze_code(code: str, knowledge_base: str) -> dict:
    analysis = {"syntax_errors": [], "improvements": []}

    try:
        ast.parse(code)
    except SyntaxError as e:
        analysis["syntax_errors"].append(str(e))

    try:
        tree = ast.parse(code)
        variables = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.add(target.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                variables.discard(node.id)

        if variables:
            analysis["improvements"].append(f"As variáveis {', '.join(variables)} foram definidas mas nunca utilizadas.")
    except Exception as e:
        analysis["improvements"].append(f"Erro durante a análise avançada: {str(e)}")

    if "não use variáveis globais" in knowledge_base.lower():
        if any(isinstance(node, ast.Global) for node in ast.walk(ast.parse(code))):
            analysis["improvements"].append("Evite o uso de variáveis globais conforme recomendado na base de conhecimento.")

    return analysis

def extract_zip(upload_file) -> str:
    temp_dir = "temp_project"
    os.makedirs(temp_dir, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(upload_file.read()), "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    return temp_dir

def generate_solution(code: str, filename: str, knowledge_base: str, model: str) -> dict:
    prompt = (
        f"Corrija o seguinte código Python do arquivo `{filename}` considerando as seguintes diretrizes:\n"
        f"{knowledge_base}\n\nCódigo original:\n{code}\n\n"
        "Forneça uma breve explicação do problema encontrado e sugira um código corrigido."
    )

    response = openrouter.generate([{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}])

    explanation, corrected_code = "", ""
    parts = response.split("\nCódigo corrigido:\n", 1)
    if len(parts) == 2:
        explanation, corrected_code = parts[0].strip(), parts[1].strip()
    else:
        explanation, corrected_code = response.strip(), ""

    return {"explanation": explanation, "corrected_code": corrected_code}

def generate_code_discussion(prompt: str, code: str, knowledge_base: str, model: str) -> str:
    full_prompt = (
        f"Você está discutindo o seguinte código Python:\n\n{code}\n\n"
        f"Considere as seguintes diretrizes:\n{knowledge_base}\n\n"
        f"Pergunta: {prompt}"
    )
    return openrouter.generate([{"role": "user", "content": full_prompt}, {"role": "assistant", "content": ""}])

# Interface principal
st.title("🤖 OpenRouter Chat Agent")
st.caption("Powered by LangChain and Mistral 24B")

# Barra lateral
with st.sidebar:
    st.header("Gerenciamento de Conversas")
    
    # Botão para nova conversa
    if st.button("Nova Conversa"):
        st.session_state.messages = []
        st.session_state.conversation_name = None
        st.experimental_rerun()

    conversations = list_conversations()
    selected_conversation = st.selectbox("Selecione uma conversa salva:", ["Nova Conversa"] + conversations)

    if selected_conversation != "Nova Conversa":
        if st.button("Carregar Conversa"):
            st.session_state.messages = load_conversation(selected_conversation)
            st.session_state.conversation_name = selected_conversation

    if st.button("Salvar Conversa Atual no Firebase"):
        if "conversation_name" in st.session_state and st.session_state.conversation_name:
            save_conversation(st.session_state.conversation_name, st.session_state.messages)
        else:
            conversation_name = st.text_input("Digite um nome para a conversa:")
            if conversation_name.strip():
                save_conversation(conversation_name, st.session_state.messages)
                st.session_state.conversation_name = conversation_name

    # Base de Conhecimento
    st.header("Base de Conhecimento")
    if "openrouter" in st.session_state:
        knowledge_base_content = st.session_state.openrouter.load_knowledge_base()
        st.markdown(f"**Conteúdo Atual:**\n{knowledge_base_content}")

    # Sugestões Temporárias
    st.header("Sugestões Temporárias")
    if "openrouter" in st.session_state:
        suggestions_content = st.session_state.openrouter.load_suggestions()
        st.text_area("Sugestões Salvas:", value=suggestions_content, height=200, disabled=True)

    # Seletor de Modelos
    st.header("Configuração do Modelo")
    selected_model = st.selectbox(
        "Selecione o modelo:",
        [
            "deepseek/deepseek-r1:free",
            "deepseek/deepseek-r1-distill-llama-70b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ]
    )

# Inicialização do OpenRouter
if "openrouter" not in st.session_state or st.session_state.selected_model != selected_model:
    st.session_state.openrouter = OpenRouterChat(
        api_key=st.secrets["OPENROUTER_API_KEY"],
        model=selected_model,
        site_url="https://chat.example.com",
        site_name="AI Chat"
    )
    st.session_state.selected_model = selected_model

openrouter = st.session_state.openrouter

# Estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_name" not in st.session_state:
    st.session_state.conversation_name = None

# Carregador de arquivos ZIP
uploaded_zip = st.file_uploader("Carregue um arquivo ZIP contendo sua pasta de projeto (.zip)", type=["zip"])
if uploaded_zip:
    if "uploaded_codes" not in st.session_state:
        st.session_state.uploaded_codes = {}

    temp_dir = extract_zip(uploaded_zip)
    st.info(f"Pasta do projeto extraída: `{temp_dir}`")

    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    code_content = f.read()
                    st.session_state.uploaded_codes[file] = code_content

                first_20_lines = "\n".join(code_content.splitlines()[:20])
                st.info(f"Arquivo carregado: `{file}`\n\nPrimeiras 20 linhas:\n```\n{first_20_lines}\n```")

    # Seleção de arquivo para análise
    selected_file = st.selectbox("Selecione o arquivo para análise:", list(st.session_state.uploaded_codes.keys()))

    if st.button("Gerar Solução para Arquivo Selecionado"):
        knowledge_base_content = openrouter.load_knowledge_base()
        code_content = st.session_state.uploaded_codes[selected_file]
        solution = generate_solution(code_content, selected_file, knowledge_base_content, openrouter.model)

        st.write(f"### Solução para `{selected_file}`:")
        st.write("#### Explicação:")
        st.write(solution["explanation"])

        if solution["corrected_code"]:
            st.write("#### Código Corrigido:")
            st.code(solution["corrected_code"], language="python")
            openrouter.save_suggestions(openrouter.load_suggestions() + "\n\n" + solution["explanation"])

# Exibição das mensagens
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(msg["content"])

# Entrada de chat
if prompt := st.chat_input("Pergunte algo sobre o código ou discuta melhorias"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        if "uploaded_codes" in st.session_state and st.session_state.uploaded_codes:
            filename, code_content = next(iter(st.session_state.uploaded_codes.items()))
            knowledge_base_content = openrouter.load_knowledge_base()
            response = generate_code_discussion(prompt, code_content, knowledge_base_content, openrouter.model)
        else:
            response = openrouter.generate(st.session_state.messages)

        full_response = response
        response_placeholder.write(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})