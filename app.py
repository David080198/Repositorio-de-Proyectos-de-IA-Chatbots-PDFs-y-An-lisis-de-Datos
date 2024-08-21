import streamlit as st
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from gtts import gTTS
from io import BytesIO
import base64
import pyttsx3
from PyPDF2 import PdfFileReader

# Librerias para proyecto 2
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community import embeddings
from langchain.chains import LLMChain
import streamlit as st
import tempfile
import shutil
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("LANGSMITH_API_KEY")
# Funciones proyecto 2

def actualizar_embeddings(uploaded_file):
    if uploaded_file:
        # Crear un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            # Guardar el archivo en el sistema de archivos temporal
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name  # Obtener la ruta del archivo temporal

        # Mostrar la ruta del archivo temporal (opcional)
        st.write(f"Ruta del archivo temporal: {temp_path}")
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        #print("Documentos cargados:", docs)

        # dividir el texto en fragmentos mas pequeños
        text_splinter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=50)
        chunked_documents = text_splinter.split_documents(docs)
        #embeddings = OllamaEmbeddings(model="nomic-embed-text")

        vector_db = Chroma.from_documents(
            documents = chunked_documents, 
            embedding = embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
            persist_directory="./chroma_db",
            collection_name='rag_chroma'
        )

        return vector_db


# Título de la aplicación
st.title("Aplicación de Proyectos")

# Opciones en la barra lateral
option = st.sidebar.selectbox(
    "Selecciona un proyecto",
    ["Proyecto 1", "Proyecto 2", "Proyecto 3","Proyecto 4","Proyecto 5","Proyecto 6"]
)

# Descripción de cada proyecto
descriptions = {
    "Proyecto 1": "Esta aplicación es un chatbot basado en el modelo Llama3 de Ollama. Utiliza el motor de texto a voz pyttsx3 para convertir las respuestas del chatbot en audio, que se reproduce automáticamente en el navegador. La interfaz de usuario en Streamlit permite seleccionar diferentes voces para personalizar la experiencia de conversación. El chatbot responde a los mensajes del usuario de manera fluida y natural, proporcionando una interacción más inmersiva. La aplicación guarda el historial de chat para permitir una conversación continua y sin interrupciones.",
    "Proyecto 2": "Este proyecto consiste en un agente de ayuda inteligente que responde preguntas relacionadas con el contenido de un archivo PDF cargado por el usuario. Utiliza el modelo Ollama LLM (llama3) y una base de datos de vectores gestionada con Chroma para encontrar las respuestas más relevantes basadas en el contexto del PDF. El usuario puede cargar un PDF, actualizar los embeddings, y luego hacer preguntas que el agente responderá estrictamente basándose en la información contenida en el documento.",
    "Proyecto 3": "Este proyecto es un Sistema de consulta y análisis de datasets que permite a los usuarios cargar archivos Excel y hacer preguntas sobre los datos cargados. Utilizando un modelo de lenguaje como llama3, el sistema puede interpretar y responder consultas específicas sobre el contenido del archivo, como obtener valores máximos de columnas o realizar operaciones con datos del DataFrame. La interfaz, desarrollada con Streamlit, ofrece una experiencia interactiva donde los usuarios pueden cargar un archivo Excel, realizar consultas en lenguaje natural, y recibir respuestas precisas basadas en los datos del archivo. Además, el sistema proporciona ejemplos de formatos de consultas válidos y explica cómo estructurar preguntas correctamente para obtener los resultados deseados.",
    "Proyecto 4": "Este proyecto es una aplicación interactiva desarrollada con Streamlit, que integra herramientas de LangChain y ChatOllama para crear un agente reactivo capaz de responder a comandos de usuario generando y ejecutando código Python en tiempo real. El agente sigue instrucciones específicas para utilizar siempre la herramienta de ejecución de código, incluso cuando conoce la respuesta. Además, el proyecto incluye funcionalidades para guardar, cargar y limpiar el historial de consultas realizadas por el usuario, y ofrece ejemplos predefinidos para demostrar las capacidades del agente. La aplicación también permite al usuario descargar el historial de consultas en un archivo de texto.",
    "Proyecto 5": "El proyecto Generador de Recetas es una aplicación interactiva que permite a los usuarios crear recetas personalizadas a partir de una lista de ingredientes ingresada. Utilizando modelos avanzados de lenguaje y generación de imágenes, como Ollama y Llava, la aplicación no solo genera una receta detallada con instrucciones paso a paso, sino que también crea una imagen fotorrealista del plato final. Además, ofrece la opción de descargar la receta y la imagen en un archivo PDF, facilitando su almacenamiento y compartición.",
    "Proyecto 6": "Este proyecto desarrolla un chatbot interactivo utilizando Streamlit y el modelo de lenguaje "llama3.1" de Ollama. El chatbot se configura como un asistente virtual personalizado, cuyo nombre y descripción pueden ser definidos por el usuario. A través de una interfaz sencilla, los usuarios pueden mantener una conversación fluida con el asistente, quien responde de manera contextual y realiza preguntas para conocer mejor al usuario. El historial de chat se actualiza en tiempo real, proporcionando una experiencia dinámica y envolvente.",
}

# Mostrar contenido según la opción seleccionada
if option == "Proyecto 1":
    st.sidebar.header("Descripción del Proyecto")
    st.sidebar.write(descriptions.get(option, "Descripción no disponible"))

    model = OllamaLLM(model="llama3")

    st.title('Chatbot con llama3')
    messages = [("system","""Eres un chatbot AI util, te llamas David, tienes que hablar con si fueras humano, nunca reveles que eres un chatbot""")]

    #def text_to_speech(text, lang='es'):
    #    tts = gTTS(text=text, lang=lang)
    #    audio_file = BytesIO()
    #    tts.write_to_fp(audio_file)
    #    audio_file.seek(0)
    #    return audio_file

    def text_to_speech(text, voice_id=None, rate=150):
        engine = pyttsx3.init()
        # Obtener y mostrar todas las voces disponibles
        voices = engine.getProperty('voices')
        for i, voice in enumerate(voices):
            print(f"Voice {i}: {voice.name} - ID: {voice.id}")
        # Establecer la voz según el ID proporcionado
        if voice_id:
            engine.setProperty('voice', voice_id)
        
        # Establecer la velocidad de habla
        engine.setProperty('rate', rate)
        
        audio_file = BytesIO()
        engine.save_to_file(text, 'output.mp3')
        engine.runAndWait()
        
        with open('output.mp3', 'rb') as f:
            audio_data = f.read()
        
        return audio_data


    # Obtener y mostrar todas las voces disponibles
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    available_voices = {voice.name: voice.id for voice in voices}

    # Selección de voz
    selected_voice_name = st.selectbox("Elige una voz:", list(available_voices.keys()))
    selected_voice_id = available_voices[selected_voice_name]

    # Inicializamos historial del chat

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes de chat del historial al recargar la app
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Reaccionar a la entrada del usuario
    if prompt := st.chat_input("Escribe un mensaje..."):
        # Mostrar mensaje del usuario en el contenedor del chat
        st.chat_message("user").markdown(prompt)
        # Agregar mensaje del usuario al historial del chat
        st.session_state.messages.append({"role":"user","content":prompt})
        messages.append(["human",prompt])
        print(model.invoke(messages))
        response = model.invoke(messages)
        audio_bytes = text_to_speech(response, voice_id=selected_voice_id)
        # Encode the audio to base64
        audio_base64 = base64.b64encode(audio_bytes).decode()

        # Prepare the JavaScript code to play the audio automatically
        audio_html = f"""
            <html>
            <body>
                <audio id="audio" style="display:none;">
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                    Your browser does not support the audio element.
                </audio>
                <script>
                    var audioElement = document.getElementById('audio');
                    audioElement.play().catch(error => {{
                        console.log('Playback prevented:', error);
                    }});
                </script>
            </body>
            </html>
            """

            # Inject the HTML and JavaScript into the Streamlit app
        st.components.v1.html(audio_html, height=0)

        with st.chat_message("assistant"):
        # Mostrar respuesta del asistente en el contenedor de mensajes del chat
            st.markdown(response)
        # Agregar resoyesta del asistente al historial del chat
        st.session_state.messages.append({"role":"assistant","content":response})

elif option == "Proyecto 2":
    st.sidebar.header("Descripción del Proyecto")
    st.title('Sistema de preguntas de PDF cargado')
    st.sidebar.write(descriptions.get(option, "Descripción no disponible"))
    st.write("Este agente responde preguntas relacionados con el PDF cargado")
    #st.image("https://via.placeholder.com/300x200", caption="Imagen del Proyecto 2")
    #st.write("Descripción del Proyecto 2: Aquí puedes poner detalles, imágenes o cualquier otro contenido relevante.")
    # Botón para cargar archivo PDF
    uploaded_file = st.file_uploader("Elige un archivo PDF", type="pdf")
    st.write(uploaded_file)
    if st.button("Actualizar Embeddings"):
        vector_database = actualizar_embeddings(uploaded_file)
        st.write("Embeddings actualizados correctamente")
    prompt_template = """Eres un agente de ayuda inteligente especializado en el archivo PDF cargado.
    Responde las preguntas de los usuarios {input} relacionados con el archivo PDF basandote estrictamente en el {context} proporcionado
    No hagas suposiciones ni proporciones información que no esté incluida en el {context}"""
    
    llm = OllamaLLM(model="llama3")
    qa_chain = LLMChain(llm = llm, prompt=PromptTemplate.from_template(prompt_template))
    pregunta = st.text_area("Haz tu pregunta sobre el archivo pdf que cargaste")
    if st.button("Enviar"):
        if pregunta:
            vectordb = Chroma(persist_directory="./chroma_db",
                              embedding_function=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'))
            resultados_similares = vectordb.similarity_search(pregunta, k=5)
            contexto = ""
            for doc in resultados_similares:
                contexto += doc.page_content
            respuesta = qa_chain.invoke({"input":pregunta, "context":contexto})

            resultado = respuesta["text"]

            st.write(resultado)
        else:
            st.write("Por favor, inmgresa una pregunta antes de enviar")


elif option == "Proyecto 3":
    st.sidebar.header("Descripción del Proyecto")
    #cargando librerias del proyecto
    from langchain_ollama import ChatOllama
    from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    from langchain.agents.agent_types import AgentType
    from langchain.schema.output_parser import StrOutputParser
    from langchain.output_parsers import PandasDataFrameOutputParser
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    import pandas as pd
    import numpy as np
    from langchain.llms import ollama
    import os
    model_name ="llama3"
    st.title('Sistema de consulta y análisis de datasets')
    st.sidebar.write(descriptions.get(option, "Descripción no disponible"))
    st.write("Este es el contenido del Proyecto 3.")
    uploaded_file = st.file_uploader("Elige un archivo Excel", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        model = OllamaLLM(model="llama3")
        llm_model = ChatOllama(model=model_name,temperature=0,verbose=True,streaming=True)
        df_parser = PandasDataFrameOutputParser(dataframe=df)
        print(df_parser.get_format_instructions())

        # Definición del prompt
        prompt = PromptTemplate(
            template="Responde al usuario su consulta. \n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": df_parser.get_format_instructions()},
        )

        chain = prompt | llm_model | df_parser
        resultado = chain.invoke(
            {
                "query": "Give me the max value of column Age"
            }
        )

        #pandas_df_agent = create_pandas_dataframe_agent(llm=llm_model, df = df, verbose=True,
        #                                                handle_parsing_errors = True,
        #                                                return_intermediate_steps=True,
        #                                                allow_dangerous_code=True)

        st.write("""Como ejemplo, para los formatos:
                1. La cadena "column:num_legs" es una instancia con formato correcto que obtiene la columna num_legs, donde num_legs es una columna posible.
                2. La cadena "row:1" es una instancia con formato correcto que obtiene la fila 1.
                3. La cadena "column:num_legs[1,2]" es una instancia con formato correcto que obtiene la columna num_legs para las filas 1 y 2, donde num_legs es una columna posible.
                4. La cadena "row:1[num_legs]" es una instancia con formato correcto que obtiene la fila 1, pero solo para la columna num_legs, donde num_legs es una columna posible.

                5. La cadena "mean:num_legs[1..3]" es una instancia con un formato correcto que toma la media de num_legs de las filas 1 a 3, donde num_legs es una columna posible y mean es una operación válida de Pandas DataFrame.
                6. La cadena "do_something:num_legs" es una instancia con un formato incorrecto, donde do_something no es una operación válida de Pandas DataFrame.
                7. La cadena "mean:invalid_col" es una instancia con un formato incorrecto, donde invalid_col no es una columna posible.""")
        pregunta = st.text_area("Haz tu pregunta sobre el archivo xlsx que cargaste")
        try:
            if st.button("Enviar"):
                resultado = chain.invoke({"query": pregunta})
                st.write(resultado)
        except:
            st.write("Hay un problema, el modelo no entendio tu pregunta")
        
    else:
        st.write("Debes cargar antes un archivo")

elif option == "Proyecto 4":
    st.sidebar.header("Descripción del Proyecto")
    # Importando librerias que se utilizaran para el proyecto 4
    import streamlit as st
    from langchain_experimental.tools import PythonREPLTool
    from langchain import hub
    from langchain_ollama import ChatOllama
    from langchain.agents import create_react_agent
    from dotenv import load_dotenv
    from langchain.agents import AgentExecutor
    import datetime
    import os
    model_name = "mistral"


    def save_history(question,answer):
        with open("history.txt", "a") as f:
            f.write(f"{datetime.datetime.now()}: {question} -> {answer}\n")

    def load_history():
        if os.path.exists("history.txt"):
            with open("history.txt", "r") as f:
                return f.readlines()
        return []
    
    def clear_history():
        if os.path.exists("history.txt"):
            open("history.txt", "w"). close()

    
    st.title('Agente de Python Interactivo para códigos Python')
    st.sidebar.write(descriptions.get(option, "Descripción no disponible"))
    st.write("Este es el contenido del Proyecto 4.")
    st.markdown(
        """
        <style>
        .stApp { backgraound-color:black;}
        .tittle { color: # ff4b4b; }
        .button {background-color: #ff4b4h; color: white; border-radius: 5px; }
        .input { border: 1px solid #ff4b4b; border-radius: 5px; }
        </style>
        """,
        unsafe_allow_html=True
        
    )

    instrucciones = """
    - Siempre usa la herramienta incluso si sabes la respuesta.
    - Debes usar siempre código Python para responder.
    - Eres un agente que puede escribir código.
    - Solo responde la pregunta escribiendo código, incluso si sabes la respuesta. 
    - Si no sabes la respuesta, responde "no se la respuesta".
    """

    st.markdown("###Instrucciones")
    st.markdown(instrucciones)
    base_prompt = hub.pull("langchain-ai/react-agent-template",api_key=api_key)
    prompt = base_prompt.partial(instructions = instrucciones)
    st.write("Prompt cargado...")

    tools = [PythonREPLTool()]
    llm = ChatOllama(model=model_name,temperature=0,verbose=True,streaming=True,api_key=api_key)
    agente = create_react_agent(llm = llm, tools=tools,prompt=prompt)
    agente_ejecutor = AgentExecutor(agent=agente, tools=tools, verbose=True)

    st.markdown("### Ejemplos:")
    ejemplos = ["Calcula la suma de 2 y 3.", 
                "Genera una lista de numeros del 1 al 100.",
                "Crear una función que calcule el factoria de un número.",
                "Crea un juego basico con pygame."]
    example = st.selectbox("Selecciona un eemplo", ejemplos)

    if st.button("Ejecutar ejemplo"):
        user_input = example
        respuesta = agente_ejecutor.invoke(input={"input":user_input})
        st.markdown("#### Respuesta del agente:")
        st.code(respuesta["output"], language='python')
        save_history(user_input, respuesta["output"])

    user_input = st.text_input("Introduce tu pregunta o comando para el agente: ", key ="input_text")
    col1, col2 = st.columns(2)
    with col1: 
        ejecutar = st.button("Ejecutar", key="execute_button")
    with col2:
        limpiar = st.button("Limpiar historial", key= "clear_button")

    if ejecutar:
        if user_input:
            respuesta = agente_ejecutor.invoke(input={"input":user_input})
            st.markdown("### Respuesta del agente:")
            st.code(respuesta["output"], language='python')
            save_history(user_input, respuesta["output"])
        else:
            st.warning("Por favor introduce una pregunta o comando")

    if limpiar:
        clear_history()

    st.markdown("### Historial de consultas: ")
    historial = load_history()
    if historial:
        for h in historial:
            st.text(h)
    else:
        st.write("No hay historial de consultas")

    with open("history.txt","r") as f:
        if st.download_button("Descargar historial",
                              f, file_name="historial_consultas.txt"):
            st.write("Descarga exitosa")
elif option == "Proyecto 5":
    # Importacion de librerias
    import streamlit as lt
    from langchain_ollama.llms import OllamaLLM
    from dotenv import load_dotenv, find_dotenv
    from fpdf import FPDF
    from PIL import Image
    import requests 
    from io import BytesIO
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.llms import Ollama
    #load_dotenv()
    #os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
    st.sidebar.header("Descripción del Proyecto")
    st.title('Generador de Recetas')
    st.sidebar.write(descriptions.get(option, "Descripción no disponible"))
    st.write("Este es el contenido del Proyecto 5.")

    load_dotenv(find_dotenv(), override=True)

    #client = ChatOllama(model="llama3")

    def generate_recipe(ingredients):
        system_prompt = '''
            Eres un chef de primera clase. '''
        user_prompt_template = f'''
            Crea una receta detallada basada únicamente en los siguientes ingredientes: {', '.join(ingredientes)}.
            Por favor, formatea la receta de la siguiente manera: 
            Titulo de la receta: 

            Ingredientes de la Receta con tamaño y porción:

            Lista de Instrucciones para esta receta: 
        '''
        print(system_prompt)
        print(user_prompt_template)
        print("*********************************")
        client = OllamaLLM(model="llama3")
        prompt = PromptTemplate(
            input_variables=["ingredients"],
            template=user_prompt_template
        )
        llm_chain = LLMChain(
            llm=client,
            prompt=prompt
        )

        # Prepara los ingredientes en un formato adecuado
        ingredients_list = ', '.join(ingredients)

        recipe = llm_chain.run(ingredients=ingredients_list)
        print("Resultado del texto")
        print(recipe)
        return recipe
    
                #response = client.chat.completions.create(message=[
        #    {'role': 'system', 'content':system_prompt},
        #    {'role': 'user', 'content': user_prompt}
        #],
        #max_tokens = 1020,
        #temperature = 0.9
        #)

        #response = ChatPromptTemplate.from_messages(
        #    [
        #        ("system",{system_prompt}),
        #        ("user","Question:{ingredients}")
        #    ]
        #)
        #print("paso el otro")
        #llm=Ollama(model="llama3")
        #output_parser=StrOutputParser()
        #chain=response|llm|output_parser
       
        #print(chain)
        #return chain
    
    def obtener_nombre_receta(texto):
        lineas = texto.splitlines()
        return lineas[1]
    
    def generar_imagen(titulo_receta):
        client = OllamaLLM(model="llava")
        prompt = f'''
        Crea una imagen fotorrealista del plato final titulado "{titulo_receta}".
        El plato debe estar bellamente representado en un plato de cerámica con un enfoque cercano en las texturas y colores de los ingredeintes.
        La ambientación debe ser sobre una mesa de madera con iluminación natural para resaltar las caracteristicas apetitosas de la comida. 
        Asegurate de que la imagen capture los colores ricos y vibrantes y los detalles intrincados de la comida, haciendola parecer recien preparada y lista para comer.
        '''
        response = client.images.generate(
            model = 'llava',
            prompt = prompt,
            style = 'vivid',
            size = '1024x1024',
            quality = 'standard',
            n = 1
        )

        return response.data[0].url
    

    def save_to_pdf(titulo_receta,receta, imagen_url):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial",size=12)

        pdf.set_font("Arial", style= 'B',size=16)
        pdf.cell(0,10,txt=titulo_receta,ln=True,align='C') 

        response = requests.get(imagen_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_path = f"{titulo_receta.replace(' ', '_')}.jpg"
        img.save(img_path, format="JPEG")

        pdf.ln(10)
        img_width = 190
        pdf.image(img_path, x=(pdf.w - img_width) / 2, w=img_width, type='JPEG')
        pdf.ln(10)

        pdf.set_font(family="Arial", size=12)
        for line in receta.split('\n'):
            pdf.multi_cell(w=0,h=10,txt = line)

        pdf_file = "receta.pdf"
        pdf.output(pdf_file)

        return pdf_file
        
    st.write("Ingrese un ingrediente para generar una receta personalizada")
    ingredientes = st.text_input("Ingredientes (separados por comas)")
    if st.button("Generar Receta"):
        print("Si funciono")
        ingredientes_list = [ing.strip() for ing in ingredientes.split(",")]
        print(ingredientes_list)
        receta = generate_recipe(ingredientes_list)
        st.session_state.receta = receta
        titulo_receta = obtener_nombre_receta(receta)
        st.session_state.titulo_receta = titulo_receta
        #imagen_receta = generar_imagen1(titulo_receta)
        #st.session_state.imagen_receta = imagen_receta

    if 'receta' in st.session_state:
        st.write(f"{st.session_state.titulo_receta}")
        st.write(st.session_state.receta)
        #st.image(st.session_state.imagen_receta, caption=st.session_state.titulo_receta)

        if st.button("Descargar recetas en PDF"):
            pdf_file = save_to_pdf(st.session_state.titulo_receta,
                                   st.session_state.receta)
            with open(pdf_file,"rb") as f:
                st.download_button(label="Descargar PDF",
                                   data = f, file_name=pdf_file,
                                   mime = "application/pdf")
elif option == "Proyecto 6":
    # Librerias
    import streamlit as st
    from langchain_ollama.llms import OllamaLLM
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    st.sidebar.header("Descripción del Proyecto")
    st.title('Chat con Llama 3.1')
    st.sidebar.write(descriptions.get(option, "Descripción no disponible"))
    st.write("Este es el contenido del Proyecto 6.")
    llm = OllamaLLM(model="llama3.1:latest")
    bot_name = st.text_input("Nombre del asistente virtual:", value="Bot")
    prompt = f"""Eres un asistente virtual te llamas {bot_name}, respondes preguntas con respuestas simples, además debes preguntas al usuario acorte al contexto del chat, también debes preguntarle al usuario para conocerlo  """
    bot_description = st.text_area("Descripción del asistente virtual: ", value=prompt)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", bot_description),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}"),
        ]
    )

    chain = prompt_template | llm

    user_input = st.text_input("Escribe tu pregunta: ", key="user_input")   

    if st.button("Enviar"):
        if user_input.lower() == "adios":
            st.stop()
        else:
            response = chain.invoke({"input":user_input,
                                     "chat_history":st.session_state["chat_history"]})
            st.session_state["chat_history"].append(HumanMessage(content=user_input))
            st.session_state["chat_history"].append(AIMessage(content=response))
       
    chat_display  = ""
    for msg in st.session_state["chat_history"]:
        if isinstance(msg, HumanMessage):
            chat_display += f"🧍Humano: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            chat_display += f"🤖 {bot_name}: {msg.content}\n"

    st.text_area("Chat", value=chat_display, height=400,key="chat_area")