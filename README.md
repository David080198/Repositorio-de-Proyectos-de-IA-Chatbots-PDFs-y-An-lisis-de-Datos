# Proyecto de Chatbots y Agentes Interactivos con Llama3

## Descripción de los Proyectos

**Proyecto 1: Chatbot Interactivo con Texto a Voz**  
Esta aplicación es un chatbot basado en el modelo Llama3 de Ollama. Utiliza el motor de texto a voz **pyttsx3** para convertir las respuestas del chatbot en audio, que se reproduce automáticamente en el navegador. La interfaz en **Streamlit** permite seleccionar diferentes voces para personalizar la experiencia. El chatbot ofrece una conversación fluida y natural, guardando el historial de chat para mantener una conversación continua.

![Chatbot Interactivo](Repositorio-de-Proyectos-de-IA-Chatbots-PDFs-y-An-lisis-de-Datos/proyecto1.png) <!-- Añade aquí la ruta de tu imagen -->

**Proyecto 2: Agente de Ayuda Inteligente para PDFs**  
Desarrollé un agente de ayuda inteligente que responde preguntas sobre el contenido de un archivo **PDF** cargado por el usuario. Utiliza el modelo Ollama **LLM (Llama3)** y una base de datos de vectores gestionada con **Chroma** para proporcionar respuestas relevantes basadas en el contenido del PDF. Los usuarios pueden cargar un PDF, actualizar los embeddings y hacer preguntas específicas para recibir respuestas basadas en el documento.

**Proyecto 3: Sistema de Consulta y Análisis de Datasets**  
Este sistema permite a los usuarios cargar archivos **Excel** y realizar consultas sobre los datos. Con el modelo **Llama3**, el sistema interpreta preguntas en lenguaje natural sobre el contenido del archivo, como obtener valores máximos o realizar operaciones con datos del DataFrame. La interfaz en **Streamlit** facilita la carga del archivo, las consultas y proporciona ejemplos de formatos válidos para obtener resultados precisos.

**Proyecto 4: Agente Reactivo para Ejecución de Código Python**  
Desarrollé una aplicación interactiva con **Streamlit** que integra herramientas de **LangChain** y **ChatOllama**. Este agente reactivo responde a comandos generando y ejecutando código Python en tiempo real. El proyecto incluye funcionalidades para guardar, cargar y limpiar el historial de consultas, además de ejemplos predefinidos y la opción de descargar el historial en un archivo de texto.

**Proyecto 5: Generador de Recetas Personalizadas**  
La aplicación Generador de Recetas permite crear recetas personalizadas a partir de una lista de ingredientes. Utilizando modelos avanzados como **Ollama** y **Llava**, la aplicación genera una receta detallada y una imagen fotorrealista del plato final. Los usuarios pueden descargar la receta y la imagen en un archivo **PDF** para almacenamiento y compartición.

**Proyecto 6: Chatbot Personalizado con Llama3**  
Este proyecto es un chatbot interactivo desarrollado con **Streamlit** y el modelo **Llama3.1** de **Ollama**. El chatbot actúa como un asistente virtual personalizado, con nombre y descripción definidos por el usuario. Ofrece una conversación contextual, realiza preguntas para conocer mejor al usuario y actualiza el historial de chat en tiempo real para una experiencia dinámica.

## Requisitos Generales

- **Python 3.7+**
- **Streamlit**
- **Ollama Llama3** (o **Llama3.1** para el Proyecto 6)
- **pyttsx3** (Proyecto 1)
- **Chroma** (Proyecto 2)
- **PyPDF2** (Proyecto 2)
- **Pandas** (Proyecto 3)
- **LangChain** (Proyecto 4)
- **ChatOllama** (Proyecto 4)
- **Llava** (Proyecto 5)

## Nota sobre Docker y Docker Compose

Para cada uno de estos proyectos, es necesario tener **Docker** y **Docker Compose** instalados para crear y gestionar las imágenes y contenedores necesarios para la ejecución de la API. Asegúrate de tener ambos instalados y configurados adecuadamente para facilitar el despliegue y la gestión de los entornos de ejecución.

