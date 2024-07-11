import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import streamlit as st

# Cargar el dataset desde Hugging Face
dataset = load_dataset("somosnlp/recetas-cocina")
df_train = pd.DataFrame(dataset['train'])

# Eliminar columna 'uuid' y eliminar filas con valores nulos
df_train = df_train.drop('uuid', axis=1)
df_train = df_train.dropna()

# Vectorizador TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_train['ingredients'].fillna(''))

# Función para buscar recetas por ingredientes
def buscar_recetas_por_ingredientes(df, lista_ingredientes):
    lista_ingredientes = [ing.lower().strip() for ing in lista_ingredientes.split(",")]
    
    def contiene_todos_ingredientes(ingredientes):
        return all(ing in ingredientes.lower() for ing in lista_ingredientes)
    
    df_filtrado = df[df['ingredients'].apply(contiene_todos_ingredientes)]
    
    if df_filtrado.empty:
        return pd.DataFrame()
    
    ingredientes_usuario = ', '.join(lista_ingredientes)
    user_tfidf = vectorizer.transform([ingredientes_usuario])
    
    tfidf_matrix_filtrado = vectorizer.transform(df_filtrado['ingredients'].fillna(''))
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix_filtrado).flatten()
    
    similar_indices = cosine_similarities.argsort()[-3:][::-1]
    
    recetas_recomendadas = df_filtrado.iloc[similar_indices]
    
    return recetas_recomendadas

def respuesta_Bot(df):
    if not df.empty:
        markdown_content = ""
        for idx, row in df.iterrows():
            markdown_content += f"## {row['title']}\n"
            markdown_content += f"**Ingredientes:** {row['ingredients']}\n\n"
            markdown_content += f"**Pasos:** {row['steps']}\n\n"
            markdown_content += f"**Link:** [{row['url']}]({row['url']})\n\n"
            markdown_content += "---\n"
        return markdown_content
    else:
        return "¡Uups!\n parece que no tenemos una receta para esos ingredientes, por favor ingrese otros."

# Interfaz de Streamlit
st.title("Sistema de recomendación de recetas según los datos del usuario")

# Inicializar historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.button("Clear"):
    st.session_state.messages = []
# Mostrar mensajes del historial
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message.get("avatar")):
        st.markdown(message["content"])

with st.chat_message("ai", avatar=":material/restaurant_menu:"):
    st.write("¡Bienvenido al sistema de recetas!\n Ingresa tus ingredientes y te recomendaré los mejores platillos.")

# Entrada de ingredientes del usuario
prompt = st.text_input("Escribe los ingredientes (separados por coma)")

if prompt:
    # Añadir mensaje del usuario al historial de chat
    st.session_state.messages.append({"role": "user", "avatar": ":material/person:", "content": prompt})
    
    # Buscar recetas
    recetas = buscar_recetas_por_ingredientes(df_train, prompt)

    # Mostrar mensaje del usuario
    with st.chat_message("user", avatar=":material/person:"):   
        st.markdown(prompt)
    
    # Mostrar respuesta del bot
    response = respuesta_Bot(recetas)
    with st.chat_message("ai", avatar=":material/restaurant_menu:"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "avatar": ":material/restaurant_menu:", "content": response})

# Opción para agregar una receta personalizada
with st.expander("Agregar una receta personalizada"):
    nombre_receta = st.text_input("Nombre de la receta", key="nombre_receta")
    ingredientes_receta = st.text_area("Ingredientes", key="ingredientes_receta")
    pasos_receta = st.text_area("Pasos", key="pasos_receta")

    if st.button("Agregar receta", key="agregar_receta"):
        if nombre_receta and ingredientes_receta and pasos_receta:
            # Crear un DataFrame temporal con la nueva receta
            nueva_receta = pd.DataFrame({
                "title": [nombre_receta],
                "ingredients": [ingredientes_receta],
                "steps": [pasos_receta],
                "url": [""],  
            })
            
            # Agregar la nueva receta al DataFrame principal
            df_train = pd.concat([df_train, nueva_receta], ignore_index=True)
            
            # Actualizar el vectorizador TF-IDF con los nuevos datos
            tfidf_matrix = vectorizer.fit_transform(df_train['ingredients'].fillna(''))
            
            st.success("Receta agregada correctamente.")
            
            # Mostrar las últimas recetas del dataset para verificar
            st.subheader("Últimas recetas agregadas:")
            st.dataframe(df_train.tail())  # Muestra las últimas filas del DataFrame actualizado
        else:
            st.warning("Por favor completa todos los campos para agregar la receta.")