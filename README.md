# NeuronCoders

## Tabla de contenidos

1. [Nombre](#Nombre)
2. [Descripción](#descripción)
3. [Fuente_del_Dataset](#Fuente del Dataset)
4. [Proceso](#Proceso)
5. [Funcionalidades](#Funcionalidades)
6. [Estado del proyecto](#EstadoDelProyecto)
7. [Agradecimientos](#Agradecimientos)


* Nombre del proyecto
Sistema de recomendación de recetas segun los ingredientes del usuario

* Descripción 
El Sistema de Recomendación de Recetas utiliza un modelo de vectorización TF-IDF y similitud de coseno para encontrar recetas similares en función de los ingredientes ingresados por el usuario. Se utiliza el conjunto de datos "recetas-cocina" de Hugging Face para entrenar el modelo inicial y proporcionar recomendaciones. Funciona como un chat bot donde se le introduce los ingredientes del usuario y te recomienda las 3 recetas mas cercanas segun tus ingredientes y las recetas guardadas en su dataset, tambien puedes añadir recetas nuevas al dataset.

* Fuente del Dataset
El dataset utilizado se carga desde Hugging Face, específicamente el conjunto de datos "recetas-cocina".

* Proceso:

Limpieza de Datos
Se eliminan filas con valores nulos y la columna 'uuid' del dataset.

Modelo de Machine Learning
Se utiliza un modelo de vectorización TF-IDF para convertir descripciones de ingredientes en vectores numéricos y se calcula la similitud de coseno entre los ingredientes del usuario y las recetas en el dataset.

Estadísticas y Métricas
Se utilizan métricas como la similitud de coseno para evaluar la similitud entre ingredientes del usuario y recetas recomendadas.

* Funcionalidades
Recomendación de Recetas: Permite al usuario ingresar ingredientes separados por coma para recibir recomendaciones de recetas similares.
Agregar Receta Personalizada: Los usuarios pueden agregar nuevas recetas al sistema, que se integran dinámicamente al dataset y recalculan la matriz TF-IDF para futuras recomendaciones.

* Estado del proyecto
Actualmente, el proyecto se encuentra en fase de desarrollo activo. Se están explorando mejoras adicionales como la integración con interfaces de usuario más robustas y la expansión del dataset para mejorar la precisión de las recomendaciones.

* Agradecimientos
Agradecemos a Hugging Face por proporcionar el dataset "recetas-cocina" y a Streamlit por la plataforma que facilita la creación de interfaces de usuario interactivas para este proyecto.
