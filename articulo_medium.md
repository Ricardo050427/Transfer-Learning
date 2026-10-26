# Pokémon detection using convolutional neural networks and transfer learning

## Introduction

Identificar Pokémon ha sido un pasatiempo fascinante para millones de personas desde la llegada de la primera generación en 1996. Sin embargo, para quien recién empieza en este mundo, distinguir entre más de 150 criaturas distintas —muchas de ellas con formas, evoluciones y variantes muy similares— puede resultar una tarea abrumadora.

En este artículo exploraremos cómo construir un clasificador de Pokémon de principio a fin utilizando visión por computadora y aprendizaje profundo. Nuestro flujo de trabajo abarca cuatro etapas principales:

1. **Análisis de datos:** Exploración de un conjunto de datos grande y diverso.
2. **Preprocesamiento:** Redimensionamiento y aumento de datos (*Data Augmentation*).
3. **Construcción del modelo:** Transfer Learning utilizando **MobileNetV2**.
4. **Prueba en vivo:** Despliegue en una aplicación web interactiva estilo Pokédex pixel art para probar el modelo en tiempo real con la cámara de un celular o pegando imágenes desde la web.

---

## The data (Los datos)

Para este proyecto utilizamos el dataset disponible en Kaggle: [`mornville/pokemonimagedataset`](https://www.kaggle.com/datasets/mornville/pokemonimagedataset).

El dataset contiene más de 40,000 imágenes organizadas en **166 clases**, correspondientes a:
- Los **151 Pokémon originales** de la región de Kanto (desde Bulbasaur #001 hasta Mew #151).
- **15 Megaevoluciones** oficiales de esa misma generación (como Mega Charizard X/Y, Mega Gengar, Mega Mewtwo X/Y, etc.).

Cada carpeta de especie contiene decenas de imágenes con variaciones estilísticas extremas: ilustraciones oficiales 2D, modelos 3D de videojuegos, cartas del juego TCG, arte conceptual y dibujos de la comunidad. Esta diversidad representa un reto interesante, ya que obliga a la red neuronal a aprender características invariantes de la especie y no simplemente memorizar un estilo gráfico particular.

---

## Data preparation (Preparación de datos)

Para procesar las imágenes de forma eficiente y evitar desbordar la memoria RAM del sistema, redimensionamos todas las imágenes a un tamaño estándar de $224 \times 224$ píxeles con 3 canales de color (RGB).

Utilizamos la utilidad de Keras `image_dataset_from_directory`, la cual carga y procesa lotes de imágenes directamente desde el disco:

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)
```

En este paso configuramos dos detalles clave:
1. **División de datos:** Separamos un 20% para el conjunto de validación y conservamos una partición independiente de prueba (`test_dataset`) para evaluar el modelo al final con imágenes que jamás vio durante el entrenamiento.
2. **Prefetching con `tf.data.AUTOTUNE`:** Permite que la CPU prepare el siguiente lote de imágenes en segundo plano mientras la GPU calcula los gradientes del lote actual, eliminando cuellos de botella en la memoria.

---

## A small review of Neural Networks (Breve repaso de Redes Neuronales)

*En caso de que ya tengas experiencia con redes neuronales, puedes saltar directamente a la siguiente sección.*

Una red neuronal es un modelo computacional inspirado en la estructura y funcionamiento de las redes neuronales biológicas de nuestro cerebro. Consiste en capas de nodos interconectados (neuronas artificiales) que procesan información mediante entradas, pesos y salidas.

Cada nodo recibe uno o más valores de entrada, los multiplica por pesos numéricos asociados, suma un sesgo (*bias*) y pasa el resultado a través de una función de activación matemática (como ReLU o Sigmoide). El aprendizaje consiste en ajustar iterativamente esos pesos mediante el algoritmo de retropropagación (*backpropagation*) para minimizar el error entre la predicción de la red y la etiqueta real.

---

## Convolutional Neural Networks (Redes Neuronales Convolucionales)

Las redes neuronales tradicionales completamente conectadas (*Dense*) no son óptimas para procesar imágenes, ya que pierden la información espacial 2D y requieren millones de parámetros innecesarios. Aquí es donde entran las **Redes Neuronales Convolucionales (CNN)**.

La característica fundamental de una CNN es su capacidad para extraer automáticamente patrones visuales mediante **filtros o kernels**. Estos filtros se deslizan sobre la matriz de píxeles de la imagen realizando operaciones de convolución:
- En las primeras capas, los filtros detectan bordes, líneas, contrastes y esquinas.
- En capas intermedias, combinan bordes para reconocer formas simples (ojos, orejas, colas, textura de piel).
- En capas profundas, forman representaciones abstractas de alto nivel que permiten identificar a la criatura completa.

---

## Transfer Learning (Transferencia de Aprendizaje)

Entrenar una red convolucional profunda desde cero requiere semanas de cómputo en servidores de alto rendimiento y millones de imágenes. La **Transferencia de Aprendizaje (Transfer Learning)** resuelve este problema: tomamos una red ya entrenada por expertos en un conjunto gigantesco (como ImageNet, con más de 1.4 millones de imágenes de 1,000 categorías) y reutilizamos sus filtros visuales para nuestra tarea específica.

Para este proyecto seleccionamos **MobileNetV2**. Se trata de una arquitectura de 53 capas optimizada con bloques de cuello de botella residuales invertidos (*inverted residual bottlenecks*), lo que le permite mantener una alta precisión mientras consume una fracción de memoria y cómputo comparada con redes como ResNet o VGG.

---

## The actual model (La construcción del modelo)

El modelo se implementó en TensorFlow / Keras siguiendo una estrategia en dos fases:

### 1. Data Augmentation y Preprocesamiento
Para evitar el sobreajuste (*overfitting*), aplicamos capas de aumento de datos antes de ingresar a la base convolucional:

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])
```

### 2. Estructura de la Red
Instanciamos MobileNetV2 sin su clasificador original (`include_top=False`) y añadimos nuestro propio cabezal de clasificación:

```python
img_shape = (224, 224, 3)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=img_shape,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Congelamos la base en la Fase 1

inputs = tf.keras.Input(shape=img_shape, name="input_image")
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1./127.5, offset=-1.0)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.2, name="dropout")(x)
outputs = layers.Dense(166, activation="softmax", name="predictions")(x)

model = tf.keras.Model(inputs, outputs, name="Pokemon_MobileNetV2")
```

### 3. Fase 1: Extracción de Características (Feature Extraction)
Compilamos con el optimizador Adam ($lr = 0.001$) y pérdida de entropía cruzada. En esta fase, los pesos de MobileNetV2 permanecen intactos y solo se entrenan las capas finales durante 10 épocas.

### 4. Fase 2: Ajuste Fino (Fine-Tuning)
Descongelamos las capas superiores de MobileNetV2 a partir de la capa 100 y reentrenamos con una tasa de aprendizaje 10 veces menor ($lr = 0.0001$). Esto permite afinar los filtros convolucionales más profundos para adaptarlos a los detalles distintivos de los Pokémon.

---

## Results (Resultados)

Tras completar las dos fases de entrenamiento:

- **Precisión en Test:** **68.2%** sobre las 166 clases independientes de prueba.
- **Curvas de Rendimiento:** Las curvas de Accuracy y Loss mostraron una mejora drástica en el momento en que se activó el *Fine-Tuning*, reduciendo la pérdida de validación y estabilizando el aprendizaje.
- **Matriz de Confusión:** La diagonal principal de la matriz de 166x166 confirma una clasificación sólida. Los errores más frecuentes se concentran entre especies con similitudes morfológicas directas (por ejemplo, Pidgeotto confundido con Pidgeot o evoluciones que comparten paleta de colores).

---

## Live testing of the model (Prueba en vivo del modelo)

¿De qué sirve un modelo de inteligencia artificial si solo vive dentro de una libreta de Jupyter? Para llevarlo al mundo real, convertimos los artefactos a formatos de inferencia ligera:
- **TensorFlow.js (`model.json` y fragmentos binarios `.bin`):** Para ejecución web directa en JavaScript.
- **TensorFlow Lite (`model.tflite`):** Optimizado para dispositivos móviles y motores WebAssembly (XNNPACK).

Diseñamos una **aplicación web interactiva inspirada en la Pokédex clásica de Kanto en estilo pixel art (8-bit)** con las siguientes capacidades:

1. **Inferencia local en el navegador (Edge AI):** Sin enviar imágenes a ningún servidor externo, el modelo realiza la clasificación en entre 30 y 90 milisegundos directamente en la GPU del usuario.
2. **Cámara en vivo:** Con opción de alternar entre cámara frontal y trasera para apuntar a cartas, figuras o pantallas.
3. **Subida y Pegado con `Ctrl + V`:** Puedes buscar cualquier Pokémon en Google Imágenes, hacer clic en "Copiar imagen" y presionar `Ctrl + V` en la Pokédex para obtener su diagnóstico instantáneo.
4. **Detalles Retro:** Pantalla con efecto CRT de líneas de barrido (*scanlines*), audio sintetizado de 8-bit con la Web Audio API y descarga automática del sprite oficial y número de la Pokédex desde PokeAPI.

Al realizar pruebas en vivo:
- Pokémon emblemáticos con rasgos muy marcados (como Pikachu, Gengar o Blastoise) obtienen certezas superiores al **90%**.
- Criaturas con megaevoluciones identifican correctamente a la familia base con un desglose porcentual de la variante detectada.

---

## Conclusion (Conclusión)

Este proyecto demostró la efectividad de la Transferencia de Aprendizaje combinada con Fine-Tuning para abordar problemas de visión artificial con datasets grandes y múltiples clases. Partiendo de MobileNetV2, logramos un modelo capaz de distinguir entre 166 especies de Pokémon y ejecutar inferencia en tiempo real en navegadores web y dispositivos móviles.

Como trabajo futuro, la arquitectura puede ampliarse fácilmente a las generaciones posteriores (Johto, Hoenn, etc.) y compilarse como una aplicación móvil nativa utilizando el archivo `model.tflite` ya generado.

### Enlaces del Proyecto:
- **Repositorio de código en GitHub:** [https://github.com/Ricardo050427/Transfer-Learning](https://github.com/Ricardo050427/Transfer-Learning)
- **Despliegue Web:** [https://pokedex-ai-transfer-learning.netlify.app](https://app.netlify.com/drop)
