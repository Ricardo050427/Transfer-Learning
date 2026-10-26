# Clasificación de 166 Pokémon mediante Transfer Learning: De MobileNetV2 a una Pokédex Web Interactiva

## 1. Introducción y Planteamiento del Problema

La visión computacional ha avanzado a pasos agigantados gracias a las redes neuronales convolucionales. Sin embargo, entrenar un modelo desde cero para clasificar un conjunto de datos grande y con alta variabilidad visual requiere semanas de cómputo y millones de imágenes etiquetadas.

En este proyecto abordamos el reto de clasificar imágenes de Pokémon pertenecientes a la Primera Generación (Kanto), abarcando un total de **166 clases** (los 151 Pokémon originales más 15 megaevoluciones oficiales) y más de 40,000 imágenes.

Para resolver este desafío sin requerir semanas de entrenamiento, aplicamos **Transferencia de Aprendizaje (Transfer Learning)** utilizando como base la arquitectura **MobileNetV2** preentrenada en el conjunto de datos ImageNet.

---

## 2. El Dataset

El conjunto de datos utilizado proviene de Kaggle (`mornville/pokemonimagedataset`). Entre sus características destacan:

- **166 clases distintas:** Incluye desde Bulbasaur (#001) hasta Mew (#151), además de variantes Mega como Mega Charizard X/Y, Mega Gengar y Mega Mewtwo.
- **Diversidad de estilos:** Las imágenes contienen renders 3D, arte oficial 2D, cartas coleccionables, capturas de videojuegos y dibujos de la comunidad.
- **División de datos:** Se utilizó una partición estratificada del 80% para entrenamiento y 20% para validación, junto con un conjunto de prueba independiente (*test set*) para la evaluación final.

---

## 3. Metodología y Arquitectura del Modelo

### Preprocesamiento y Aumento de Datos (Data Augmentation)
Para evitar el sobreajuste (*overfitting*) y permitir que el modelo generalice ante fotos tomadas desde cámaras de celular o con fondos variados, implementamos una canalización de aumento de datos:
- Volteo horizontal aleatorio (`RandomFlip`).
- Rotación ligera de hasta un 20% (`RandomRotation`).
- Zoom aleatorio de hasta un 10% (`RandomZoom`).
- Normalización de píxeles al rango [-1, 1], requerido por la arquitectura MobileNetV2.

### Fase 1: Extracción de Características (Feature Extraction)
En esta primera etapa:
1. Cargamos **MobileNetV2** con sus pesos preentrenados en ImageNet y congelamos toda su base convolucional (`trainable = False`).
2. Agregamos en la parte superior:
   - Una capa de agrupamiento promedio global (`GlobalAveragePooling2D`).
   - Una capa de regularización `Dropout(0.2)` para prevenir co-adaptación de neuronas.
   - Una capa densa final con activación `Softmax` de 166 salidas.
3. Entrenamos únicamente el clasificador superior durante 10 épocas con el optimizador Adam y una tasa de aprendizaje base de $10^{-3}$.

### Fase 2: Ajuste Fino (Fine-Tuning)
Una vez que el clasificador superior aprendió a reconocer los patrones generales:
1. Descongelamos las capas superiores de MobileNetV2 a partir de la capa 100.
2. Reentrenamos la red completa durante 10 épocas adicionales reduciendo la tasa de aprendizaje a $10^{-4}$ ($lr / 10$) para adaptar los filtros convolucionales a los detalles específicos de los Pokémon sin destruir las características generales aprendidas en ImageNet.

---

## 4. Resultados y Evaluación

El modelo alcanzó una precisión de prueba (*test accuracy*) del **68.2%** sobre las 166 clases, un resultado sobresaliente considerando la granularidad del problema (clasificar entre 166 categorías visuales complejas).

### Métricas de Rendimiento:
- **Curvas de Entrenamiento:** La pérdida (*loss*) disminuyó de forma consistente tanto en entrenamiento como en validación. La fase de Fine-Tuning permitió dar un salto cualitativo en la precisión final.
- **Matriz de Confusión:** La diagonal principal muestra una fuerte concentración de aciertos. Las principales confusiones ocurren de forma natural entre evoluciones directas (por ejemplo, Pidgey vs. Pidgeotto) o entre la forma base y su megaevolución correspondiente.

---

## 5. De la Teoría a la Práctica: La Pokédex Web Interactiva

Para demostrar la utilidad práctica del modelo, no nos limitamos a un script de consola; construimos una **aplicación web interactiva inspirada en la Pokédex clásica de Kanto en estilo pixel art (8-bit)**.

### Características de la Aplicación:
- **Inferencia en el navegador (Edge AI):** El modelo fue exportado a formato binario cuantizado y se ejecuta localmente mediante **TensorFlow.js y WebAssembly (TFLite WASM)**. La inferencia toma únicamente entre 30 y 90 milisegundos por fotograma en cualquier navegador moderno.
- **Cámara en vivo:** Permite escanear cartas, pantallas o figuras en tiempo real, con soporte para cambiar entre cámara frontal y trasera en teléfonos móviles.
- **Subida y Pegado de Imágenes (Ctrl + V):** Permite copiar cualquier imagen de Google o recortar la pantalla y pegarla directamente en la aplicación para obtener su diagnóstico instantáneo sin tener que descargarla.
- **Detalles Retro:** Pantalla con efecto de líneas CRT (*scanlines*), gema óptica animada, sintetizador de audio retro de 8-bit con la Web Audio API y descarga automática del sprite oficial y número de la Pokédex desde PokeAPI.

---

## 6. Conclusiones y Enlaces

La transferencia de aprendizaje demostró ser la técnica ideal para abordar problemas de visión artificial con datasets grandes y múltiples clases, reduciendo drásticamente los recursos computacionales necesarios y logrando un desempeño robusto.

- **Repositorio en GitHub:** https://github.com/Ricardo050427/Transfer-Learning
- **Aplicación Web en Netlify:** [Desplegable en Netlify Drop]
- **Video de Demostración:** [Enlace al video demostrativo]
