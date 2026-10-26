# Clasificador de Pokémon con Transfer Learning

Este proyecto utiliza transferencia de aprendizaje (Transfer Learning) con la red MobileNetV2 para clasificar imágenes de Pokémon de la primera generación (166 clases, incluyendo megaevoluciones).

El entrenamiento se realizó a partir del dataset de Kaggle (mornville/pokemonimagedataset) y cuenta con una aplicación web estilo Pokédex para probar el modelo en tiempo real con la cámara, subiendo fotos o pegando imágenes desde el portapapeles.

## Estructura del proyecto

- Transfer_Learning_Pokemon.ipynb: Libreta interactiva de Jupyter para entrenar en Google Colab.
- pokemon_transfer_learning.py: Script en Python para ejecutar el entrenamiento completo en terminal o clúster.
- script.slrm: Archivo de configuración para enviar el trabajo al clúster YUCA mediante SLURM.
- environment.yml: Archivo con las librerías necesarias para crear el ambiente en Conda.
- pokedex_web/: Aplicación web interactiva inspirada en una Pokédex retro.
- pokedex_netlify.zip: Archivo comprimido listo para arrastrar y publicar en Netlify.
- modelo_web_pokemon/: Archivos del modelo exportado para la web y móvil (model.json, group1-shard1of1.bin, model.tflite y class_names.json).

## Cómo ejecutar el entrenamiento

### En Google Colab
1. Abre la libreta Transfer_Learning_Pokemon.ipynb en Google Colab.
2. Activa una GPU en Entorno de ejecución > Cambiar tipo de entorno > T4 GPU.
3. Corre las celdas en orden. Al terminar generará las gráficas de rendimiento y los archivos del modelo.

### En el clúster YUCA
1. Crea y activa el ambiente de Conda:
   ```bash
   conda env create -f environment.yml
   source activate pokemon_env
   ```
2. Envía el trabajo a la cola de SLURM:
   ```bash
   sbatch script.slrm
   ```
3. Revisa el avance del trabajo:
   ```bash
   squeue -u $USER
   tail -f salida_pokemon_*.log
   ```

## Aplicación Web (Pokédex)

Para probar el modelo de forma interactiva en tu computadora:
1. Abre una terminal dentro de la carpeta del proyecto.
2. Inicia un servidor local rápido:
   ```bash
   python -m http.server 8080 --directory pokedex_web
   ```
3. Abre http://localhost:8080 en tu navegador.

Para usarla en el celular con la cámara o compartirla en internet, puedes arrastrar la carpeta pokedex_web o el archivo pokedex_netlify.zip a https://app.netlify.com/drop.
