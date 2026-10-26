# Clasificador de Pokémon con Transfer Learning

Este proyecto utiliza transferencia de aprendizaje (Transfer Learning) con la red MobileNetV2 para clasificar imágenes de Pokémon de la primera generación (166 clases, incluyendo megaevoluciones), entrenado a partir del dataset de Kaggle (mornville/pokemonimagedataset).

---

## Entregables del proyecto

A continuación se presentan todas las evidencias requeridas para la entrega del proyecto:

1. **Archivo fuente Python:**
   - [pokemon_transfer_learning.py](pokemon_transfer_learning.py): Script autónomo para entrenamiento, evaluación y exportación del modelo.
   - [Transfer_Learning_Pokemon.ipynb](Transfer_Learning_Pokemon.ipynb): Libreta interactiva paso a paso para Google Colab.

2. **Archivo SLRM:**
   - [script.slrm](script.slrm): Archivo de configuración para enviar el trabajo al clúster YUCA mediante SLURM con GPU.

3. **Gráfica de accuracy y loss, y matriz de confusión:**
   - `grafica_acc_loss_pokemon.png`: Curvas de aprendizaje durante la fase de extracción de características y fine-tuning.
   - `matriz_confusion_pokemon.png`: Matriz de confusión sobre las 166 clases del conjunto de prueba.

4. **Liga al artículo de Medium:**
   - [Artículo en Medium: Clasificación de 166 Pokémon mediante Transfer Learning](ENLACE_AL_ARTICULO_MEDIUM_AQUI)
   - Nota: El texto completo preparado para copiar y pegar en Medium se encuentra redactado en el archivo [articulo_medium.md](articulo_medium.md).

5. **Liga al proyecto en GitHub:**
   - https://github.com/Ricardo050427/Transfer-Learning

6. **Video utilizando la aplicación en celular:**
   - [Video demostrativo de clasificación en móvil](ENLACE_AL_VIDEO_AQUI)

7. **Archivo de ambiente Conda configurado:**
   - [environment.yml](environment.yml): Especificación de dependencias del entorno `pokemon_env` reproducible.

---

## Estructura del repositorio

- `Transfer_Learning_Pokemon.ipynb`: Libreta de Jupyter para Google Colab.
- `pokemon_transfer_learning.py`: Script de entrenamiento en Python.
- `script.slrm`: Script de lanzamiento para clúster YUCA.
- `environment.yml`: Archivo de configuración del entorno Conda.
- `articulo_medium.md`: Borrador completo listo para publicar en Medium.
- `pokedex_web/`: Aplicación web interactiva estilo Pokédex.
- `pokedex_netlify.zip`: Archivo comprimido listo para desplegar en Netlify.
- `modelo_web_pokemon/`: Archivos del modelo exportado para inferencia web y móvil.

---

## Cómo ejecutar el entrenamiento

### En Google Colab
1. Abre la libreta `Transfer_Learning_Pokemon.ipynb` en Google Colab.
2. Activa una GPU en Entorno de ejecución > Cambiar tipo de entorno > T4 GPU.
3. Ejecuta las celdas en orden. Al finalizar se generarán las gráficas y los archivos del modelo.

### En el clúster YUCA
1. Crea y activa el entorno Conda:
   ```bash
   conda env create -f environment.yml
   source activate pokemon_env
   ```
2. Envía el trabajo a la cola de SLURM:
   ```bash
   sbatch script.slrm
   ```
3. Revisa el estado del trabajo:
   ```bash
   squeue -u $USER
   tail -f salida_pokemon_*.log
   ```

---

## Aplicación Web (Pokédex)

Para probar el modelo interactivamente en tu computadora:
1. Abre una terminal dentro de la carpeta del proyecto.
2. Inicia un servidor local:
   ```bash
   python -m http.server 8080 --directory pokedex_web
   ```
3. Abre http://localhost:8080 en tu navegador.

Para usar la Pokédex desde tu teléfono celular con la cámara en vivo, puedes arrastrar la carpeta pokedex_web o el archivo pokedex_netlify.zip a https://app.netlify.com/drop para obtener un enlace público HTTPS en segundos.
