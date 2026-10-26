#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Transfer Learning para Clasificación de Pokémon (166 Clases)
Arquitectura: MobileNetV2 preentrenada en ImageNet
Exportación: TensorFlow.js (model.json + .bin) para Netlify y App Móvil
=============================================================================
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path

# =============================================================================
# 1. GESTIÓN E INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS
# =============================================================================
DEPENDENCIAS_REQUERIDAS = [
    ("kagglehub", "kagglehub"),
    ("tensorflow", "tensorflow"),
    ("scikit-learn", "sklearn"),
    ("pandas", "pandas"),
    ("seaborn", "seaborn"),
    ("matplotlib", "matplotlib"),
    ("pillow", "PIL"),
]

def verificar_e_instalar_dependencias():
    """Comprueba que los paquetes requeridos estén instalados; si falta alguno, lo instala vía pip."""
    print("=" * 70)
    print("Comprobando entorno de dependencias...")
    faltantes = []
    import importlib.util
    for paquete, modulo in DEPENDENCIAS_REQUERIDAS:
        if importlib.util.find_spec(modulo) is None:
            faltantes.append(paquete)
            
    if faltantes:
        print(f"Instalando paquetes faltantes en el entorno: {faltantes}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *faltantes])
            print("Todas las dependencias se instalaron correctamente.")
        except subprocess.CalledProcessError as e:
            print(f"Error al instalar dependencias con pip: {e}")
            print("Si estás en Conda, puedes instalarlas con: conda install <paquete> o pip install <paquete>")
    else:
        print("Todas las dependencias necesarias ya se encuentran instaladas.")
    print("=" * 70)

verificar_e_instalar_dependencias()

# Importaciones posteriores a la verificación de paquetes
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Modo headless para ejecución en clústeres SLURM (sin servidor X11)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# 2. DESCARGA Y ALMACENAMIENTO LOCAL DE TODAS LAS IMÁGENES
# =============================================================================
def descargar_y_guardar_imagenes(destino_local="./data/pokemon"):
    """
    Descarga el dataset 'mornville/pokemonimagedataset' mediante kagglehub,
    resuelve la estructura anidada de directorios y copia/organiza todas las
    imágenes directamente en la carpeta local del proyecto.
    """
    print("\n" + "=" * 70)
    print("Descargando y organizando imágenes del dataset de Pokémon...")
    print("=" * 70)
    
    destino_path = Path(destino_local).resolve()
    train_dest = destino_path / "train"
    test_dest = destino_path / "test"
    
    # Si las carpetas ya existen y tienen múltiples clases de Pokémon, reutilizarlas
    if train_dest.exists():
        clases_existentes = [d for d in train_dest.iterdir() if d.is_dir()]
        if len(clases_existentes) > 10:
            print(f"Las imágenes ya se encuentran guardadas localmente en: {destino_path}")
            print(f"Detectadas {len(clases_existentes)} clases de Pokémon.")
            return train_dest, (test_dest if test_dest.exists() else None)

    import kagglehub
    print("Descargando dataset 'mornville/pokemonimagedataset' desde Kaggle...")
    path_cache = kagglehub.dataset_download("mornville/pokemonimagedataset")
    path_cache_dir = Path(path_cache)
    print(f"Descarga completada en caché: {path_cache_dir}")

    # Búsqueda recursiva para resolver subcarpetas anidadas como dataset/train
    rutas_train = list(path_cache_dir.glob("**/train"))
    rutas_test = list(path_cache_dir.glob("**/test"))

    if not rutas_train:
        raise FileNotFoundError(f"No se encontró carpeta 'train' dentro de: {path_cache_dir}")

    origen_train = rutas_train[0]
    origen_test = rutas_test[0] if rutas_test else None

    print(f"Origen de Train detectado: {origen_train}")
    print(f"Origen de Test detectado:  {origen_test}")

    # Crear directorios locales de destino
    destino_path.mkdir(parents=True, exist_ok=True)

    print(f"Copiando todas las imágenes hacia el directorio local: {destino_path} ...")
    if origen_train.exists():
        if not train_dest.exists():
            shutil.copytree(origen_train, train_dest)
            print(f"Carpeta de entrenamiento copiada exitosamente a: {train_dest}")
    
    if origen_test and origen_test.exists():
        if not test_dest.exists():
            shutil.copytree(origen_test, test_dest)
            print(f"Carpeta de prueba copiada exitosamente a: {test_dest}")
    elif not test_dest.exists():
        test_dest = None

    # Limpieza preventiva de archivos no estándar (por ejemplo .gif animados)
    extensiones_validas = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    eliminados = 0
    for archivo in list(destino_path.glob("*/*/*")):
        if archivo.is_file() and archivo.suffix.lower() not in extensiones_validas:
            archivo.unlink()
            eliminados += 1
    if eliminados > 0:
        print(f"Se filtraron y eliminaron {eliminados} archivos no estándar (ej. gifs animados).")

    # Resumen de imágenes guardadas
    clases_guardadas = [d.name for d in train_dest.iterdir() if d.is_dir()]
    total_imgs_train = len(list(train_dest.glob("*/*.*")))
    total_imgs_test = len(list(test_dest.glob("*/*.*"))) if test_dest else 0
    
    print("\nResumen del almacenamiento local:")
    print(f"  - Clases de Pokémon encontradas: {len(clases_guardadas)}")
    print(f"  - Imágenes en Train: {total_imgs_train}")
    print(f"  - Imágenes en Test:  {total_imgs_test}")
    print(f"  - Total global:      {total_imgs_train + total_imgs_test}")
    print("=" * 70 + "\n")
    
    return train_dest, test_dest

# =============================================================================
# 3. PIPELINE DE ENTRENAMIENTO TRANSFER LEARNING (MobileNetV2)
# =============================================================================
def ejecutar_pipeline(
    train_dir,
    test_dir=None,
    img_size=(224, 224),
    batch_size=32,
    initial_epochs=10,
    fine_tune_epochs=10,
    base_lr=0.001,
    export_dir="modelo_web"
):
    tiempo_inicio = time.time()
    
    # Comprobación de aceleradores GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Aceleración GPU activa ({len(gpus)}): {gpus}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print("Configuración de memoria dinámica GPU:", e)
    else:
        print("Entrenando en CPU (se recomienda GPU para tiempos reducidos).")

    # Identificación de nombres de clases reales
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    n_classes = len(class_names)
    print(f"\nNúmero total de clases para clasificación: {n_classes}")

    # Creación de datasets con tf.keras.utils.image_dataset_from_directory
    print("\nCargando datos en memoria...")
    train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="both",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    if test_dir and test_dir.exists():
        test_dataset = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        val_batches = tf.data.experimental.cardinality(val_dataset)
        test_dataset = val_dataset.take(val_batches // 2)
        val_dataset = val_dataset.skip(val_batches // 2)

    # Optimización del flujo de datos con prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # Data Augmentation y Preprocesamiento
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ], name="data_augmentation")

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Creación de la arquitectura funcional con MobileNetV2
    img_shape = img_size + (3,)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_shape, name="input_image")
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="Pokemon_MobileNetV2")

    # =========================================================================
    # FASE 1: FEATURE EXTRACTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("FASE 1: Extracción de Características (Base Convolucional Congelada)")
    print("=" * 70)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_dataset,
        epochs=initial_epochs,
        validation_data=val_dataset,
        callbacks=[early_stopping],
        verbose=1
    )

    # =========================================================================
    # FASE 2: FINE-TUNING
    # =========================================================================
    print("\n" + "=" * 70)
    print("FASE 2: Ajuste Fino (Fine-Tuning de Capas Superiores)")
    print("=" * 70)
    
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_lr / 10),
        metrics=['accuracy']
    )
    print(f"Capas base descongeladas a partir de la capa: {fine_tune_at}")
    print(f"Variables entrenables para Fine-Tuning: {len(model.trainable_variables)}")

    total_epochs = len(history.epoch) + fine_tune_epochs
    history_fine = model.fit(
        train_dataset,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1] + 1,
        validation_data=val_dataset,
        callbacks=[early_stopping],
        verbose=1
    )

    # =========================================================================
    # GUARDAR GRÁFICA DE ACCURACY Y LOSS
    # =========================================================================
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.axvline(x=len(history.epoch) - 1, color='green', linestyle='--', label='Inicio Fine-Tuning')
    plt.legend(loc='lower right')
    plt.title(f'Curvas de Accuracy ({n_classes} clases)')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.axvline(x=len(history.epoch) - 1, color='green', linestyle='--', label='Inicio Fine-Tuning')
    plt.legend(loc='upper right')
    plt.title('Curvas de Loss (Cross Entropy)')
    plt.ylabel('Loss')
    plt.xlabel('Época')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    grafica_acc_loss_path = "grafica_acc_loss_pokemon.png"
    plt.savefig(grafica_acc_loss_path, dpi=300)
    plt.close()
    print(f"\nGráfica de Accuracy y Loss guardada en: {grafica_acc_loss_path}")

    # =========================================================================
    # EVALUACIÓN EN TEST Y MATRIZ DE CONFUSIÓN
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUACIÓN EN CONJUNTO DE PRUEBA (TEST)")
    print("=" * 70)
    
    test_loss, test_acc = model.evaluate(test_dataset, verbose=1)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

    y_true = np.concatenate([y.numpy() for _, y in test_dataset])
    y_probs = model.predict(test_dataset, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)

    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, cmap='Blues', cbar=True, xticklabels=False, yticklabels=False)
    plt.title(f'Matriz de Confusión - Test ({n_classes} clases) | Accuracy: {test_acc:.3f}')
    plt.xlabel('Clase Predicha')
    plt.ylabel('Clase Real')
    plt.tight_layout()
    matriz_conf_path = "matriz_confusion_pokemon.png"
    plt.savefig(matriz_conf_path, dpi=300)
    plt.close()
    print(f"Matriz de confusión guardada en: {matriz_conf_path}")

    # =========================================================================
    # EXPORTACIÓN PARA NETLIFY Y APLICACIÓN MÓVIL (.bin y .json)
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPORTACIÓN DE ARTEFACTOS (TensorFlow.js para Netlify / Móvil)")
    print("=" * 70)
    
    os.makedirs(export_dir, exist_ok=True)
    
    # 1. Guardar modelo nativo en formato Keras
    ruta_keras = "pokemon_model.keras"
    model.save(ruta_keras)
    print(f"Modelo Keras guardado: {ruta_keras}")

    # 2. Guardar archivo class_names.json
    ruta_json_clases = os.path.join(export_dir, "class_names.json")
    with open(ruta_json_clases, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"Lista de clases guardada: {ruta_json_clases}")

    # 3. Exportación nativa a TensorFlow.js (model.json + group1-shard1of1.bin)
    # Construir modelo limpio de inferencia con Rescaling (sin capas de Data Augmentation para producción)
    try:
        base_m = model.get_layer("mobilenetv2_1.00_224") if "mobilenetv2_1.00_224" in [l.name for l in model.layers] else model.layers[3]
        dense_m = model.get_layer("predictions")
        infer_inputs = tf.keras.Input(shape=(224, 224, 3), name="input_1")
        x = tf.keras.layers.Rescaling(1./127.5, offset=-1.0, name="rescaling")(infer_inputs)
        x = base_m(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
        infer_outputs = dense_m(x)
        export_model = tf.keras.Model(infer_inputs, infer_outputs, name="Pokemon_Classifier")
    except Exception as e:
        print("Usando modelo directo para exportación:", e)
        export_model = model

    weights_manifest_entries = []
    binary_data = bytearray()
    for w in export_model.weights:
        w_name = w.name.rstrip(":0")
        w_arr = w.numpy().astype(np.float32)
        weights_manifest_entries.append({
            "name": w_name,
            "shape": list(w_arr.shape),
            "dtype": "float32"
        })
        binary_data.extend(w_arr.tobytes())

    bin_filename = "group1-shard1of1.bin"
    with open(os.path.join(export_dir, bin_filename), "wb") as f:
        f.write(binary_data)

    try:
        topo = json.loads(export_model.to_json())
    except Exception:
        topo = {"class_name": "Functional", "config": export_model.get_config()}

    # Asegurar compatibilidad de InputLayer con navegadores y TensorFlow.js
    def _fix_input_shapes(obj):
        if isinstance(obj, dict):
            if obj.get('class_name') == 'InputLayer':
                c = obj.get('config', {})
                bs = c.get('batch_shape') or c.get('batch_input_shape') or [None, 224, 224, 3]
                c['batch_input_shape'] = bs
                c['batchInputShape'] = bs
                c['inputShape'] = bs[1:]
            for v in obj.values():
                _fix_input_shapes(v)
        elif isinstance(obj, list):
            for item in obj:
                _fix_input_shapes(item)
    _fix_input_shapes(topo)

    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{getattr(tf.keras, '__version__', '3.x')}",
        "convertedBy": "Native Keras-to-TFJS Exporter",
        "modelTopology": topo,
        "weightsManifest": [
            {
                "paths": [bin_filename],
                "weights": weights_manifest_entries
            }
        ]
    }
    with open(os.path.join(export_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2)

    # Exportar también en formato TFLite
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
        tflite_bytes = converter.convert()
        with open(os.path.join(export_dir, "model.tflite"), "wb") as f:
            f.write(tflite_bytes)
        print("Formato TFLite (.tflite) generado con éxito.")
    except Exception as e:
        print("Nota sobre TFLite:", e)

    print(f"Conversión nativa exitosa en: {export_dir}/")
    for arch in sorted(os.listdir(export_dir)):
        print(f"  - {arch} ({os.path.getsize(os.path.join(export_dir, arch)) / 1024:.1f} KB)")

    # 4. Comprimir en archivo ZIP para despliegue
    zip_salida = shutil.make_archive("modelo_web_pokemon", "zip", export_dir)
    print(f"Carpeta comprimida lista para Netlify / Móvil: {zip_salida}")

    tiempo_total = (time.time() - tiempo_inicio) / 60
    print(f"\nTiempo total del pipeline: {tiempo_total:.2f} minutos.")
    print("=" * 70 + "\n")

# =============================================================================
# 4. PUNTO DE ENTRADA CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer Learning Pokémon Classifier")
    parser.add_argument("--save_images_only", action="store_true", help="Solo descarga y guarda las imágenes localmente sin entrenar")
    parser.add_argument("--data_dir", type=str, default="./data/pokemon", help="Directorio destino para guardar las imágenes")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--initial_epochs", type=int, default=10, help="Épocas de la fase 1 (feature extraction)")
    parser.add_argument("--fine_tune_epochs", type=int, default=10, help="Épocas de la fase 2 (fine-tuning)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate base")
    parser.add_argument("--export_dir", type=str, default="modelo_web", help="Carpeta de exportación para TensorFlow.js")
    args = parser.parse_args()

    train_dir, test_dir = descargar_y_guardar_imagenes(destino_local=args.data_dir)

    if args.save_images_only:
        print("Modo --save_images_only: Las imágenes se guardaron correctamente. Finalizando ejecución.")
        sys.exit(0)

    ejecutar_pipeline(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        initial_epochs=args.initial_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        base_lr=args.lr,
        export_dir=args.export_dir
    )
