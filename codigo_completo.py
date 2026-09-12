import os
import numpy as np
import matplotlib.pyplot as plt
os.environ["KERAS_BACKEND"] = "torch"  # Ejecuta sobre PyTorch por debajo
import keras
from keras import layers
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# 1. Cargar el dataset
datos = np.load("fashion_mnist_es1_6000.npz")
# Tomamos el dato de la etiqueta y
y = datos["y"]
# datos donde el modelo usa para aprender
X = datos["X"]
# clases definidas
clases = datos["clases"]

keras.utils.set_random_seed(42)

# 1. Preprocesamiento de los datos
# -1 indica la cantidad de datos la cual se ajusta automaticamente
# 28, 28 tamaño de la imagen
# 1 indica que es una imagen en escala de grises, como dato para una rgb se usaria 3
# por utlimo normalizamos los datos para que tenga un valor entre 0 y 1 diviendo entre 255
# esto es porque los pixeles de escalade grises contienen valores entre el 0 y 255 
X =  X.reshape(-1, 28, 28, 1) / 255.0

# 2. Separamos el entrenamiento en 80% (4800) de datos de entrenar y 20% (1200) en datos de test
# la variable y que es la etiqueta es la que queremos predecir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Configurar el modelo
# 3.1 primero se definen 2 capas donde la primera tiene 32 nucleos los cuales se encarga de encontrar caracteristicas en un tamaño de 3x3
# a esto le definimos la forma de los datos que es (28, 28, 1)
# 3.2 luego se reduce a un vector bidimensional con el fin de reducir el costo de computo
# 3.3 segunda neurona convulcional de 64 nucleos, aca busca areas mas grandes que la primera teniendo un mayor nivel de abastracción
# tambien se reduce a bidimensional
# 3.4 la 3 capa son 64 neuronas encuentra las relaciones entre los patrones formando figuras completas usando relu para relaciones
# no lineales complejas
# 3.5 capa final que decide la clase final cada neurona con una clase
# con softmax cada clase se le da un % de cual es mas probable de ser la clase correcta
# 3.6  despues de cada capa de neuronas se incluye un dropout, que es una tecnica
# que desactiva algunas neuronas dandole valor 0 generando que las neuronas restantes tengan que aprender a generalizar mejor y no depender de otras neuronas
# esto ayuda a evitar el sobreajuste

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        # Bloque convolucional 1
        layers.Conv2D(32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.25),  # Regularización suave en características visuales
        # Bloque convolucional 2
        layers.Conv2D(64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Dropout(0.25),
        # Clasificador
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),  # Regularización fuerte en la capa densa
        layers.Dense(10, activation="softmax"),
    ]
)


# compilamos el modelo antes del entrenamiento
# usando adam como funcion de gradiente, sparse_categorical_crossentropy como funcion de perdida
# y metrica de rendimiento accruracy
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Pasamos los datos del entrenamiento, con 20 epocas
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
)
# Predicción y reporte

predictions = model.predict(X_test)
y_pred = predictions.argmax(axis=1)
nombres_clases = [str(c) for c in clases]

# 3. Generar e imprimir el reporte
reporte = classification_report(
    y_test,
    y_pred,
    target_names=nombres_clases,
    digits=4,  
)

print(reporte)

# Configurar la cuadrícula de visualización (por ejemplo, 12 imágenes: 3 filas x 4 columnas)
num_rows = 3
num_cols = 4
num_images = num_rows * num_cols

plt.figure(figsize=(2.5 * num_cols, 3.0 * num_rows))

for i in range(num_images):
    # Obtener clase predicha, confianza y etiqueta real
    pred_label = np.argmax(predictions[i])
    confidence = 100 * np.max(predictions[i])
    true_label = y_test[i]

    # Color verde si acertó, rojo si falló
    color = "green" if pred_label == true_label else "red"

    # Graficar la imagen
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.title(
        f"Pred: {clases[pred_label]} ({confidence:.1f}%)\nReal: {clases[true_label]}",
        color=color,
        fontsize=9,
        pad=6,
    )
    plt.axis("off")

plt.tight_layout(h_pad=1.5)
plt.show()

# uso de imagen especifica
x_imagen = X_test[1]
y_imagen = y_test[1]
clase_imagen = clases[y_imagen] 

predict_imagen = model.predict(x_imagen.reshape(1, 28, 28, 1))
etiqueta_predicha = np.argmax(predict_imagen)

if etiqueta_predicha == y_imagen:
    print(f"La predicción es correcta: {clases[etiqueta_predicha]} ({etiqueta_predicha}), Real: {clases[y_imagen]} ({y_imagen})")
else:
    print(f"La predicción es incorrecta. Predicha: {clases[etiqueta_predicha]} ({etiqueta_predicha}), Real: {clases[y_imagen]} ({y_imagen})")


plt.figure(figsize=(3, 3))
plt.imshow(x_imagen.squeeze(), cmap="gray")
plt.title(
    f"Pred: {clases[etiqueta_predicha]} ({100 * np.max(predictions[1]):.1f}%)\nReal: {clases[y_imagen]} ({y_imagen})",
    color="green" if etiqueta_predicha == y_imagen else "red"
)
plt.axis("off")
plt.show()
