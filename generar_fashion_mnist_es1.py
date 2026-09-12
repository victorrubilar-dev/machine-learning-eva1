import numpy as np

# ------------------------------------------------------------
# Generador del dataset ES1 - Machine Learning
# Fuente: Fashion-MNIST (Keras)
# Salida: fashion_mnist_es1_6000.npz
# ------------------------------------------------------------

try:
    from keras.datasets import fashion_mnist
except ImportError:
    from tensorflow.keras.datasets import fashion_mnist

# 1) Cargar Fashion-MNIST.
# La primera ejecución puede descargar el dataset desde Internet.
(x_train, y_train), _ = fashion_mnist.load_data()

# 2) Crear una muestra equilibrada:
#    600 imágenes por cada una de las 10 clases = 6.000 imágenes.
rng = np.random.default_rng(42)

imagenes = []
etiquetas = []

for clase in range(10):
    indices = np.where(y_train == clase)[0]
    seleccion = rng.choice(indices, size=600, replace=False)
    imagenes.append(x_train[seleccion])
    etiquetas.append(y_train[seleccion])

X = np.concatenate(imagenes, axis=0).astype(np.uint8)
y = np.concatenate(etiquetas, axis=0).astype(np.uint8)

# 3) Mezclar las 6.000 observaciones manteniendo X e y asociados.
orden = rng.permutation(len(y))
X = X[orden]
y = y[orden]

# 4) Nombres de las clases.
clases = np.array([
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
])

# 5) Guardar comprimido.
np.savez_compressed(
    "fashion_mnist_es1_6000.npz",
    X=X,
    y=y,
    clases=clases
)

# 6) Verificación final.
print("Archivo creado: fashion_mnist_es1_6000.npz")
print("Forma de X:", X.shape)
print("Forma de y:", y.shape)
print("Tipo de X:", X.dtype)
print("Rango de píxeles:", X.min(), "a", X.max())

valores, cantidades = np.unique(y, return_counts=True)
print("\nDistribución por clase:")
for clase, cantidad in zip(valores, cantidades):
    print(f"Clase {clase}: {cantidad} imágenes")
