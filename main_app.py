import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from streamlit_drawable_canvas import st_canvas

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="MNIST Clasificador", layout="wide")

st.title("✍️ Clasificador de Dígitos (MNIST)")
st.write("App interactiva para reconocer números escritos a mano.")

# -----------------------------
# LOAD DATA (cache)
# -----------------------------
@st.cache_data
def load_data():
    digits = load_digits()
    X = digits.data / 16.0
    y = digits.target
    return X, y

X, y = load_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Configuración")

model_option = st.sidebar.selectbox(
    "Modelo",
    ["KNN", "SVM", "Árbol"]
)

test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2)

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# -----------------------------
# MODEL
# -----------------------------
if model_option == "KNN":
    k = st.sidebar.slider("K vecinos", 1, 5, 3)
    model = KNeighborsClassifier(n_neighbors=k)

elif model_option == "SVM":
    c = st.sidebar.slider("C", 0.1, 5.0, 1.0)
    model = SVC(C=c)

elif model_option == "Árbol":
    depth = st.sidebar.slider("Profundidad", 1, 10, 5)
    model = DecisionTreeClassifier(max_depth=depth)

# -----------------------------
# TRAIN (subset para rapidez)
# -----------------------------
subset = 10000  # importante para que no se demore
X_train_sub = X_train[:subset]
y_train_sub = y_train[:subset]

model.fit(X_train_sub, y_train_sub)

y_pred = model.predict(X_test[:2000])

accuracy = accuracy_score(y_test[:2000], y_pred)

# -----------------------------
# METRICS
# -----------------------------
st.subheader("📊 Métricas")

st.write(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test[:2000], y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=False, cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
st.pyplot(fig)

# -----------------------------
# EJEMPLO DE IMAGEN
# -----------------------------
st.subheader("🖼️ Ejemplo del dataset")

idx = st.slider("Selecciona una imagen", 0, 1000, 0)

img = X_test[idx].reshape(8, 8)

fig2, ax2 = plt.subplots()
ax2.imshow(img, cmap='gray')
ax2.set_title(f"Etiqueta real: {y_test[idx]}")
ax2.axis("off")

st.pyplot(fig2)

pred = model.predict([X_test[idx]])[0]
st.success(f"Predicción del modelo: {pred}")

# -----------------------------
# DIBUJAR NÚMERO
# -----------------------------
st.subheader("✏️ Dibuja un número")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:

    img = canvas_result.image_data[:, :, 0]

    # resize a 28x28
    img_resized = Image.fromarray(img).resize((8, 8)) 
    img_array = np.array(img_resized)

    # normalizar
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, -1)

    st.image(img_resized, caption="Imagen procesada", width=150)

    if st.button("Predecir dibujo"):
        pred = model.predict(img_array)[0]
        st.success(f"🔢 Predicción: {pred}")

# -----------------------------
# EXPLICACIÓN
# -----------------------------
st.subheader("📘 Explicación")

st.markdown("""
- **MNIST** contiene imágenes de números del 0 al 9.
- Cada imagen es de 28x28 píxeles.
- El modelo aprende patrones de píxeles para clasificar.

### Modelos:
- **KNN**: compara con ejemplos similares
- **SVM**: separa clases con fronteras
- **Árbol**: usa reglas de decisión

Puedes probar dibujando números o explorando el dataset.
""")
