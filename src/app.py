
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# --- CONFIGURAÇÕES ---
st.set_page_config(page_title="KMNIST Neural Net Demo", layout="wide")

# Mapeamento de Classes KMNIST
LABELS_MAP = {
    0: 'o (お)', 1: 'ki (き)', 2: 'su (す)', 3: 'tsu (つ)', 4: 'na (な)',
    5: 'ha (は)', 6: 'ma (ま)', 7: 'ya (や)', 8: 're (れ)', 9: 'wo (を)'
}

# --- FUNÇÕES ---
@st.cache_resource
def load_model():
    # Pega o diretório onde o app.py está (ou seja, a pasta src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'CNN_best.keras') 
    
    return tf.keras.models.load_model(model_path)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# --- INTERFACE ---
st.title("🖌️ Reconhecimento de KMNIST (Kana) com CNN")
st.write("Upload de imagem de um caractere Hiragana antigo para classificação e análise via Grad-CAM.")

model = load_model()

# Achar camada conv automaticamente
last_conv_layer_name = ""
for layer in reversed(model.layers):
    if 'conv' in layer.name:
        last_conv_layer_name = layer.name
        break

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Escolha uma imagem (png/jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Processar Imagem
    image = Image.open(uploaded_file).convert('L') # Converter para Escala de Cinza
    image_resized = image.resize((28, 28))
    img_array = np.array(image_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1) # (28, 28, 1)
    img_batch = np.expand_dims(img_array, axis=0)  # (1, 28, 28, 1)

    # Predição
    preds = model.predict(img_batch)
    pred_label = np.argmax(preds)
    confidence = np.max(preds)

    with col1:
        st.image(image, caption="Imagem Original", width=200)
        st.success(f"**Predição:** {LABELS_MAP[pred_label]}")
        st.info(f"**Confiança:** {confidence:.2%}")

    with col2:
        st.subheader("Análise Explicável (Grad-CAM)")
        
        # Gerar Grad-CAM
        heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name)
        
        # Visualização
        heatmap_resized = cv2.resize(heatmap, (280, 280)) # Aumentar para visualizar melhor
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Mostrar apenas o heatmap ou overlay (opcional)
        st.image(heatmap_colored, caption="Mapa de Ativação (Onde a IA olhou)", width=200, channels="BGR")
        st.write("As áreas vermelhas indicam os traços que mais influenciaram a decisão da IA.")

else:
    st.warning("Por favor, faça upload de uma imagem para começar.")

# Rodapé
st.markdown("---")
st.caption("Projeto de Avaliação 1 - Redes Neurais NES")
