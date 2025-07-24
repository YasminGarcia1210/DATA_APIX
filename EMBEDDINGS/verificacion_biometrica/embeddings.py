import cv2
from insightface.app import FaceAnalysis
import numpy as np
import os

# Inicializa solo una vez
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU

def obtener_embedding(ruta_imagen, guardar_rostro=True):
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise ValueError(f"❌ No se pudo cargar la imagen: {ruta_imagen}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)

    if not faces:
        raise ValueError(f"⚠️ No se detectaron rostros en: {ruta_imagen}")

    face = faces[0]
    embedding = face.normed_embedding

    if guardar_rostro:
        try:
            cara_crop = face.crop_face(img_rgb)
            nombre_archivo = os.path.splitext(os.path.basename(ruta_imagen))[0]
            ruta_guardado = f"cara_{nombre_archivo}.jpg"
            cara_bgr = cv2.cvtColor(cara_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(ruta_guardado, cara_bgr)
            print(f"🖼️ Rostro guardado en: {ruta_guardado}")
        except Exception as e:
            print(f"⚠️ No se pudo guardar rostro: {e}")

    return embedding


