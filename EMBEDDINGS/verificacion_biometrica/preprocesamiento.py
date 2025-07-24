import cv2
import numpy as np

def cargar_imagen(ruta):
    """
    Carga una imagen desde una ruta y la convierte a RGB.
    """
    img_bgr = cv2.imread(ruta)
    if img_bgr is None:
        raise ValueError(f"❌ No se pudo leer la imagen en: {ruta}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb
