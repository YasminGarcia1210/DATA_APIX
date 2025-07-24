from embeddings import obtener_embedding
from verificacion import similitud_coseno, es_match

# Rutas a las imágenes
#path1 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\data\cedulas\1.jpg"
#path2 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\data\cedulas\1.1.jpg"

path1 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\data\cedulas\2.png"
path2 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\data\cedulas\1.2.jpg"

#path1 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\output\test_famosos\formato_prueba_famoso_1_page_1_face_1.jpg"
#path2 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\output\test_famosos\formato_prueba_famoso_2_page_1_face_1.jpg"

# Obtener vectores
embed1 = obtener_embedding(path1)
embed2 = obtener_embedding(path2)

# Calcular similitud y resultado
sim = similitud_coseno(embed1, embed2)
match = es_match(sim)

# Mostrar resultados
print(f"📏 Similitud coseno: {sim:.4f}")
print(f"✅ ¿Es un MATCH?: {'Sí 💚' if match else 'No ❌'}")
