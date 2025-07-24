from embeddings import obtener_embedding

#path1 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\data\cedulas\1.jpg"

#path2 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\data\cedulas\1.1.jpg"


path1 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\data\cedulas\2.png"
path2 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\data\cedulas\1.2.jpg"

#path1 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\output\test_famosos\formato_prueba_famoso_1_page_1_face_1.jpg"
#path2 = r"G:\Mi unidad\APIUX-TECH\EMBEDDINGS\verificacion_biometrica\output\test_famosos\formato_prueba_famoso_3_page_1_face_1.jpg"

embed1 = obtener_embedding(path1)
embed2 = obtener_embedding(path2)

print("🔍 Vector 1:", embed1[:5])
print("🔍 Vector 2:", embed2[:5])
