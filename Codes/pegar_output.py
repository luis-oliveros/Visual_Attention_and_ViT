import os
import shutil

# Ruta de la carpeta con los CSV organizados
ruta_unificados = os.path.join(os.path.expanduser("~"), "Desktop", "CSV_Unificados")

# Ruta de destino para las carpetas de cestería y jarra
ruta_destino = r"C:\\Users\\Luis Oliveros\\Documents\\Articulo_1\\archivos_experimentos\\Imagenes_experimentos\\Resultados_vit"

# Nombres de las carpetas de destino
carpetas_cesteria = [f"cesteria_{i:02}" for i in range(1, 11)]
carpetas_jarra = [f"jarra_{i:02}" for i in range(1, 11)]

# Crear carpetas de destino si no existen
for carpeta in carpetas_cesteria + carpetas_jarra:
    os.makedirs(os.path.join(ruta_destino, carpeta), exist_ok=True)

# Copiar archivos de las carpetas 1-10 a las carpetas de cestería
for i in range(1, 11):
    ruta_origen = os.path.join(ruta_unificados, str(i))
    ruta_destino_carpeta = os.path.join(ruta_destino, carpetas_cesteria[i - 1])
    for archivo in os.listdir(ruta_origen):
        ruta_archivo = os.path.join(ruta_origen, archivo)
        shutil.copy(ruta_archivo, ruta_destino_carpeta)

# Copiar archivos de las carpetas 11-20 a las carpetas de jarra
for i in range(11, 21):
    ruta_origen = os.path.join(ruta_unificados, str(i))
    ruta_destino_carpeta = os.path.join(ruta_destino, carpetas_jarra[i - 11])
    for archivo in os.listdir(ruta_origen):
        ruta_archivo = os.path.join(ruta_origen, archivo)
        shutil.copy(ruta_archivo, ruta_destino_carpeta)

print("Archivos copiados correctamente en las carpetas de destino.")
