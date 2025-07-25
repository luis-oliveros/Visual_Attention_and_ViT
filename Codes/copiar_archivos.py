import os
import shutil

# Ruta principal donde están las carpetas
ruta_principal = "C:\\Users\\Luis Oliveros\\Documents\\Articulo_1\\archivos_experimentos"  # Cambia esto por la ruta de las carpetas (donde están las carpetas como '01022024')

# Ruta de destino donde se copiarán los CSV
ruta_destino = os.path.join(os.path.expanduser("~"), "Desktop", "CSV_Unificados")
os.makedirs(ruta_destino, exist_ok=True)

# Recorrer las carpetas y subcarpetas
for carpeta_raiz, subcarpetas, archivos in os.walk(ruta_principal):
    for archivo in archivos:
        if archivo.endswith(".csv"):
            # Ruta completa del archivo CSV
            ruta_archivo = os.path.join(carpeta_raiz, archivo)
            # Copiar el archivo al destino
            shutil.copy(ruta_archivo, ruta_destino)

print(f"Todos los archivos CSV se han copiado a {ruta_destino}")