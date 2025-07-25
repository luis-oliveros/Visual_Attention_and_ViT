import os
import shutil

# Ruta de la carpeta con los CSV unificados
#ruta_unificados = os.path.join(os.path.expanduser("~"), "Desktop", "CSV_Unificados")

# Crear carpetas del 1 al 20
#for i in range(1, 21):
#    ruta_carpeta = os.path.join(ruta_unificados, str(i))
#    os.makedirs(ruta_carpeta, exist_ok=True)

# Mover archivos a sus respectivas carpetas
#for archivo in os.listdir(ruta_unificados):
#    if archivo.endswith(".csv"):
#        for i in range(1, 21):
#            if archivo.endswith(f"output_{i}.csv"):
#                ruta_origen = os.path.join(ruta_unificados, archivo)
#                ruta_destino = os.path.join(ruta_unificados, str(i), archivo)
#                shutil.move(ruta_origen, ruta_destino)
#                break

#print("Archivos organizados en carpetas del 1 al 20.")

# ------------------------------------------------------------------------- #

##### Codigo para el segundo experimento ########
ruta_unificados = r"C:\Users\UsuarioCompuElite\Desktop\Tesis_doctorado\Articulo_1\nuevos_position_surface"

# Crear carpetas del 1 al 20 dentro de la ruta especificada
for i in range(1, 21):
    ruta_carpeta = os.path.join(ruta_unificados, str(i))
    os.makedirs(ruta_carpeta, exist_ok=True)

# Recorrer todas las subcarpetas y archivos en la ruta especificada
for root, dirs, files in os.walk(ruta_unificados):
    for archivo in files:
        if archivo.endswith(".csv"):
            for i in range(1, 21):
                if archivo.endswith(f"output_{i}.csv"):
                    # Definir las rutas de origen y destino para mover los archivos
                    ruta_origen = os.path.join(root, archivo)
                    ruta_destino = os.path.join(ruta_unificados, str(i), archivo)
                    
                    # Mover el archivo
                    shutil.move(ruta_origen, ruta_destino)
                    break

print("Archivos organizados en carpetas del 1 al 20.")