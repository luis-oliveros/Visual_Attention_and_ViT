'''
Este código separa cada imagen del video que resulta del experimento con pupil player y los lentes. 
Se cruzan los csv de los frames con los csv obtenidos en los videos (gaze_position_surfaces). 
Sólo se hace para el experimento 001 (cesteria y jarra)
'''

import os
import glob
import pandas as pd

# Define the root directory where all the subfolders are located
root_directory = r"C:\Users\UsuarioCompuElite\Desktop\Tesis_doctorado\archivos_experimentos_csv"
output_file_prefix = "_001_output_"

# Iterate over each subfolder in the root directory
for folder in glob.glob(os.path.join(root_directory, '*')):
    if os.path.isdir(folder):
        print(f"Procesando carpeta: {folder}")

        # Buscar el archivo con intervalos en la carpeta (patrón '*_001.csv')
        interval_files = glob.glob(os.path.join(folder, '*_001.csv'))
        if not interval_files:
            print(f"No se encontraron archivos de intervalos en la carpeta {folder}.")
            continue

        # Procesar cada archivo de intervalos
        for interval_file in interval_files:
            print(f"Procesando archivo de intervalos: {interval_file}")
            try:
                # Leer los intervalos
                intervals_df = pd.read_csv(interval_file)
                if 'Inicio' in intervals_df.columns and 'Final' in intervals_df.columns:
                    intervals = intervals_df[['Inicio', 'Final']].values.tolist()
                else:
                    print(f"El archivo {interval_file} no contiene las columnas 'Inicio' y 'Final'.")
                    continue
            except Exception as e:
                print(f"Error al leer el archivo de intervalos {interval_file}: {e}")
                continue

            # Buscar el archivo de datos principal correspondiente
            interval_file_base = os.path.splitext(os.path.basename(interval_file))[0]
            data_file_pattern = os.path.join(folder, f"{interval_file_base}_gaze_positions_on_surface_Surface 1.csv")
            data_file = glob.glob(data_file_pattern)

            if not data_file:
                print(f"No se encontró un archivo principal correspondiente a {interval_file}.")
                continue
            data_file = data_file[0]

            # Crear una carpeta de salida específica para este archivo
            output_directory = os.path.join(folder, f"{interval_file_base}_processed")
            os.makedirs(output_directory, exist_ok=True)

            # Procesar el archivo de datos principal
            try:
                print(f"Procesando archivo de datos principal: {data_file}")
                df = pd.read_csv(data_file)

                # Verificar la existencia de la columna 'world_index'
                frame_column = 'world_index'
                if frame_column not in df.columns:
                    print(f"La columna '{frame_column}' no se encontró en el archivo {data_file}.")
                    continue

                # Convertir la columna de tiempo a enteros
                df[frame_column] = df[frame_column].astype(int)

                # Crear archivos de salida para cada intervalo
                for i, (interval_start, interval_end) in enumerate(intervals, start=1):
                    # Filtrar los datos según el intervalo
                    interval_data = df[(df[frame_column] >= interval_start) & (df[frame_column] <= interval_end)]

                    # Nombre del archivo de salida dentro de la carpeta de salida específica
                    output_file_name = os.path.join(output_directory, f"{output_file_prefix}{i}.csv")

                    # Guardar los datos en el archivo de salida
                    interval_data.to_csv(output_file_name, index=False)
                    print(f"Archivo '{output_file_name}' creado con datos desde {interval_start} hasta {interval_end}.")
            except Exception as e:
                print(f"Error al procesar el archivo de datos {data_file}: {e}")