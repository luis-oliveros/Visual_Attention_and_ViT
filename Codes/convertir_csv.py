import os
import pandas as pd

# Ruta de la carpeta principal que contiene las subcarpetas y archivos Excel
base_folder = "C:\\Users\\Luis Oliveros\\Documents\\Articulo_1\\archivos_experimentos"

# Recorrer todas las carpetas y subcarpetas
for root, dirs, files in os.walk(base_folder):
    for file in files:
        if file.endswith(".xlsx") or file.endswith(".xls"):  # Filtrar solo archivos Excel
            excel_file = os.path.join(root, file)  # Ruta completa del archivo Excel
            
            # Cargar todas las hojas del archivo de Excel
            try:
                excel_data = pd.ExcelFile(excel_file)
                
                # Iterar por cada hoja y guardar como CSV
                for sheet_name in excel_data.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    # Crear el nombre del archivo CSV
                    csv_file = os.path.join(root, f"{sheet_name}.csv")
                    
                    # Guardar la hoja como archivo CSV
                    df.to_csv(csv_file, index=False, encoding="utf-8")
                    print(f"Archivo guardado: {csv_file}")
            
            except Exception as e:
                print(f"Error procesando {excel_file}: {e}")
