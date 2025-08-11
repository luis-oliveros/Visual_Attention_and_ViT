import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_heatmaps(dir_obj):
    # Ruta base de la carpeta
    base_path = r"C:\Users\UsuarioCompuElite\Desktop\Tesis_doctorado\Articulo_1\metodologia\Resultados_vit_experimento_001"
    csv_folder = os.path.join(base_path, dir_obj, "hum")

    # Rutas de los archivos
    avg_path = os.path.join(csv_folder, "cum_reminder.csv")
    median_path = os.path.join(csv_folder, "cum_median.csv")

    # Leer los archivos CSV
    try:
        cum_avg = pd.read_csv(avg_path, header=None).values  # Convertir a numpy array
        cum_median = pd.read_csv(median_path, header=None).values
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Crear gráficos
    plt.figure(figsize=(12, 6))

    # Gráfico del promedio acumulativo
    plt.subplot(1, 2, 1)
    plt.title("Promedio Acumulativo (cum_avg_reminder)")
    plt.imshow(cum_avg, cmap='jet', interpolation='nearest')
    plt.colorbar(label="Intensidad")
    plt.axis('off')

    # Gráfico de la mediana acumulativa
    plt.subplot(1, 2, 2)
    plt.title("Mediana Acumulativa (cum_median)")
    plt.imshow(cum_median, cmap='jet', interpolation='nearest')
    plt.colorbar(label="Intensidad")
    plt.axis('off')

    # Crear directorio para guardar imágenes si no existe
    output_folder = os.path.join(base_path, "avg_median_heatmaps")
    os.makedirs(output_folder, exist_ok=True)

    # Guardar las imágenes
    avg_output_path = os.path.join(output_folder, f"{dir_obj}_avg_heatmap.png")
    median_output_path = os.path.join(output_folder, f"{dir_obj}_median_heatmap.png")
    plt.savefig(avg_output_path, bbox_inches='tight', pad_inches=0)
    plt.savefig(median_output_path, bbox_inches='tight', pad_inches=0)

    print(f"Imágenes guardadas en:\n{avg_output_path}\n{median_output_path}")

    # Cerrar la figura para evitar problemas de memoria
    plt.close()

# Lista de directorios
list = [
    "basketry_01", "basketry_02", "basketry_03", "basketry_04", "basketry_05",
    "basketry_06", "basketry_07", "basketry_08", "basketry_09", "basketry_10",
    "jar_01", "jar_02", "jar_03", "jar_04", "jar_05",
    "jar_06", "jar_07", "jar_08", "jar_09", "jar_10"] 

# Procesar cada directorio
for dir_obj in list:
    print(f"Procesando: {dir_obj}")
    plot_heatmaps(dir_obj)
