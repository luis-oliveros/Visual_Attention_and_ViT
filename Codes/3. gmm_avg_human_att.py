import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# funcion que crea distribucion gaussiana atencion, guarda csv regular y ajustada a vit
def create_heatmap(surface_df,csv_file, cover_img, i, dir_obj):

    cover_img = plt.imread(cover_img)
    gaze_on_surf = pd.read_csv(surface_df)

    #gaze_on_surf = surface_df
    #gaze_on_surf = surface_df[surface_df.on_surf == True]
    #gaze_on_surf = surface_df[(surface_df.confidence > 0.8)]

    grid = cover_img.shape[0:2] # height, width of the loaded image
    heatmap_detail = 0.01 # this will determine the gaussian blur kerner of the image (higher number = more blur)

    #print(grid)

    gaze_on_surf_x = gaze_on_surf['x_norm']
    gaze_on_surf_y = gaze_on_surf['y_norm']

    # flip the fixation points
    # from the original coordinate system,
    # where the origin is at botton left,
    # to the image coordinate system,
    # where the origin is at top left
    gaze_on_surf_y = 1 - gaze_on_surf_y

    # make the histogram
    hist, x_edges, y_edges = np.histogram2d(
        gaze_on_surf_y,
        gaze_on_surf_x,
        range=[[0, 1.0], [0, 1.0]],
        #normed=False,
        bins=grid
    )

    # gaussian blur kernel as a function of grid/surface size
    filter_h = int(heatmap_detail * grid[0]) // 2 * 2 + 1
    filter_w = int(heatmap_detail * grid[1]) // 2 * 2 + 1
    heatmap = gaussian_filter(hist, sigma=(filter_w, filter_h), order=0)

    #print(heatmap)
    #print(type(heatmap))
    # Specify the file path and name
    #file_path = 'heatmap.csv'
    # Save the array to a CSV file
    #np.savetxt(file_path, heatmap, delimiter=',')


    # Step 1: Get height and width
    height, width = heatmap.shape

    # Step 2: Calculate remainder
    height_remainder = height % 16
    width_remainder = width % 16

    # Step 3: Eliminate last rows and columns
    new_height = height - height_remainder
    new_width = width - width_remainder

    # Update the array by keeping only the relevant portion
    your_array = heatmap[:new_height, :new_width]


    # Save the array to a CSV file

    np.savetxt(
    fr"C:\Users\UsuarioCompuElite\Desktop\Tesis_doctorado\Articulo_1\metodologia\Resultados_vit_experimento_001\{dir_obj}\hum\reminder_heatmap_{csv_file[:-4]}_{dir_obj}.csv",
    your_array,
    delimiter=',')



def avg_heatmap(dir_obj):
# Initialize an empty DataFrame to store the cumulative sum of values
    cumulative_sum = None
    #print("avg")
    # Specify the path to the folder containing CSV files
    csv_folder = rf"C:\Users\UsuarioCompuElite\Desktop\Tesis_doctorado\Articulo_1\metodologia\Resultados_vit_experimento_001\{dir_obj}\hum"
    csv_files = sorted([file for file in os.listdir(csv_folder) if file.endswith(".csv") and file not in ["cum_reminder.csv", "cum_median.csv"]])
    print(f"Archivos seleccionados para el cálculo de la media: {csv_files}")

    for csv_file in csv_files:
      if csv_file != "cum_reminder.csv":
          csv_path = os.path.join(csv_folder, csv_file)
          if cumulative_sum is None:
            cumulative_sum =  pd.read_csv(csv_path, header=None)
          else:
            cumulative_sum += (pd.read_csv(csv_path, header=None))

    cumulative_sum = cumulative_sum / len(csv_files)

    # Optionally, save the result to a new CSV file
    #average_values.to_csv('average_values_reminder.csv', index=False)
    cumulative_sum.to_csv(rf"C:\Users\UsuarioCompuElite\Desktop\Tesis_doctorado\Articulo_1\metodologia\Resultados_vit_experimento_001\{dir_obj}\hum\cum_reminder.csv", index=False, header=None)

    #cover_img = plt.imread(jpg_file)
    #plt.figure(figsize=(8,8))
    #plt.imshow(cover_img)
    #plt.imshow(cumulative_sum, cmap='jet', alpha=0.5)
    #plt.axis('off')
    #plt.savefig("cum_heatmap.png")

    return()

def median_heatmap(dir_obj):
    # Especifica la carpeta con los archivos CSV
    csv_folder = rf"C:\Users\UsuarioCompuElite\Desktop\Tesis_doctorado\Articulo_1\metodologia\Resultados_vit_experimento_001\{dir_obj}\hum"
    csv_files = sorted([file for file in os.listdir(csv_folder) if file.endswith(".csv") and file not in ["cum_reminder.csv", "cum_median.csv"]])
    print(f"Archivos seleccionados para el cálculo de la mediana: {csv_files}")

    # Lista para almacenar los valores de todos los mapas
    all_heatmaps = []

    # Lee cada archivo CSV y almacena sus valores en una lista
    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder, csv_file)
        heatmap = pd.read_csv(csv_path, header=None).values  # Convertir a numpy array
        all_heatmaps.append(heatmap)

    # Convierte la lista en un arreglo numpy de 3 dimensiones (n_archivos, filas, columnas)
    stacked_heatmaps = np.stack(all_heatmaps)

    # Calcula la mediana a lo largo de la primera dimensión (n_archivos)
    median_heatmap = np.median(stacked_heatmaps, axis=0)

    # Guarda la mediana como un nuevo archivo CSV
    output_path = rf"{csv_folder}\cum_median.csv"
    np.savetxt(output_path, median_heatmap, delimiter=',')
    print(f"Archivo de mediana guardado en: {output_path}")

def gmm(list):
  for i in list:
    csv_folder = rf"C:\Users\UsuarioCompuElite\Desktop\Tesis_doctorado\Articulo_1\metodologia\Resultados_vit_experimento_001\{i}\pupil_data"


    # List files in both CSV and JPG folders
    csv_files = sorted([file for file in os.listdir(csv_folder) if file.endswith(".csv")])
    #print(f"Archivos seleccionados para el cálculo del gmm: {csv_files}")
    csv_files = sorted([file for file in os.listdir(csv_folder) if file.endswith(".csv") and not file.endswith(("cum_reminder.csv", "cum_median.csv"))])

    jpg_file = rf"C:\Users\UsuarioCompuElite\Desktop\Tesis_doctorado\Articulo_1\metodologia\Data_visual_transformer\imagenes_experimento_001\{i}.jpg"

# Process each pair of CSV and JPG files

    for csv_file in csv_files:
      csv_path = os.path.join(csv_folder, csv_file)

      #creacion csv de dits guassiana
      create_heatmap(csv_path,csv_file, jpg_file, jpg_file[:-4], i)

    #print(csv_file)
    #print(jpg_file)

    avg_heatmap(i)
    median_heatmap(i)
    
list = [
    "basketry_01", "basketry_02", "basketry_03", "basketry_04", "basketry_05",
    "basketry_06", "basketry_07", "basketry_08", "basketry_09", "basketry_10",
    "jar_01", "jar_02", "jar_03", "jar_04", "jar_05",
    "jar_06", "jar_07", "jar_08", "jar_09", "jar_10"] 


gmm(list)