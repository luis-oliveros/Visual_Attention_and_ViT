import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.ndimage import gaussian_gradient_magnitude

#imagenes = [f"cesteria_{str(i).zfill(2)}" for i in range(1, 11)] + [f"jarra_{str(i).zfill(2)}" for i in range(1, 11)]
imagenes = [f"jarra_{str(i).zfill(2)}" for i in range(1, 11)]


metricas_df = pd.DataFrame(columns=["Imagen", "KLD", "JSD", "Hellinger", "KS", "Sobolev"])
epsilon = 1e-10

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['font.size'] = 6

fig, ax = plt.subplots(len(imagenes), 5, figsize=(25, 5), dpi=300)
fig.set_figwidth(20)
fig.set_figheight(40)

for fila, img in enumerate(imagenes):
    print(f"Procesando imagen: {fila}")
    try:
        # Traducir nombre de imagen a inglés
        img_english = img.replace("cesteria", "basketry").replace("jarra", "jar")
        search_pattern = f"C:/Users/UsuarioCompuElite/Desktop/Tesis_doctorado/Articulo_1/metodologia/Resultados_GMM22/{img}/normalized_csv/*_normalized.csv"
        csv_files = glob.glob(search_pattern)
        if not csv_files:
            print(f"No se encontró archivo para {img}")
            continue

        file_participants = csv_files[0]
        file_vit = f"C:/Users/UsuarioCompuElite/Desktop/Tesis_doctorado/Articulo_1/metodologia/Resultados_vit_experimento_001_/{img_english}/normalized_csv/{img}_attention_mean_normalized.csv"

        df_participants = pd.read_csv(file_participants, header=None)
        df_vit = pd.read_csv(file_vit, header=None)

        heatmap_participants = df_participants.values
        heatmap_vit = df_vit.values

        min_rows = min(heatmap_participants.shape[0], heatmap_vit.shape[0])
        min_cols = min(heatmap_participants.shape[1], heatmap_vit.shape[1])
        heatmap_participants = heatmap_participants[:min_rows, :min_cols]
        heatmap_vit = heatmap_vit[:min_rows, :min_cols]

        hp_safe = heatmap_participants + epsilon
        hv_safe = heatmap_vit + epsilon

        kl_div = np.sum(hp_safe * np.log(hp_safe / hv_safe))
        m = 0.5 * (hp_safe + hv_safe)
        js_div = 0.5 * np.sum(hp_safe * np.log(hp_safe / m)) + 0.5 * np.sum(hv_safe * np.log(hv_safe / m))
        hellinger_dist = np.sqrt(0.5 * np.sum((np.sqrt(hp_safe) - np.sqrt(hv_safe)) ** 2))
        grad_participants = gaussian_gradient_magnitude(heatmap_participants, sigma=1)
        grad_vit = gaussian_gradient_magnitude(heatmap_vit, sigma=1)
        sobolev_map = np.abs(grad_participants - grad_vit)
        sobolev_distance = np.sum(sobolev_map)
        ks_stat, _ = ks_2samp(hp_safe.flatten(), hv_safe.flatten())
        ks_map = np.abs(hp_safe - hv_safe)

        metricas_df = pd.concat([metricas_df, pd.DataFrame([{
            "Imagen": img_english,
            "KLD": kl_div,
            "JSD": js_div,
            "Hellinger": hellinger_dist,
            "KS": ks_stat,
            "Sobolev": sobolev_distance
        }])], ignore_index=True)

        # Visualización
        
        kl_map = hp_safe * np.log(hp_safe / hv_safe)
        js_map = 0.5 * (hp_safe * np.log(hp_safe / m)) + 0.5 * (hv_safe * np.log(hv_safe / m))
        hellinger_map = np.sqrt(0.5 * ((np.sqrt(hp_safe) - np.sqrt(hv_safe)) ** 2))

        # Traducir nombre de imagen a inglés
        img_english = img.replace("cesteria", "basketry").replace("jarra", "jar")

        # Títulos con nombres traducidos
        titles = [
        f"{img_english} - KL Divergence",
        f"{img_english} - Jensen-Shannon Divergence",
        f"{img_english} - Hellinger Distance",
        "Distancia de Sobolev (Gradientes)",
        "Kolmogorov-Smirnov (Pixel a Pixel)"
        ]
        maps = [kl_map, js_map, hellinger_map, sobolev_map, ks_map]

        for columna, (data, title) in enumerate(zip(maps, titles)):

            im = ax[fila, columna].imshow(np.nan_to_num(data), cmap='coolwarm', interpolation='nearest', aspect='auto')
            ax[fila, columna].set_title(title)
            ax[fila, columna].axis("off")
            fig.colorbar(im, ax=ax[fila, columna])

        

    except Exception as e:
        print(f"❌ Error procesando {img}: {e}")

plt.savefig('resultados_metricas_jar.png', dpi=300, bbox_inches='tight')
plt.tight_layout()


