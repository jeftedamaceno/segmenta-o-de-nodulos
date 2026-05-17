import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# caminho da pasta com os .dcm
path = r"C:\LIDC\pacientes\pacientes_estado_avançado\LIDC-IDRI-0068\1.3.6.1.4.1.14519.5.2.1.6279.6001.272402179924508785386397167107\1.3.6.1.4.1.14519.5.2.1.6279.6001.195674076148062852502601112560"

# pegar arquivos .dcm
arquivos_dcm = [
    os.path.join(path, f)
    for f in os.listdir(path)
    if f.endswith(".dcm")
]

print(f"Total de arquivos: {len(arquivos_dcm)}")

# ler dicom
slices = []

for arquivo in arquivos_dcm:
    try:
        ds = pydicom.dcmread(arquivo)
        slices.append(ds)
    except:
        print("Erro ao ler:", arquivo)

print(f"Slices carregados: {len(slices)}")

# ======================================
# FUNÇÃO DE ORDENAÇÃO SEGURA
# ======================================

def chave_ordenacao(ds):

    # prioridade 1
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])

    # prioridade 2
    elif hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)

    # prioridade 3
    elif hasattr(ds, "InstanceNumber"):
        return int(ds.InstanceNumber)

    # fallback
    return 0

# ordenar
slices.sort(key=chave_ordenacao)

# converter imagens
imagens = np.stack([s.pixel_array for s in slices])

print("Formato:", imagens.shape)

# =========================
# VISUALIZADOR
# =========================

indice = len(imagens) // 2

fig, ax = plt.subplots(figsize=(8,8))

img_plot = ax.imshow(imagens[indice], cmap='gray')

ax.set_title(f"Slice {indice}")
ax.axis('off')

def tecla(event):
    global indice

    if event.key == 'right':
        indice = min(indice + 1, len(imagens)-1)

    elif event.key == 'left':
        indice = max(indice - 1, 0)

    img_plot.set_data(imagens[indice])
    ax.set_title(f"Slice {indice}")

    fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', tecla)

plt.show()