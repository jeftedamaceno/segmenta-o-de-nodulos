import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt

from matplotlib.widgets import Button

# ENCONTRAR SÉRIES

ROOT = r"C:\LIDC"

series_encontradas = []

for raiz, dirs, arquivos in os.walk(ROOT):

    arquivos_dcm = [
        f for f in arquivos
        if f.endswith(".dcm")
    ]

    if len(arquivos_dcm) > 20:

        series_encontradas.append({
            "path": raiz,
            "quantidade": len(arquivos_dcm)
        })

print(f"\nTotal de séries: {len(series_encontradas)}")

# ESCOLHER PACIENTE

indice_paciente = 0

path = series_encontradas[indice_paciente]["path"]

print("\nPaciente selecionado:")
print(path)

# CARREGAR DICOM

slices = []

arquivos = [
    os.path.join(path, f)
    for f in os.listdir(path)
    if f.endswith(".dcm")
]

for arquivo in arquivos:

    try:

        ds = pydicom.dcmread(arquivo)

        if hasattr(ds, "pixel_array"):
            slices.append(ds)

    except:
        pass


print(f"\nSlices carregados: {len(slices)}")

# ORDENAÇÃO SEGURA

def chave(ds):

    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])

    elif hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)

    elif hasattr(ds, "InstanceNumber"):
        return int(ds.InstanceNumber)

    return 0

slices.sort(key=chave)

# CONVERSÃO PARA HU

def converter_para_hu(slices):

    volume = np.stack([
        s.pixel_array for s in slices
    ]).astype(np.int16)

    volume[volume == -2000] = 0

    for i in range(len(slices)):

        intercept = slices[i].RescaleIntercept
        slope = slices[i].RescaleSlope

        if slope != 1:

            volume[i] = slope * volume[i].astype(np.float64)
            volume[i] = volume[i].astype(np.int16)

        volume[i] += np.int16(intercept)

    return volume


volume_hu = converter_para_hu(slices)

print("\nVolume:", volume_hu.shape)

# INTERVALOS DE HU
"""
Faixas aproximadas usadas na radiologia
"""

HU_MODOS = {

    "Pulmão": {
        "min": -1000,
        "max": -400
    },

    "Tecido Mole": {
        "min": -100,
        "max": 100
    },

    "Osso": {
        "min": 300,
        "max": 2000
    },

    "Gordura": {
        "min": -200,
        "max": -50
    },

    "Nódulos Suspeitos": {
        "min": -300,
        "max": 100
    }
}


modos = list(HU_MODOS.keys())

modo_atual = 0



# FUNÇÃO WINDOWING


def aplicar_janela(slice_img, hu_min, hu_max):

    img = np.clip(
        slice_img,
        hu_min,
        hu_max
    )

    img = img.astype(np.float32)

    img -= img.min()

    if img.max() != 0:
        img /= img.max()

    return img



# VISUALIZADOR


indice_slice = len(volume_hu) // 2

fig, ax = plt.subplots(figsize=(9, 9))

plt.subplots_adjust(bottom=0.15)

nome_modo = modos[modo_atual]

hu_min = HU_MODOS[nome_modo]["min"]
hu_max = HU_MODOS[nome_modo]["max"]

img = aplicar_janela(
    volume_hu[indice_slice],
    hu_min,
    hu_max
)

img_plot = ax.imshow(
    img,
    cmap="gray"
)

titulo = ax.set_title(
    f"{nome_modo} | Slice {indice_slice}\nHU [{hu_min}, {hu_max}]",
    fontsize=12
)

ax.axis("off")



# ATUALIZAR IMAGEM


def atualizar():

    global nome_modo
    global hu_min
    global hu_max

    nome_modo = modos[modo_atual]

    hu_min = HU_MODOS[nome_modo]["min"]
    hu_max = HU_MODOS[nome_modo]["max"]

    nova_img = aplicar_janela(
        volume_hu[indice_slice],
        hu_min,
        hu_max
    )

    img_plot.set_data(nova_img)

    titulo.set_text(
        f"{nome_modo} | Slice {indice_slice}\nHU [{hu_min}, {hu_max}]"
    )

    fig.canvas.draw_idle()



# CONTROLES DO TECLADO


def tecla(event):

    global indice_slice
    global modo_atual

    # navegar slices
    if event.key == "right":

        indice_slice = min(
            indice_slice + 1,
            len(volume_hu) - 1
        )

    elif event.key == "left":

        indice_slice = max(
            indice_slice - 1,
            0
        )

    # trocar modo HU
    elif event.key == "up":

        modo_atual = (
            modo_atual + 1
        ) % len(modos)

    elif event.key == "down":

        modo_atual = (
            modo_atual - 1
        ) % len(modos)

    atualizar()


fig.canvas.mpl_connect(
    "key_press_event",
    tecla
)



# INSTRUÇÕES


print("""
CONTROLES:

→  próximo slice
←  slice anterior

↑  próximo modo HU
↓  modo HU anterior
""")
# MOSTRAR

plt.show()