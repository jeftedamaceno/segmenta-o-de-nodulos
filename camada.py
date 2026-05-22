import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt

from skimage import morphology
from scipy import ndimage



# CONFIGURAÇÕES


ROOT = r"C:\exames_dos_pacientes"

PACIENTE = "LIDC-IDRI-0068"



pasta_paciente = os.path.join(
    ROOT,
    PACIENTE
)

serie_path = None

for raiz, dirs, arquivos in os.walk(pasta_paciente):

    dcm = [
        f for f in arquivos
        if f.endswith(".dcm")
    ]

    if len(dcm) > 20:

        serie_path = raiz
        break


print("\nSérie encontrada:")
print(serie_path)

# CARREGAR DICOM
slices = []

for arquivo in os.listdir(serie_path):

    if arquivo.endswith(".dcm"):

        caminho = os.path.join(
            serie_path,
            arquivo
        )

        try:

            ds = pydicom.dcmread(caminho)

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

# CONVERTER PARA HU
def converter_para_hu(slices):

    imagens = []
    slices_validas = []

    shape_referencia = None

    for s in slices:

        try:
            img = s.pixel_array

            if shape_referencia is None:
                shape_referencia = img.shape

            if img.shape != shape_referencia:
                print(f"Slice ignorada: {img.shape}")
                continue

            imagens.append(img)
            slices_validas.append(s)

        except Exception as e:
            print("Erro slice:", e)

    volume = np.stack(imagens).astype(np.int16)

    for i, s in enumerate(slices_validas):

        intercept = s.RescaleIntercept
        slope = s.RescaleSlope

        if slope != 1:
            volume[i] = slope * volume[i].astype(np.float64)
            volume[i] = volume[i].astype(np.int16)

        volume[i] += np.int16(intercept)

    return volume


volume_hu = converter_para_hu(slices)

print("\nVolume HU:", volume_hu.shape)

# SEGMENTAÇÃO DO PULMÃO
"""
Pulmão geralmente:
HU entre -1000 e -400
"""

mascara_pulmao = np.logical_and(
    volume_hu > -1000,
    volume_hu < -400
)
# REMOVER RUÍDOS
mascara_pulmao = morphology.remove_small_objects(
    mascara_pulmao,
    min_size=500
)

mascara_pulmao = ndimage.binary_fill_holes(
    mascara_pulmao
)
# POSSÍVEIS NÓDULOS
"""
Nódulos normalmente:
HU entre -300 e 100
"""

mascara_nodulos = np.logical_and(
    volume_hu > -300,
    volume_hu < 100
)

# manter apenas dentro do pulmão
mascara_nodulos = np.logical_and(
    mascara_nodulos,
    mascara_pulmao
)
# VISUALIZAÇÃO

indice = len(volume_hu) // 2

fig, ax = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

# imagem original
ax[0].imshow(
    volume_hu[indice],
    cmap="gray",
    vmin=-1000,
    vmax=400
)

ax[0].set_title("Tomografia")
ax[0].axis("off")


# máscara pulmão
ax[1].imshow(
    mascara_pulmao[indice],
    cmap="gray"
)

ax[1].set_title("Pulmão Segmentado")
ax[1].axis("off")


# possíveis nódulos
ax[2].imshow(
    mascara_nodulos[indice],
    cmap="gray"
)

ax[2].set_title("Possíveis Nódulos")
ax[2].axis("off")


plt.tight_layout()
plt.show()