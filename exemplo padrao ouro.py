import os
import xml.etree.ElementTree as ET

import numpy as np
import pydicom
import matplotlib.pyplot as plt

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# =========================================================
# CONFIGURAÇÕES
# =========================================================

PACIENTE_ID = "LIDC-IDRI-0068"

ROOT_DICOM = r"C:\exames_dos_pacientes"

ROOT_XML = r"padrao ouro\tcia-lidc-xml\189"


# =========================================================
# ENCONTRAR SÉRIE DICOM
# =========================================================

paciente_path = os.path.join(
    ROOT_DICOM,
    PACIENTE_ID
)


def encontrar_serie(root):

    melhor = None
    maior = 0

    for raiz, dirs, arquivos in os.walk(root):

        dcm = [
            f for f in arquivos
            if f.lower().endswith(".dcm")
        ]

        if len(dcm) > maior:

            maior = len(dcm)
            melhor = raiz

    return melhor


serie_path = encontrar_serie(paciente_path)

print("\nSérie encontrada:")
print(serie_path)


# =========================================================
# CARREGAR SLICES
# =========================================================

def carregar_slices(path):

    slices = []

    for arquivo in os.listdir(path):

        if arquivo.lower().endswith(".dcm"):

            caminho = os.path.join(path, arquivo)

            try:

                ds = pydicom.dcmread(caminho)

                if hasattr(ds, "pixel_array"):

                    slices.append(ds)

            except:
                pass

    return slices


print("\nCarregando slices...")

slices = carregar_slices(serie_path)

print("Slices carregados:", len(slices))


# =========================================================
# ORDENAÇÃO SEGURA
# =========================================================

def chave(ds):

    if hasattr(ds, "ImagePositionPatient"):

        return float(ds.ImagePositionPatient[2])

    elif hasattr(ds, "SliceLocation"):

        return float(ds.SliceLocation)

    elif hasattr(ds, "InstanceNumber"):

        return int(ds.InstanceNumber)

    return 0


slices.sort(key=chave)


# =========================================================
# CONVERSÃO PARA HU
# =========================================================

def converter_para_hu(slices):

    imagens = []

    shape_ref = None

    slices_validos = []

    for s in slices:

        try:

            img = s.pixel_array

            if shape_ref is None:

                shape_ref = img.shape

            if img.shape == shape_ref:

                imagens.append(img)
                slices_validos.append(s)

        except:
            pass

    volume = np.stack(imagens).astype(np.int16)

    volume[volume == -2000] = 0

    for i in range(len(slices_validos)):

        intercept = slices_validos[i].RescaleIntercept
        slope = slices_validos[i].RescaleSlope

        if slope != 1:

            volume[i] = slope * volume[i].astype(np.float64)
            volume[i] = volume[i].astype(np.int16)

        volume[i] += np.int16(intercept)

    return volume, slices_validos


print("\nConvertendo para HU...")

volume_hu, slices = converter_para_hu(slices)

print("Formato:", volume_hu.shape)


# =========================================================
# ENCONTRAR XML DO PACIENTE
# =========================================================

def encontrar_xml(root_xml, paciente_id):

    for raiz, dirs, arquivos in os.walk(root_xml):

        for arq in arquivos:

            if arq.endswith(".xml"):

                caminho = os.path.join(raiz, arq)

                try:

                    texto = open(
                        caminho,
                        encoding="utf-8"
                    ).read()

                    if paciente_id in texto:

                        return caminho

                except:
                    pass

    return None


xml_path = encontrar_xml(
    ROOT_XML,
    PACIENTE_ID
)

print("\nXML encontrado:")
print(xml_path)


# =========================================================
# LER NÓDULOS DO XML
# =========================================================

def carregar_nodulos(xml_path):

    tree = ET.parse(xml_path)

    root = tree.getroot()

    nodulos = []

    for roi in root.iter():

        if "roi" in roi.tag.lower():

            pontos = []

            z = None

            for item in roi:

                tag = item.tag.lower()

                if "imagezposition" in tag:

                    z = float(item.text)

                if "edgemap" in tag.lower():

                    x = None
                    y = None

                    for p in item:

                        if "xcoord" in p.tag.lower():

                            x = int(p.text)

                        if "ycoord" in p.tag.lower():

                            y = int(p.text)

                    if x is not None and y is not None:

                        pontos.append((x, y))

            if len(pontos) > 0 and z is not None:

                nodulos.append({
                    "z": z,
                    "pontos": pontos
                })

    return nodulos


nodulos = carregar_nodulos(xml_path)

print("\nROIs encontradas:", len(nodulos))


# =========================================================
# MAPEAR Z -> ÍNDICE DO SLICE
# =========================================================

z_slices = []

for s in slices:

    if hasattr(s, "ImagePositionPatient"):

        z_slices.append(
            float(s.ImagePositionPatient[2])
        )

    else:

        z_slices.append(0)


# =========================================================
# CRIAR MÁSCARA DOS NÓDULOS
# =========================================================

mask = np.zeros_like(volume_hu, dtype=np.uint8)

for nodulo in nodulos:

    z_xml = nodulo["z"]

    indice = np.argmin(
        np.abs(np.array(z_slices) - z_xml)
    )

    for (x, y) in nodulo["pontos"]:

        if (
            0 <= y < mask.shape[1]
            and
            0 <= x < mask.shape[2]
        ):

            mask[indice, y, x] = 1


# =========================================================
# EXPANDIR REGIÃO
# =========================================================

from scipy.ndimage import binary_dilation

mask = binary_dilation(
    mask,
    iterations=3
)


# =========================================================
# VISUALIZAR 2D
# =========================================================

indice = np.argmax(
    np.sum(mask, axis=(1, 2))
)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(
    volume_hu[indice],
    cmap="gray"
)

ax[0].set_title("Tomografia")

ax[1].imshow(
    volume_hu[indice],
    cmap="gray"
)

ax[1].imshow(
    mask[indice],
    cmap="Reds",
    alpha=0.5
)

ax[1].set_title("Padrão Ouro")

for a in ax:
    a.axis("off")

plt.show()


# =========================================================
# VISUALIZAÇÃO 3D
# =========================================================

print("\nGerando reconstrução 3D...")


verts, faces, normals, values = measure.marching_cubes(
    mask,
    level=0.5
)

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(
    111,
    projection='3d'
)

mesh = Poly3DCollection(
    verts[faces],
    alpha=0.7
)

ax.add_collection3d(mesh)

ax.set_xlim(0, mask.shape[2])
ax.set_ylim(0, mask.shape[1])
ax.set_zlim(0, mask.shape[0])

ax.set_title("Nódulos - Padrão Ouro")

plt.show()