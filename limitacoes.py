import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from scipy import ndimage
from scipy.ndimage import gaussian_filter

from skimage import morphology
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import pyvista as pv

# Importa as utilidades centrais compartilhadas
import utils

# =========================================================
# CONFIGURAÇÕES
# =========================================================

ROOT = r"C:\exames_dos_pacientes"
PACIENTE = "LIDC-IDRI-0068"
XML_ROOT = r"C:\padrao ouro\tcia-lidc-xml"

MIN_VOLUME = 30
MAX_VOLUME = 3000
MIN_ESFERICIDADE = 0.5
MAX_RAZAO = 2.0
MAX_SLICES = 20
MAX_DIAMETRO_MM = 40

# =========================================================
# ENCONTRAR XML DO PACIENTE
# =========================================================

def encontrar_xml(xml_root, paciente):
    numero = paciente.split("-")[-1]
    numero = str(int(numero))

    for raiz, dirs, arquivos in os.walk(xml_root):
        for arq in arquivos:
            if arq.endswith(".xml"):
                nome = arq.lower()
                if numero in nome:
                    return os.path.join(raiz, arq)
    return None

xml_path = encontrar_xml(XML_ROOT, PACIENTE)
print("\nXML encontrado:")
print(xml_path)

# =========================================================
# ENCONTRAR E CARREGAR DICOM VIA UTILS
# =========================================================

pasta_paciente = os.path.join(ROOT, PACIENTE)
serie_path = utils.encontrar_serie_dicom(pasta_paciente)

print("\nSérie encontrada:")
print(serie_path)

# Carrega e ordena de maneira inteligente baseando-se nas coordenadas z reais
slices = utils.carregar_slices(serie_path)

# =========================================================
# EXTRAÇÃO DE ESPAÇAMENTO ESPACIAL (METADADOS)
# =========================================================

pixel_spacing = slices[0].PixelSpacing
slice_thickness = float(slices[0].SliceThickness)

spacing_y = float(pixel_spacing[0])
spacing_x = float(pixel_spacing[1])
spacing_z = float(slice_thickness)

print("\nSpacing:")
print(spacing_x, spacing_y, spacing_z)

# =========================================================
# PROCESSAMENTO DE INTENSIDADE VIA UTILS (HU + WINDOWING)
# =========================================================

volume_hu = utils.converter_para_hu(slices)
print("\nVolume:")
print(volume_hu.shape)

volume_window = utils.aplicar_window(volume_hu)

# =========================================================
# SUAVIZAÇÃO E SEGMENTAÇÃO DO PULMÃO
# =========================================================

volume_suave = gaussian_filter(volume_window, sigma=1)

mascara_pulmao = np.logical_and(volume_suave > -1000, volume_suave < -400)
mascara_pulmao = morphology.remove_small_objects(mascara_pulmao, min_size=1000)
mascara_pulmao = ndimage.binary_fill_holes(mascara_pulmao)

# Isolar os 2 maiores componentes (Pulmão esquerdo e direito)
labels, num = ndimage.label(mascara_pulmao)
sizes = ndimage.sum(mascara_pulmao, labels, range(1, num + 1))
indices = np.argsort(sizes)[-2:] + 1
mascara_pulmao = np.isin(labels, indices)

# Isolar Região de Interesse (ROI) interna
volume_roi = volume_suave.copy()
volume_roi[~mascara_pulmao] = -1000

# =========================================================
# SEGMENTAÇÃO E TRATAMENTO MORFOLÓGICO DOS CANDIDATOS
# =========================================================

mascara_nodulos = np.logical_and(volume_roi > -250, volume_roi < 200)
mascara_nodulos = morphology.binary_closing(mascara_nodulos, morphology.ball(1))
mascara_nodulos = ndimage.binary_fill_holes(mascara_nodulos)
mascara_nodulos = morphology.remove_small_objects(mascara_nodulos, min_size=20)

# =========================================================
# SEPARAÇÃO POR WATERSHED 3D
# =========================================================

distance = ndimage.distance_transform_edt(mascara_nodulos)
coords = peak_local_max(distance, footprint=np.ones((3, 3, 3)), labels=mascara_nodulos)

mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True

markers, _ = ndimage.label(mask)
labels_ws = watershed(-distance, markers, mask=mascara_nodulos)

# =========================================================
# FILTRAGEM GEOMÉTRICA FOCADA (MÉTRICAS BIOMÉTRICAS)
# =========================================================

mascara_final = np.zeros_like(mascara_nodulos, dtype=bool)
props = measure.regionprops(labels_ws)

print("\nComponentes encontrados:", len(props))

for prop in props:
    volume = prop.area

    if volume < MIN_VOLUME or volume > MAX_VOLUME:
        continue

    bbox = prop.bbox
    dz = bbox[3] - bbox[0]
    dy = bbox[4] - bbox[1]
    dx = bbox[5] - bbox[2]

    maior = max(dx, dy, dz)
    menor = min(dx, dy, dz)

    if menor == 0:
        continue

    razao = maior / menor

    # Remove estruturas alongadas (vasos sanguíneos verticais/horizontais)
    if razao > MAX_RAZAO:
        continue

    # Esfericidade adaptada
    esfericidade = menor / maior
    if esfericidade < MIN_ESFERICIDADE:
        continue

    # Persistência máxima de fatias
    if dz > MAX_SLICES:
        continue

    # Diâmetro tridimensional real convertido para milímetros (mm)
    diametro_x = dx * spacing_x
    diametro_y = dy * spacing_y
    diametro_z = dz * spacing_z

    diametro_max = max(diametro_x, diametro_y, diametro_z)
    if diametro_max > MAX_DIAMETRO_MM:
        continue

    mascara_final[labels_ws == prop.label] = True

# Pós-processamento de limpeza de ruídos remanescentes
mascara_final = morphology.remove_small_objects(mascara_final, min_size=MIN_VOLUME)

# =========================================================
# VISUALIZAÇÃO SUCINTA EM Slices 2D
# =========================================================

slice_id = volume_hu.shape[0] // 2

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(volume_hu[slice_id], cmap="gray")
ax[0].set_title("Original")

ax[1].imshow(mascara_pulmao[slice_id], cmap="gray")
ax[1].set_title("Pulmão")

ax[2].imshow(mascara_final[slice_id], cmap="gray")
ax[2].set_title("Nódulos")

ax[3].imshow(volume_hu[slice_id], cmap="gray")
ax[3].imshow(mascara_final[slice_id], cmap="autumn", alpha=0.5)
ax[3].set_title("Overlay")
plt.show()

# =========================================================
# CONVERSÃO DE MALHA REUTILIZÁVEL VIA UTILS & VISUALIZAÇÃO 3D
# =========================================================

mesh_pulmao = utils.criar_mesh(mascara_pulmao.astype(np.uint8))
mesh_nodulos = utils.criar_mesh(mascara_final.astype(np.uint8))

plotter = pv.Plotter()

if mesh_pulmao is not None:
    plotter.add_mesh(mesh_pulmao, color="lightblue", opacity=0.05)

if mesh_nodulos is not None:
    plotter.add_mesh(mesh_nodulos, color="red")
else:
    print("\nNenhum nódulo encontrado para exibição 3D.\n")

plotter.set_background("black")
plotter.show()