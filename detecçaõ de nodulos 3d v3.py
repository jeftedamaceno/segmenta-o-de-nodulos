import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.ndimage import gaussian_filter

from skimage import morphology
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.draw import polygon

import pyvista as pv

import utils


ROOT = r"C:\exames_dos_pacientes"
XML_ROOT = r"C:\Users\jefte\Downloads\padrao ouro\tcia-lidc-xml"
PACIENTE = "LIDC-IDRI-0164"

MOSTRAR_SLICE = True
MOSTRAR_3D = True


serie_path = os.path.join(ROOT, PACIENTE)

print("\nPaciente:")
print(PACIENTE)
print("\nSérie:")
print(serie_path)

# Carrega e ordena automaticamente usando a lógica do utils.py
slices = utils.carregar_slices(serie_path)
print(f"\nSlices carregadas: {len(slices)}")


SERIES_UID = slices[0].SeriesInstanceUID
print("\nSeries UID:")
print(SERIES_UID)

xml_encontrado = None
for raiz, dirs, arquivos in os.walk(XML_ROOT):
    for arq in arquivos:
        if arq.endswith(".xml"):
            xml_path = os.path.join(raiz, arq)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                texto = ET.tostring(root, encoding="unicode")
                
                if SERIES_UID in texto:
                    xml_encontrado = xml_path
                    break
            except:
                pass
    if xml_encontrado:
        break

print("\nXML encontrado:")
print(xml_encontrado)


volume_hu = utils.converter_para_hu(slices)
print("\nVolume:")
print(volume_hu.shape)

volume_window = utils.aplicar_window(volume_hu)

volume_suave = gaussian_filter(volume_window, sigma=1)

mascara_pulmao = np.logical_and(volume_suave > -1000, volume_suave < -400)
mascara_pulmao = morphology.remove_small_objects(mascara_pulmao, min_size=1000)
mascara_pulmao = ndimage.binary_fill_holes(mascara_pulmao)

# Maiores componentes (Pulmões esquerdo e direito)
labels, num = ndimage.label(mascara_pulmao)
sizes = ndimage.sum(mascara_pulmao, labels, range(1, num + 1))
indices = np.argsort(sizes)[-2:] + 1
mascara_pulmao = np.isin(labels, indices)

# Dilatação periférica
mascara_pulmao = morphology.binary_dilation(mascara_pulmao, morphology.ball(2))

# ROI Pulmonar
volume_roi = volume_suave.copy()
volume_roi[~mascara_pulmao] = -1000


# SEGMENTAÇÃO DOS NÓDULOS (Limiar + Morfologia + Watershed)


mascara_nodulos = np.logical_and(volume_roi > -250, volume_roi < 150)
mascara_nodulos = morphology.binary_closing(mascara_nodulos, morphology.ball(1))
mascara_nodulos = ndimage.binary_fill_holes(mascara_nodulos)
mascara_nodulos = morphology.remove_small_objects(mascara_nodulos, min_size=20)

distance = ndimage.distance_transform_edt(mascara_nodulos)
coords = peak_local_max(distance, footprint=np.ones((3, 3, 3)), labels=mascara_nodulos)

mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True

markers, _ = ndimage.label(mask)
labels_ws = watershed(-distance, markers, mask=mascara_nodulos)

# FILTRAGEM GEOMÉTRICA DO WATERSHED

mascara_final = np.zeros_like(mascara_nodulos, dtype=bool)
props = measure.regionprops(labels_ws)

print("\nAnalisando componentes...\n")
for prop in props:
    volume = prop.area

    if volume < 40 or volume > 8000:
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
    if razao > 3:
        continue

    # Persistência entre fatias (slices)
    slices_ocupadas = np.unique(np.where(labels_ws == prop.label)[0])
    if len(slices_ocupadas) < 3:
        continue

    mascara_final[labels_ws == prop.label] = True

    print(
        f"Nódulo candidato:"
        f" volume={volume}"
        f" razão={razao:.2f}"
        f" slices={len(slices_ocupadas)}"
    )

# =========================================================
# LER PADRÃO OURO XML
# =========================================================

mascara_gold = np.zeros_like(mascara_final, dtype=bool)

if xml_encontrado is not None:
    tree = ET.parse(xml_encontrado)
    root = tree.getroot()
    namespaces = {'ns': root.tag.split('}')[0].strip('{')}

    z_to_slice = {}
    for i, s in enumerate(slices):
        if hasattr(s, "ImagePositionPatient"):
            z = float(s.ImagePositionPatient[2])
            z_to_slice[round(z, 1)] = i

    rois = root.findall(".//ns:roi", namespaces)
    for roi in rois:
        try:
            z = float(roi.find("ns:imageZposition", namespaces).text)
            z_round = round(z, 1)

            if z_round not in z_to_slice:
                continue

            slice_id = z_to_slice[z_round]
            edge_maps = roi.findall(".//ns:edgeMap", namespaces)

            xs = []
            ys = []
            for edge in edge_maps:
                x = int(edge.find("ns:xCoord", namespaces).text)
                y = int(edge.find("ns:yCoord", namespaces).text)
                xs.append(x)
                ys.append(y)

            if len(xs) > 3:
                rr, cc = polygon(ys, xs, shape=mascara_gold[slice_id].shape)
                mascara_gold[slice_id, rr, cc] = True
        except:
            pass


if MOSTRAR_SLICE:
    slice_id = volume_hu.shape[0] // 2
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))

    ax[0].imshow(volume_hu[slice_id], cmap="gray")
    ax[0].set_title("Original")

    ax[1].imshow(mascara_pulmao[slice_id], cmap="gray")
    ax[1].set_title("Pulmão")

    ax[2].imshow(mascara_final[slice_id], cmap="gray")
    ax[2].set_title("Algoritmo")

    ax[3].imshow(mascara_gold[slice_id], cmap="gray")
    ax[3].set_title("Padrão Ouro")

    ax[4].imshow(volume_hu[slice_id], cmap="gray")
    ax[4].imshow(mascara_gold[slice_id], cmap="Greens", alpha=0.4)
    ax[4].imshow(mascara_final[slice_id], cmap="autumn", alpha=0.4)
    ax[4].set_title("Verde=Gold | Vermelho=Algoritmo")

    plt.show()


inter = np.logical_and(mascara_final, mascara_gold).sum()
dice = (2 * inter) / (mascara_final.sum() + mascara_gold.sum() + 1e-8)
print(f"\nDice coefficient: {dice:.4f}")

mesh_pulmao = utils.criar_mesh(mascara_pulmao.astype(np.uint8))
mesh_nodulos = utils.criar_mesh(mascara_final.astype(np.uint8))
mesh_gold = utils.criar_mesh(mascara_gold.astype(np.uint8))


if MOSTRAR_3D:
    plotter = pv.Plotter()

    if mesh_pulmao is not None:
        plotter.add_mesh(mesh_pulmao, color="lightblue", opacity=0.15)

    if mesh_nodulos is not None:
        plotter.add_mesh(mesh_nodulos, color="red")

    if mesh_gold is not None:
        plotter.add_mesh(mesh_gold, color="green")

    plotter.set_background("black")
    plotter.show()