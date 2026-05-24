import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.ndimage import gaussian_filter

from skimage import morphology
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

import pyvista as pv

# exemplos interesantes 0207, 0164, 0137
# estranho 159, 0118
# 0100, 0091
# geometricamente parecidos mais fora do pumao 0088
ROOT = r"C:\exames_dos_pacientes"
PACIENTE = "LIDC-IDRI-0068"


# =========================================================
# ENCONTRAR SÉRIE
# =========================================================

pasta_paciente = os.path.join(ROOT, PACIENTE)

serie_path = None

for raiz, dirs, arquivos in os.walk(pasta_paciente):

    dcm = [f for f in arquivos if f.endswith(".dcm")]

    if len(dcm) > 20:
        serie_path = raiz
        break

print("Série encontrada:")
print(serie_path)


# =========================================================
# CARREGAR SLICES
# =========================================================

slices = []

for arquivo in os.listdir(serie_path):

    if arquivo.endswith(".dcm"):

        caminho = os.path.join(serie_path, arquivo)

        try:

            ds = pydicom.dcmread(caminho)

            if hasattr(ds, "pixel_array"):
                slices.append(ds)

        except:
            pass


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
# CONVERTER PARA HU
# =========================================================

def converter_para_hu(slices):

    imagens = []
    slices_validas = []

    shape_ref = None

    for s in slices:

        try:

            img = s.pixel_array

            if shape_ref is None:
                shape_ref = img.shape

            if img.shape != shape_ref:
                continue

            imagens.append(img)
            slices_validas.append(s)

        except:
            pass

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

print("Volume:", volume_hu.shape)


# =========================================================
# WINDOWING PULMONAR
# =========================================================

def aplicar_window(img, level=-600, width=1500):

    min_hu = level - width // 2
    max_hu = level + width // 2

    img = np.clip(img, min_hu, max_hu)

    return img


volume_window = aplicar_window(volume_hu)


# =========================================================
# GAUSSIAN BLUR
# =========================================================

volume_suave = gaussian_filter(
    volume_window,
    sigma=1
)


# =========================================================
# SEGMENTAÇÃO DO PULMÃO
# =========================================================

mascara_pulmao = np.logical_and(
    volume_suave > -1000,
    volume_suave < -400
)

mascara_pulmao = morphology.remove_small_objects(
    mascara_pulmao,
    min_size=1000
)

mascara_pulmao = ndimage.binary_fill_holes(
    mascara_pulmao
)


# =========================================================
# PEGAR 2 MAIORES COMPONENTES
# =========================================================

labels, num = ndimage.label(mascara_pulmao)

sizes = ndimage.sum(
    mascara_pulmao,
    labels,
    range(1, num + 1)
)

indices = np.argsort(sizes)[-2:] + 1

mascara_pulmao = np.isin(
    labels,
    indices
)


# =========================================================
# ROI PULMONAR
# =========================================================

volume_roi = volume_suave.copy()

volume_roi[~mascara_pulmao] = -1000


# =========================================================
# POSSÍVEIS NÓDULOS
# =========================================================

mascara_nodulos = np.logical_and(
    volume_roi > -250,
    volume_roi < 150
)


# =========================================================
# MORFOLOGIA
# =========================================================

mascara_nodulos = morphology.binary_closing(
    mascara_nodulos,
    morphology.ball(1)
)

mascara_nodulos = ndimage.binary_fill_holes(
    mascara_nodulos
)

mascara_nodulos = morphology.remove_small_objects(
    mascara_nodulos,
    min_size=30
)


# =========================================================
# WATERSHED
# =========================================================

distance = ndimage.distance_transform_edt(
    mascara_nodulos
)

coords = peak_local_max(
    distance,
    footprint=np.ones((3, 3, 3)),
    labels=mascara_nodulos
)

mask = np.zeros(distance.shape, dtype=bool)

mask[tuple(coords.T)] = True

markers, _ = ndimage.label(mask)

labels_ws = watershed(
    -distance,
    markers,
    mask=mascara_nodulos
)


# =========================================================
# FILTRAR COMPONENTES
# =========================================================

mascara_final = np.zeros_like(
    mascara_nodulos,
    dtype=bool
)

props = measure.regionprops(labels_ws)

for prop in props:

    volume = prop.area

    if volume < 40:
        continue

    if volume > 8000:
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

    # remove estruturas alongadas (vasos)
    if razao > 3:
        continue

    mascara_final[
        labels_ws == prop.label
    ] = True


# =========================================================
# VISUALIZAÇÃO SLICE
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
ax[3].imshow(
    mascara_final[slice_id],
    cmap="autumn",
    alpha=0.5
)

ax[3].set_title("Overlay")

plt.show()


# =========================================================
# GERAR MALHA 3D
# =========================================================

# def criar_mesh(binario):

#     verts, faces, normals, values = measure.marching_cubes(
#         binario,
#         level=0
#     )

#     faces = np.hstack([
#         np.full((faces.shape[0], 1), 3),
#         faces
#     ]).astype(np.int32)

#     mesh = pv.PolyData(
#         verts,
#         faces
#     )

#     return mesh




def criar_mesh(binario):

    if np.max(binario) == 0:
        return None

    if np.sum(binario) < 10:
        return None

    try:

        verts, faces, normals, values = measure.marching_cubes(
            binario,
            level=0
        )

        faces = np.hstack([
            np.full((faces.shape[0], 1), 3),
            faces
        ]).astype(np.int32)

        mesh = pv.PolyData(
            verts,
            faces
        )

        return mesh

    except RuntimeError:

        return None

mesh_pulmao = criar_mesh(
    mascara_pulmao.astype(np.uint8)
)

mesh_nodulos = criar_mesh(
    mascara_final.astype(np.uint8)
)
# =========================================================
# VISUALIZAÇÃO 3D
# =========================================================

plotter = pv.Plotter()

plotter.add_mesh(
    mesh_pulmao,
    color="lightblue",
    opacity=0.05
)

# plotter.add_mesh(
#     mesh_nodulos,
#     color="red"
# )
if mesh_nodulos is not None:

    plotter.add_mesh(
        mesh_nodulos,
        color="red"
    )

else:

    print("\nNenhum nódulo encontrado.\n")

plotter.set_background("black")

plotter.show()