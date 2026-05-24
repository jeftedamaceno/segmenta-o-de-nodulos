import os
import glob
import numpy as np
import pydicom

from scipy.ndimage import (
    gaussian_filter,
    binary_fill_holes
)

from skimage import measure
from skimage.morphology import (
    binary_closing,
    binary_opening,
    ball,
    remove_small_objects
)

import pyvista as pv


# =========================================================
# CAMINHO
# =========================================================

PACIENTE_PATH = r"C:\exames_dos_pacientes\LIDC-IDRI-0068"


# =========================================================
# CARREGAR DICOM
# =========================================================

def carregar_dicom(pasta):

    arquivos = glob.glob(
        os.path.join(pasta, "**", "*.dcm"),
        recursive=True
    )

    slices = []

    for arq in arquivos:

        try:

            ds = pydicom.dcmread(arq)

            if hasattr(ds, "ImagePositionPatient"):

                slices.append(ds)

        except:
            pass

    slices.sort(
        key=lambda s: float(s.ImagePositionPatient[2])
    )

    volume = np.stack(
        [s.pixel_array for s in slices]
    ).astype(np.int16)

    return volume, slices


# =========================================================
# CONVERSÃO HU
# =========================================================

def converter_hu(volume, slices):

    hu = np.copy(volume)

    for i, s in enumerate(slices):

        slope = s.RescaleSlope
        intercept = s.RescaleIntercept

        hu[i] = hu[i] * slope + intercept

    return hu


# =========================================================
# WINDOW PULMONAR
# =========================================================

def window_pulmao(img,
                  center=-600,
                  width=1500):

    min_val = center - width // 2
    max_val = center + width // 2

    img = np.clip(img, min_val, max_val)

    return img


# =========================================================
# SEGMENTAÇÃO PULMÃO
# =========================================================

def segmentar_pulmao(volume):

    mascara = volume < -400

    mascara = binary_closing(
        mascara,
        ball(2)
    )

    mascara = binary_fill_holes(
        mascara
    )

    return mascara


# =========================================================
# DETECÇÃO DE NÓDULOS
# =========================================================

def detectar_nodulos(volume,
                      mascara_pulmao):

    candidatos = np.logical_and(
        volume > -300,
        volume < 300
    )

    candidatos = np.logical_and(
        candidatos,
        mascara_pulmao
    )

    candidatos = gaussian_filter(
        candidatos.astype(float),
        sigma=1
    ) > 0.2

    candidatos = binary_opening(
        candidatos,
        ball(1)
    )

    candidatos = binary_closing(
        candidatos,
        ball(1)
    )

    candidatos = binary_fill_holes(
        candidatos
    )

    candidatos = remove_small_objects(
        candidatos,
        min_size=20
    )

    return candidatos


# =========================================================
# FILTRO GEOMÉTRICO
# =========================================================

def filtrar_nodulos(binario):

    labels = measure.label(binario)

    props = measure.regionprops(labels)

    final = np.zeros_like(binario)

    for prop in props:

        volume = prop.area

        if volume < 30:
            continue

        if volume > 4000:
            continue

        z, y, x = prop.centroid

        bbox = prop.bbox

        dz = bbox[3] - bbox[0]
        dy = bbox[4] - bbox[1]
        dx = bbox[5] - bbox[2]

        maior = max(dx, dy, dz)
        menor = max(1, min(dx, dy, dz))

        razao = maior / menor

        # remove estruturas muito alongadas
        if razao > 4:
            continue

        # evita objetos encostados na borda
        if (
            x < 20 or
            y < 20 or
            x > binario.shape[2] - 20 or
            y > binario.shape[1] - 20
        ):
            continue

        final[labels == prop.label] = 1

    return final.astype(np.uint8)


# =========================================================
# MESH
# =========================================================

def criar_mesh(binario):

    if np.sum(binario) == 0:
        return None

    verts, faces, _, _ = measure.marching_cubes(
        binario,
        level=0.5
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


# =========================================================
# PIPELINE
# =========================================================

print("\nCarregando DICOMs...")

volume, slices = carregar_dicom(
    PACIENTE_PATH
)

print("Convertendo para HU...")

hu = converter_hu(
    volume,
    slices
)

print("Aplicando window pulmonar...")

hu = window_pulmao(hu)

print("Segmentando pulmão...")

mascara_pulmao = segmentar_pulmao(
    hu
)

print("Detectando candidatos...")

candidatos = detectar_nodulos(
    hu,
    mascara_pulmao
)

print("Aplicando filtros geométricos...")

mascara_final = filtrar_nodulos(
    candidatos
)

print("Salvando mascara_final.npy ...")

np.save(
    "mascara_final.npy",
    mascara_final
)

print("Arquivo salvo com sucesso!")

print(
    f"Nódulos encontrados: {np.max(measure.label(mascara_final))}"
)


# =========================================================
# VISUALIZAÇÃO 3D
# =========================================================

mesh = criar_mesh(
    mascara_final
)

if mesh is not None:

    plotter = pv.Plotter()

    plotter.add_mesh(
        mesh,
        color="red",
        opacity=0.8
    )

    plotter.show()

else:

    print("Nenhum nódulo encontrado.")