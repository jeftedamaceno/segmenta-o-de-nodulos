import os
import numpy as np
import pydicom

from scipy import ndimage
from skimage import morphology
from skimage import measure

import pyvista as pv



ROOT = r"C:\exames_dos_pacientes"
PACIENTE = "LIDC-IDRI-0068"


# ==========================================
# ENCONTRAR SÉRIE
# ==========================================

pasta_paciente = os.path.join(ROOT, PACIENTE)

serie_path = None

for raiz, dirs, arquivos in os.walk(pasta_paciente):

    dcm = [f for f in arquivos if f.endswith(".dcm")]

    if len(dcm) > 20:
        serie_path = raiz
        break

print("Série encontrada:")
print(serie_path)


# ==========================================
# CARREGAR SLICES
# ==========================================

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


# ==========================================
# CONVERTER PARA HU
# ==========================================

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


# ==========================================
# SEGMENTAÇÃO MELHORADA DO PULMÃO
# ==========================================

mascara_pulmao = np.logical_and(
    volume_hu > -1000,
    volume_hu < -400
)

# remover pequenos objetos
mascara_pulmao = morphology.remove_small_objects(
    mascara_pulmao,
    min_size=1000
)

# preencher buracos
mascara_pulmao = ndimage.binary_fill_holes(
    mascara_pulmao
)

# ==========================================
# PEGAR APENAS OS 2 MAIORES COMPONENTES
# ==========================================

labels, num = ndimage.label(mascara_pulmao)

sizes = ndimage.sum(
    mascara_pulmao,
    labels,
    range(1, num + 1)
)

indices = np.argsort(sizes)[-2:] + 1

mascara_limpa = np.isin(
    labels,
    indices
)

# ==========================================
# EROSÃO PARA REMOVER A CASCA EXTERNA
# ==========================================

estrutura = morphology.ball(3)

mascara_interna = morphology.binary_erosion(
    mascara_limpa,
    estrutura
)

# usar esta máscara final
mascara_pulmao = mascara_interna

# ==========================================
# POSSÍVEIS NÓDULOS
# ==========================================

mascara_nodulos = np.logical_and(
    volume_hu > -300,
    volume_hu < 100
)

mascara_nodulos = np.logical_and(
    mascara_nodulos,
    mascara_pulmao
)


# ==========================================
# REMOVER RUÍDOS DOS NÓDULOS
# ==========================================

mascara_nodulos = morphology.remove_small_objects(
    mascara_nodulos,
    min_size=20
)


# ==========================================
# COMPONENTES CONECTADOS
# ==========================================

labels, num = ndimage.label(mascara_nodulos)

print(f"Nódulos encontrados: {num}")


# ==========================================
# GERAR MALHA 3D
# ==========================================

def criar_mesh(binario):

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


# ==========================================
# MALHA DO PULMÃO
# ==========================================

mesh_pulmao = criar_mesh(
    mascara_pulmao.astype(np.uint8)
)


# ==========================================
# MALHA DOS NÓDULOS
# ==========================================

mesh_nodulos = criar_mesh(
    mascara_nodulos.astype(np.uint8)
)


# ==========================================
# VISUALIZAÇÃO
# ==========================================

plotter = pv.Plotter()

# pulmão transparente
plotter.add_mesh(
    mesh_pulmao,
    color="lightblue",
    opacity=0.08
)

# nódulos destacados
plotter.add_mesh(
    mesh_nodulos,
    color="red"
)

plotter.set_background("black")

plotter.show()