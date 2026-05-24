import os
import glob
import numpy as np
import pydicom
import xml.etree.ElementTree as ET

from skimage.draw import polygon
from skimage.measure import marching_cubes
from scipy.ndimage import binary_fill_holes

import pyvista as pv


# =========================================================
# CAMINHOS
# =========================================================

PACIENTE_PATH = r"C:\exames_dos_pacientes\LIDC-IDRI-0088"

XML_PATH = r'C:\Users\jefte\projetos em python\ufc 2025 a 2026\segmentação de nodulos\padrao ouro\tcia-lidc-xml\185\088.xml'


# =========================================================
# CARREGAR DICOMS
# =========================================================

def carregar_dicom(pasta):

    arquivos = glob.glob(os.path.join(pasta, "**", "*.dcm"), recursive=True)

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

    uid_to_index = {}

    for i, s in enumerate(slices):

        uid_to_index[s.SOPInstanceUID] = i

    return volume, slices, uid_to_index


# =========================================================
# HU
# =========================================================

def converter_hu(volume, slices):

    hu = np.copy(volume)

    for i, s in enumerate(slices):

        slope = s.RescaleSlope
        intercept = s.RescaleIntercept

        hu[i] = hu[i] * slope + intercept

    return hu


# =========================================================
# MÁSCARA PADRÃO OURO
# =========================================================

def criar_mascara_padrao_ouro(xml_path,
                              shape,
                              uid_to_index):

    mascara = np.zeros(shape, dtype=np.uint8)

    tree = ET.parse(xml_path)

    root = tree.getroot()

    ns = {'ns': root.tag.split('}')[0].strip('{')}

    for nodule in root.findall(".//ns:unblindedReadNodule", ns):

        for roi in nodule.findall(".//ns:roi", ns):

            uid = roi.find("ns:imageSOP_UID", ns)

            if uid is None:
                continue

            uid = uid.text

            if uid not in uid_to_index:
                continue

            z = uid_to_index[uid]

            xs = []
            ys = []

            for edge in roi.findall(".//ns:edgeMap", ns):

                x = int(edge.find("ns:xCoord", ns).text)
                y = int(edge.find("ns:yCoord", ns).text)

                xs.append(x)
                ys.append(y)

            if len(xs) < 3:
                continue

            rr, cc = polygon(ys, xs)

            mascara[z, rr, cc] = 1

    mascara = binary_fill_holes(mascara)

    return mascara.astype(np.uint8)


# =========================================================
# DICE SCORE
# =========================================================

def dice_score(gt, pred):

    inter = np.sum(gt * pred)

    soma = np.sum(gt) + np.sum(pred)

    if soma == 0:
        return 0

    return 2 * inter / soma


# =========================================================
# IOU
# =========================================================

def iou_score(gt, pred):

    inter = np.logical_and(gt, pred).sum()

    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return 0

    return inter / union


# =========================================================
# MESH
# =========================================================

def criar_mesh(binario):

    if np.sum(binario) == 0:
        return None

    verts, faces, _, _ = marching_cubes(
        binario,
        level=0.5
    )

    faces = np.hstack(
        [np.full((faces.shape[0], 1), 3), faces]
    ).astype(np.int32)

    mesh = pv.PolyData(
        verts,
        faces
    )

    return mesh


# =========================================================
# CARREGAMENTO
# =========================================================

volume, slices, uid_to_index = carregar_dicom(
    PACIENTE_PATH
)

hu = converter_hu(volume, slices)


# =========================================================
# SUA SEGMENTAÇÃO
# =========================================================

# carregue sua máscara final aqui

mascara_usuario = np.load(
    "mascara_final.npy"
)

mascara_usuario = mascara_usuario.astype(np.uint8)


# =========================================================
# PADRÃO OURO
# =========================================================

mascara_gt = criar_mascara_padrao_ouro(
    XML_PATH,
    hu.shape,
    uid_to_index
)


# =========================================================
# MÉTRICAS
# =========================================================

dice = dice_score(
    mascara_gt,
    mascara_usuario
)

iou = iou_score(
    mascara_gt,
    mascara_usuario
)

print(f"\nDice Score: {dice:.4f}")
print(f"IoU Score:  {iou:.4f}")


# =========================================================
# MESHES
# =========================================================

mesh_gt = criar_mesh(mascara_gt)

mesh_pred = criar_mesh(mascara_usuario)


# =========================================================
# VISUALIZAÇÃO
# =========================================================

plotter = pv.Plotter()

if mesh_gt is not None:

    plotter.add_mesh(
        mesh_gt,
        color="green",
        opacity=0.6,
        label="Padrao Ouro"
    )

if mesh_pred is not None:

    plotter.add_mesh(
        mesh_pred,
        color="red",
        opacity=0.5,
        label="Segmentacao"
    )

plotter.add_legend()

plotter.show()