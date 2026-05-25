# verção 3.7.py
import os
import numpy as np
import xml.etree.ElementTree as ET
from skimage.draw import polygon
from scipy.ndimage import binary_fill_holes
import pyvista as pv

import utils

PACIENTE_PATH = r"C:\exames_dos_pacientes\LIDC-IDRI-0164"
XML_PATH = r'C:\Users\jefte\projetos em python\ufc 2025 a 2026\segmentação de nodulos\padrao ouro\tcia-lidc-xml\189\068.xml'

def criar_mascara_padrao_ouro(xml_path, shape, uid_to_index):
    mascara = np.zeros(shape, dtype=np.uint8)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'ns': root.tag.split('}')[0].strip('{')}
    
    for nodule in root.findall(".//ns:unblindedReadNodule", ns):
        for roi in nodule.findall(".//ns:roi", ns):
            uid = roi.find("ns:imageSOP_UID", ns)
            if uid is None or uid.text not in uid_to_index:
                continue
            z = uid_to_index[uid.text]
            xs = [int(e.find("ns:xCoord", ns).text) for e in roi.findall(".//ns:edgeMap", ns)]
            ys = [int(e.find("ns:yCoord", ns).text) for e in roi.findall(".//ns:edgeMap", ns)]
            if len(xs) >= 3:
                rr, cc = polygon(ys, xs)
                mascara[z, rr, cc] = 1
    return binary_fill_holes(mascara).astype(np.uint8)

def dice_score(gt, pred):
    soma = np.sum(gt) + np.sum(pred)
    return 2 * np.sum(gt * pred) / soma if soma != 0 else 0

def iou_score(gt, pred):
    union = np.logical_or(gt, pred).sum()
    return np.logical_and(gt, pred).sum() / union if union != 0 else 0

# Fluxo de processamento
volume, slices = utils.carregar_dicom_recursivo(PACIENTE_PATH)
uid_to_index = {s.SOPInstanceUID: i for i, s in enumerate(slices)}
hu = utils.converter_para_hu(slices)

mascara_usuario = np.load("mascara_com vasus.npy").astype(np.uint8)
# mascara_usuario = np.load("mascara_test.npy").astype(np.uint8)
mascara_gt = criar_mascara_padrao_ouro(XML_PATH, hu.shape, uid_to_index)

print(f"\nDice Score: {dice_score(mascara_gt, mascara_usuario):.4f}")
print(f"IoU Score:  {iou_score(mascara_gt, mascara_usuario):.4f}")

# Exibição Tridimensional
plotter = pv.Plotter()
mesh_gt = utils.criar_mesh(mascara_gt, level=0.5)
mesh_pred = utils.criar_mesh(mascara_usuario, level=0.5)

if mesh_gt:
    plotter.add_mesh(mesh_gt, color="green", opacity=0.5, label="Ground Truth")
if mesh_pred:
    plotter.add_mesh(mesh_pred, color="red", opacity=0.5, label="Predição")
plotter.add_legend()
plotter.show()