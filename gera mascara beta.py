import os
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage import measure
from skimage.morphology import binary_closing, binary_opening, ball, remove_small_objects

import utils

PACIENTE_PATH = r"C:\exames_dos_pacientes\LIDC-IDRI-0068"

def segmentar_pulmao(volume):
    mascara = volume < -400
    return binary_fill_holes(binary_closing(mascara, ball(2)))

def detectar_nodulos(volume, mascara_pulmao):
    candidatos = np.logical_and(volume > -300, volume < 300)
    candidatos = np.logical_and(candidatos, mascara_pulmao)
    candidatos = gaussian_filter(candidatos.astype(float), sigma=1) > 0.2
    candidatos = binary_opening(candidatos, ball(1))
    candidatos = binary_closing(candidatos, ball(1))
    return remove_small_objects(binary_fill_holes(candidatos), min_size=20)

def filtrar_nodulos(binario):
    labels = measure.label(binario)
    props = measure.regionprops(labels)
    final = np.zeros_like(binario)
    for prop in props:
        if prop.area < 30 or prop.area > 4000:
            continue
        bbox = prop.bbox
        dz, dy, dx = bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]
        if (max(dx, dy, dz) / max(1, min(dx, dy, dz))) > 4:
            continue
        final[labels == prop.label] = 1
    return final.astype(np.uint8)

# Execução limpa do fluxo principal
print("\nCarregando DICOMs...")
volume, slices = utils.carregar_dicom_recursivo(PACIENTE_PATH)

print("Convertendo e filtrando em HU...")
hu = utils.converter_para_hu(slices)
hu_windowed = utils.aplicar_window(hu)

mascara_pulmao = segmentar_pulmao(hu_windowed)
candidatos = detectar_nodulos(hu_windowed, mascara_pulmao)
mascara_final = filtrar_nodulos(candidatos)

np.save("mascara_final.npy", mascara_final)
print("Arquivo mascara_final.npy salvo com sucesso!")