# utils.py
import os
import glob
import numpy as np
import pydicom
import xml.etree.ElementTree as ET
from skimage import measure
import pyvista as pv

def encontrar_serie_dicom(pasta_paciente, min_arquivos=20):
    """Varre recursivamente a pasta do paciente para encontrar o diretório com os arquivos .dcm."""
    for raiz, _, arquivos in os.walk(pasta_paciente):
        dcms = [f for f in arquivos if f.endswith(".dcm")]
        if len(dcms) > min_arquivos:
            return raiz
    return None

def obter_chave_ordenacao(ds):
    """Retorna a coordenada Z ou indexador para ordenar as fatias DICOM corretamente."""
    if hasattr(ds, "ImagePositionPatient"):
        return float(ds.ImagePositionPatient[2])
    elif hasattr(ds, "SliceLocation"):
        return float(ds.SliceLocation)
    elif hasattr(ds, "InstanceNumber"):
        return int(ds.InstanceNumber)
    return 0

def carregar_slices(serie_path):
    """Carrega e ordena todos os arquivos DICOM de uma série."""
    slices = []
    for arquivo in os.listdir(serie_path):
        if arquivo.endswith(".dcm"):
            caminho = os.path.join(serie_path, arquivo)
            try:
                ds = pydicom.dcmread(caminho)
                if hasattr(ds, "pixel_array"):
                    slices.append(ds)
            except Exception:
                pass
    slices.sort(key=obter_chave_ordenacao)
    return slices

def carregar_dicom_recursivo(pasta):
    """Carrega fatias de forma genérica usando glob recursivo (padrão de alguns scripts)."""
    arquivos = glob.glob(os.path.join(pasta, "**", "*.dcm"), recursive=True)
    slices = []
    for arq in arquivos:
        try:
            ds = pydicom.dcmread(arq)
            if hasattr(ds, "ImagePositionPatient"):
                slices.append(ds)
        except Exception:
            pass
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    return volume, slices

def converter_para_hu(slices):
    """Converte o array cru de fatias DICOM para Unidades Hounsfield (HU)."""
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
        except Exception:
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

def aplicar_window(img, level=-600, width=1500):
    """Aplica o corte de janela (Windowing) para realçar o pulmão."""
    min_hu = level - width // 2
    max_hu = level + width // 2
    return np.clip(img, min_hu, max_hu)

def criar_mesh(binario, level=0.5):
    """Gera uma estrutura PolyData (Mesh 3D) a partir de um volume binário."""
    if np.max(binario) == 0 or np.sum(binario) < 10:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(binario, level=level)
        # Formata as faces para o PyVista: [número de vértices na face, id1, id2, id3]
        faces_pv = np.hstack([
            np.full((faces.shape[0], 1), 3),
            faces
        ]).astype(np.int32)
        mesh = pv.PolyData(verts, faces_pv)
        return mesh
    except (RuntimeError, ValueError):
        return None