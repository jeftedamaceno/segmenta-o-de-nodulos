import os
import numpy as np
import pydicom
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from scipy import ndimage

from skimage.draw import polygon
from skimage.filters import threshold_otsu, frangi
from skimage.measure import label, regionprops
from skimage.morphology import (
    remove_small_objects,
    binary_closing,
    binary_opening,
    disk
)

# =========================================================
# CAMINHOS
# =========================================================

ROOT = r"C:\exames_dos_pacientes"

PACIENTE = "LIDC-IDRI-0068"

XML_PATH = r"C:\Users\jefte\Downloads\padrao ouro\tcia-lidc-xml\189\068.xml"

PACIENTE_PATH = os.path.join(ROOT, PACIENTE)

# =========================================================
# LER XML
# =========================================================

tree = ET.parse(XML_PATH)
root = tree.getroot()

ns = {'ns': 'http://www.nih.gov'}

# =========================================================
# PEGAR UID DO XML
# =========================================================

series_uid_xml = None

for elem in root.iter():

    tag = elem.tag.lower()

    if "seriesinstanceuid" in tag:
        series_uid_xml = elem.text
        break

print("\n==============================")
print("UID ENCONTRADO NO XML")
print("==============================")
print(series_uid_xml)

# =========================================================
# LER DICOMS
# =========================================================

arquivos = []

for raiz, dirs, files in os.walk(PACIENTE_PATH):

    for f in files:

        if f.endswith(".dcm"):

            arquivos.append(os.path.join(raiz, f))

print("\nTotal DICOMs encontrados:", len(arquivos))

# =========================================================
# CARREGAR SLICES
# =========================================================

slices = []

for arq in arquivos:

    try:

        ds = pydicom.dcmread(arq)

        if hasattr(ds, 'ImagePositionPatient'):

            slices.append(ds)

    except:
        pass

# =========================================================
# ORDENAR SLICES CORRETAMENTE
# =========================================================

slices.sort(
    key=lambda s: float(s.ImagePositionPatient[2])
)

print("\nSlices carregados:", len(slices))

# =========================================================
# UID DO DICOM
# =========================================================

series_uid_dicom = slices[0].SeriesInstanceUID

print("\n==============================")
print("UID ENCONTRADO NO DICOM")
print("==============================")
print(series_uid_dicom)

# =========================================================
# VALIDAR PACIENTE
# =========================================================

if series_uid_xml == series_uid_dicom:

    print("\n[OK] XML COMPATÍVEL COM PACIENTE")

else:

    print("\n[ERRO] XML NÃO PERTENCE A ESTE PACIENTE")

# =========================================================
# CONVERTER PARA VOLUME
# =========================================================

volume = np.stack([
    s.pixel_array for s in slices
]).astype(np.int16)

# =========================================================
# CONVERTER PARA HU
# =========================================================

for i, s in enumerate(slices):

    intercept = s.RescaleIntercept
    slope = s.RescaleSlope

    volume[i] = volume[i] * slope + intercept

# =========================================================
# NORMALIZAÇÃO
# =========================================================

volume = np.clip(volume, -1000, 400)

# =========================================================
# SEGMENTAÇÃO DO PULMÃO
# =========================================================

mascara_pulmao = np.zeros_like(volume).astype(bool)

for i in range(volume.shape[0]):

    img = volume[i]

    # pulmão geralmente abaixo de -400 HU
    mask = img < -400

    # morfologia
    mask = binary_opening(mask, disk(2))
    mask = binary_closing(mask, disk(5))

    # remover pequenos objetos
    mask = remove_small_objects(mask, 500)

    # componentes conectados
    lbl = label(mask)

    props = regionprops(lbl)

    nova = np.zeros_like(mask)

    for p in props:

        if p.area > 1000:

            nova[lbl == p.label] = 1

    mascara_pulmao[i] = nova

# =========================================================
# APLICAR MÁSCARA PULMONAR
# =========================================================

volume_pulmao = volume.copy()

volume_pulmao[~mascara_pulmao] = -1000

# =========================================================
# FILTRO FRANGI
# REMOVE VASOS
# =========================================================

print("\nAplicando filtro Frangi...")

volume_frangi = np.zeros_like(volume_pulmao).astype(float)

for i in range(volume.shape[0]):

    img = volume_pulmao[i]

    norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    volume_frangi[i] = frangi(norm)

# =========================================================
# DETECÇÃO DE NÓDULOS
# =========================================================

mascara_modelo = np.zeros_like(volume).astype(bool)

for i in range(volume.shape[0]):

    img = volume_pulmao[i]

    # threshold nodular
    candidatos = (
        (img > -300) &
        (img < 300)
    )

    # remover vasos
    candidatos &= volume_frangi[i] < 0.05

    candidatos = binary_opening(
        candidatos,
        disk(2)
    )

    candidatos = binary_closing(
        candidatos,
        disk(3)
    )

    candidatos = remove_small_objects(
        candidatos,
        15
    )

    lbl = label(candidatos)

    final = np.zeros_like(candidatos)

    for reg in regionprops(lbl):

        area = reg.area

        ecc = reg.eccentricity

        solidity = reg.solidity

        # FILTRO DE NÓDULO
        if (
            20 < area < 1500 and
            ecc < 0.80 and
            solidity > 0.6
        ):

            final[lbl == reg.label] = 1

    mascara_modelo[i] = final

# =========================================================
# CRIAR PADRÃO OURO
# =========================================================

mascara_ouro = np.zeros_like(volume).astype(bool)

# mapa z -> slice
z_slices = np.array([
    float(s.ImagePositionPatient[2])
    for s in slices
])

# =========================================================
# EXTRAIR NÓDULOS DO XML
# =========================================================

for roi in root.iter():

    if "roi" not in roi.tag.lower():
        continue

    z = None

    pontos_x = []
    pontos_y = []

    for item in roi:

        tag = item.tag.lower()

        if "imagezposition" in tag:

            z = float(item.text)

        if "edgemap" in tag.lower():

            x = None
            y = None

            for p in item:

                ptag = p.tag.lower()

                if "xcoord" in ptag:
                    x = int(float(p.text))

                if "ycoord" in ptag:
                    y = int(float(p.text))

            if x is not None and y is not None:

                pontos_x.append(x)
                pontos_y.append(y)

    if z is None:
        continue

    if len(pontos_x) < 3:
        continue

    # achar slice correto
    indice = np.argmin(np.abs(z_slices - z))

    rr, cc = polygon(
        pontos_y,
        pontos_x,
        volume.shape[1:]
    )

    mascara_ouro[indice, rr, cc] = 1

# =========================================================
# MÉTRICAS
# =========================================================

inter = np.logical_and(
    mascara_modelo,
    mascara_ouro
).sum()

union = np.logical_or(
    mascara_modelo,
    mascara_ouro
).sum()

iou = inter / (union + 1e-8)

dice = (
    2 * inter
) / (
    mascara_modelo.sum() +
    mascara_ouro.sum() + 1e-8
)

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(f"IoU: {iou:.4f}")
print(f"Dice: {dice:.4f}")

# =========================================================
# TP FP FN
# =========================================================

tp = mascara_modelo & mascara_ouro
fp = mascara_modelo & (~mascara_ouro)
fn = mascara_ouro & (~mascara_modelo)

# =========================================================
# PEGAR SLICES COM NÓDULOS
# =========================================================

slices_nodulos = np.where(
    mascara_ouro.sum(axis=(1,2)) > 0
)[0]

print("\nSlices com nódulos:", slices_nodulos)

# =========================================================
# VISUALIZAÇÃO
# =========================================================

if len(slices_nodulos) > 0:

    idx = slices_nodulos[
        len(slices_nodulos)//2
    ]

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(20,5)
    )

    ax[0].imshow(
        volume_pulmao[idx],
        cmap='gray'
    )

    ax[0].set_title("Pulmão")

    ax[1].imshow(
        volume_pulmao[idx],
        cmap='gray'
    )

    ax[1].imshow(
        mascara_ouro[idx],
        alpha=0.6,
        cmap='Greens'
    )

    ax[1].set_title("Padrão Ouro")

    ax[2].imshow(
        volume_pulmao[idx],
        cmap='gray'
    )

    ax[2].imshow(
        mascara_modelo[idx],
        alpha=0.6,
        cmap='Reds'
    )

    ax[2].set_title("Seu Modelo")

    overlay = np.zeros(
        (*tp[idx].shape, 3)
    )

    overlay[tp[idx]] = [0,1,0]
    overlay[fp[idx]] = [0,0,1]
    overlay[fn[idx]] = [1,0,0]

    ax[3].imshow(
        volume_pulmao[idx],
        cmap='gray'
    )

    ax[3].imshow(
        overlay,
        alpha=0.7
    )

    ax[3].set_title(
        "Verde=TP Azul=FP Vermelho=FN"
    )

    for a in ax:
        a.axis("off")

    plt.tight_layout()
    plt.show()

else:

    print("\nNenhum nódulo encontrado no XML")