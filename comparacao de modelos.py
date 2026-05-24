import os
import numpy as np
import pydicom
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from scipy import ndimage
from skimage import morphology
from skimage.draw import polygon
from skimage import measure

import pyvista as pv



# =========================================================
# CONFIGURAÇÕES
# =========================================================

ROOT = r"C:\exames_dos_pacientes"

PACIENTE = "LIDC-IDRI-0068"

XML_PATH = r"C:\Users\jefte\Downloads\padrao ouro\tcia-lidc-xml\188\068.xml"



# =========================================================
# ENCONTRAR SÉRIE DICOM
# =========================================================

pasta_paciente = os.path.join(
    ROOT,
    PACIENTE
)

serie_path = None

for raiz, dirs, arquivos in os.walk(pasta_paciente):

    dcm = [
        f for f in arquivos
        if f.endswith(".dcm")
    ]

    if len(dcm) > 20:

        serie_path = raiz
        break


print("\nSérie encontrada:")
print(serie_path)



# =========================================================
# CARREGAR DICOM
# =========================================================

slices = []

for arquivo in os.listdir(serie_path):

    if arquivo.endswith(".dcm"):

        caminho = os.path.join(
            serie_path,
            arquivo
        )

        try:

            ds = pydicom.dcmread(caminho)

            if hasattr(ds, "pixel_array"):
                slices.append(ds)

        except:
            pass



# =========================================================
# ORDENAÇÃO
# =========================================================

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
# CONVERSÃO PARA HU
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

print("\nVolume HU:", volume_hu.shape)



# =========================================================
# SEGMENTAÇÃO DO PULMÃO
# =========================================================

mascara_pulmao = np.logical_and(
    volume_hu > -1000,
    volume_hu < -400
)

mascara_pulmao = morphology.remove_small_objects(
    mascara_pulmao,
    min_size=1500
)

mascara_pulmao = ndimage.binary_fill_holes(
    mascara_pulmao
)



# =========================================================
# PEGAR SOMENTE OS 2 MAIORES COMPONENTES
# =========================================================

labels, num = ndimage.label(
    mascara_pulmao
)

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
# REMOVER CASCA EXTERNA
# =========================================================

estrutura = morphology.ball(3)

mascara_pulmao = morphology.binary_erosion(
    mascara_pulmao,
    estrutura
)



# =========================================================
# MODELO DE DETECÇÃO
# =========================================================

mascara_modelo = np.logical_and(
    volume_hu > -300,
    volume_hu < 100
)

mascara_modelo = np.logical_and(
    mascara_modelo,
    mascara_pulmao
)



# =========================================================
# REMOVER PEQUENOS OBJETOS
# =========================================================

mascara_modelo = morphology.remove_small_objects(
    mascara_modelo,
    min_size=25
)



# =========================================================
# LABELING 3D
# =========================================================

labels_modelo, num_modelos = ndimage.label(
    mascara_modelo
)

objetos = ndimage.find_objects(
    labels_modelo
)

mascara_filtrada = np.zeros_like(
    mascara_modelo
)



# =========================================================
# FILTRO DE VOLUME
# =========================================================

for i, obj in enumerate(objetos):

    if obj is None:
        continue

    regiao = labels_modelo[obj] == (i + 1)

    volume_obj = regiao.sum()

    if 20 < volume_obj < 10000:

        mascara_filtrada[obj] |= regiao


mascara_modelo = mascara_filtrada



# =========================================================
# MÁSCARA PADRÃO OURO
# =========================================================

mascara_ouro = np.zeros(
    volume_hu.shape,
    dtype=np.uint8
)



# =========================================================
# MAPA Z
# =========================================================

mapa_z = {}

for i, s in enumerate(slices):

    if hasattr(s, "ImagePositionPatient"):

        z = float(
            s.ImagePositionPatient[2]
        )

        mapa_z[z] = i



# =========================================================
# LER XML
# =========================================================

tree = ET.parse(XML_PATH)

root = tree.getroot()

print("\nLendo XML...")


total_rois = 0



# =========================================================
# EXTRAÇÃO DAS ROIS
# =========================================================

for nodule in root.iter():

    if "unblindedreadnodule" in nodule.tag.lower():

        for roi in nodule.iter():

            if "roi" in roi.tag.lower():

                z_pos = None

                pontos_x = []
                pontos_y = []

                for item in roi:

                    tag = item.tag.lower()

                    if "imagezposition" in tag:

                        z_pos = float(item.text)

                    if "edgemap" in tag:

                        x = None
                        y = None

                        for coord in item:

                            ctag = coord.tag.lower()

                            if "xcoord" in ctag:
                                x = int(coord.text)

                            if "ycoord" in ctag:
                                y = int(coord.text)

                        if x is not None and y is not None:

                            pontos_x.append(x)
                            pontos_y.append(y)

                if z_pos is not None and len(pontos_x) > 2:

                    z_lista = np.array(
                        list(mapa_z.keys())
                    )

                    indice_mais_proximo = np.argmin(
                        np.abs(z_lista - z_pos)
                    )

                    z_real = z_lista[
                        indice_mais_proximo
                    ]

                    indice = mapa_z[z_real]

                    rr, cc = polygon(
                        pontos_y,
                        pontos_x,
                        volume_hu[indice].shape
                    )

                    mascara_ouro[
                        indice,
                        rr,
                        cc
                    ] = 1

                    total_rois += 1



print(f"\nROIs encontradas: {total_rois}")

print(
    f"Voxels padrão ouro: {mascara_ouro.sum()}"
)



# =========================================================
# MÉTRICAS
# =========================================================

intersecao = np.logical_and(
    mascara_modelo,
    mascara_ouro
).sum()

uniao = np.logical_or(
    mascara_modelo,
    mascara_ouro
).sum()

dice = (
    2 * intersecao
) / (
    mascara_modelo.sum() +
    mascara_ouro.sum() + 1e-8
)

iou = (
    intersecao /
    (uniao + 1e-8)
)

sensibilidade = (
    intersecao /
    (mascara_ouro.sum() + 1e-8)
)

falso_positivo = np.logical_and(
    mascara_modelo,
    ~mascara_ouro
).sum()



print("\n========== RESULTADOS ==========")

print(f"IoU: {iou:.4f}")

print(f"Dice: {dice:.4f}")

print(f"Sensibilidade: {sensibilidade:.4f}")

print(f"Falsos Positivos: {falso_positivo}")



# =========================================================
# SLICES COM NÓDULOS
# =========================================================

slices_com_nodulo = np.where(
    mascara_ouro.sum(axis=(1,2)) > 0
)[0]

print(
    "\nSlices com nódulo:",
    slices_com_nodulo
)



# =========================================================
# VISUALIZAÇÃO 2D
# =========================================================

for indice in slices_com_nodulo[:5]:

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(22,6)
    )

    # ORIGINAL
    ax[0].imshow(
        volume_hu[indice],
        cmap="gray",
        vmin=-1000,
        vmax=400
    )

    ax[0].set_title(
        f"Tomografia Slice {indice}"
    )

    ax[0].axis("off")


    # PADRÃO OURO
    ax[1].imshow(
        volume_hu[indice],
        cmap="gray",
        vmin=-1000,
        vmax=400
    )

    ax[1].imshow(
        mascara_ouro[indice],
        cmap="Reds",
        alpha=0.6
    )

    ax[1].set_title(
        "Padrão Ouro"
    )

    ax[1].axis("off")


    # MODELO
    ax[2].imshow(
        volume_hu[indice],
        cmap="gray",
        vmin=-1000,
        vmax=400
    )

    ax[2].imshow(
        mascara_modelo[indice],
        cmap="Blues",
        alpha=0.6
    )

    ax[2].set_title(
        "Seu Modelo"
    )

    ax[2].axis("off")


    # COMPARAÇÃO
    comparacao = np.zeros(
        (*mascara_ouro[indice].shape, 3)
    )

    # verde -> acerto
    comparacao[
        np.logical_and(
            mascara_ouro[indice],
            mascara_modelo[indice]
        )
    ] = [0,1,0]

    # vermelho -> ouro perdido
    comparacao[
        np.logical_and(
            mascara_ouro[indice],
            ~mascara_modelo[indice]
        )
    ] = [1,0,0]

    # azul -> falso positivo
    comparacao[
        np.logical_and(
            ~mascara_ouro[indice],
            mascara_modelo[indice]
        )
    ] = [0,0,1]


    ax[3].imshow(
        volume_hu[indice],
        cmap="gray",
        vmin=-1000,
        vmax=400
    )

    ax[3].imshow(
        comparacao,
        alpha=0.7
    )

    ax[3].set_title(
        "Verde=Acerto | Vermelho=Erro | Azul=FP"
    )

    ax[3].axis("off")

    plt.tight_layout()
    plt.show()



# =========================================================
# FUNÇÃO MALHA 3D
# =========================================================

def criar_mesh(binario):

    verts, faces, normals, values = measure.marching_cubes(
        binario,
        level=0
    )

    faces = np.hstack([
        np.full((faces.shape[0],1), 3),
        faces
    ]).astype(np.int32)

    return pv.PolyData(
        verts,
        faces
    )



# =========================================================
# GERAR MALHAS
# =========================================================

print("\nCriando malhas 3D...")


mesh_pulmao = criar_mesh(
    mascara_pulmao.astype(np.uint8)
)

mesh_ouro = criar_mesh(
    mascara_ouro.astype(np.uint8)
)

mesh_modelo = criar_mesh(
    mascara_modelo.astype(np.uint8)
)



# =========================================================
# VISUALIZAÇÃO 3D
# =========================================================

plotter = pv.Plotter()

# pulmão transparente
plotter.add_mesh(
    mesh_pulmao,
    color="white",
    opacity=0.05
)

# padrão ouro
plotter.add_mesh(
    mesh_ouro,
    color="green",
    opacity=0.8,
    label="Padrão Ouro"
)

# modelo
plotter.add_mesh(
    mesh_modelo,
    color="red",
    opacity=0.5,
    label="Seu Modelo"
)

plotter.add_legend()

plotter.set_background(
    "black"
)

plotter.show()