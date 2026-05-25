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

# Importa o arquivo central de utilidades
import utils

# =========================================================
# CONFIGURAÇÕES E CAMINHOS
# =========================================================
ROOT = r"C:\exames_dos_pacientes"
XML_ROOT = r"C:\Users\jefte\Downloads\padrao ouro\tcia-lidc-xml"
PACIENTE = "LIDC-IDRI-0164"  # ID do paciente para validação

MOSTRAR_SLICE = True
MOSTRAR_3D = True

# Parâmetros Geométricos Otimizados (Retirados do gerar_mascara.py)
MIN_VOLUME = 20          # Permite nódulos menores
MAX_VOLUME = 4000
MIN_ESFERICIDADE = 0.1   # Mais tolerante a nódulos ovais ou justavasculares
MAX_RAZAO = 5.5          # Não tão rígido, evita descartar nódulos ligeiramente espiculados
MAX_SLICES = 20          # Limita o número de slices (elimina vasos verticais muito longos)
MAX_DIAMETRO_MM = 35

# =========================================================
# 1. CARREGAMENTO E EXTRAÇÃO DE METADADOS ESPACIAIS
# =========================================================
serie_path = os.path.join(ROOT, PACIENTE)
print(f"\n[1/7] Processando Paciente: {PACIENTE}")

slices = utils.carregar_slices(serie_path)
spacing_x = float(slices[0].PixelSpacing[1])
spacing_y = float(slices[0].PixelSpacing[0])
spacing_z = float(slices[0].SliceThickness)

SERIES_UID = slices[0].SeriesInstanceUID

# =========================================================
# 2. BUSCA AUTOMÁTICA DO ARQUIVO XML (PADRÃO OURO)
# =========================================================
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

print(f"XML correspondente encontrado: {xml_encontrado}")

# =========================================================
# 3. PRÉ-PROCESSAMENTO: FILTRAGEM ANATÔMICA E REMOÇÃO DE VIAS AÉREAS
# =========================================================
print("[2/7] Pré-processamento e Janelamento HU...")
volume_hu = utils.converter_para_hu(slices)
volume_window = utils.aplicar_window(volume_hu)
volume_suave = gaussian_filter(volume_window, sigma=1)

print("[3/7] Segmentando parênquima e removendo estruturas tubulares...")
# Segmentação inicial dos lobos pulmonares
mascara_pulmao = np.logical_and(volume_suave > -1000, volume_suave < -400)
mascara_pulmao = morphology.remove_small_objects(mascara_pulmao, min_size=1000)
mascara_pulmao = ndimage.binary_fill_holes(mascara_pulmao)

labels, num = ndimage.label(mascara_pulmao)
sizes = ndimage.sum(mascara_pulmao, labels, range(1, num + 1))
indices = np.argsort(sizes)[-2:] + 1  # Mantém apenas os dois lobos principais
mascara_pulmao = np.isin(labels, indices)

# Estratégia de redução ativa de vias aéreas centrais (Brônquios)
vias_aereas = np.logical_and(volume_suave > -1050, volume_suave < -850)
vias_aereas = np.logical_and(vias_aereas, mascara_pulmao)
vias_aereas = morphology.binary_opening(vias_aereas, morphology.ball(2))

# Subtração morfológica da árvore brônquica
mascara_pulmao_limpa = np.logical_and(mascara_pulmao, ~vias_aereas)

volume_roi = volume_suave.copy()
volume_roi[~mascara_pulmao_limpa] = -1000

# =========================================================
# 4. EXTRAÇÃO DE CANDIDATOS E WATERSHED TRIDIMENSIONAL
# =========================================================
print("[4/7] Isolando estruturas candidatas por densidade...")
mascara_nodulos = np.logical_and(volume_roi > -250, volume_roi < 150)
mascara_nodulos = morphology.binary_closing(mascara_nodulos, morphology.ball(1))
mascara_nodulos = ndimage.binary_fill_holes(mascara_nodulos)
mascara_nodulos = morphology.remove_small_objects(mascara_nodulos, min_size=15)

print("[5/7] Executando Watershed 3D para desconectar vasos de nódulos...")
distance = ndimage.distance_transform_edt(mascara_nodulos)
coords = peak_local_max(distance, footprint=np.ones((3, 3, 3)), labels=mascara_nodulos)

mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndimage.label(mask)
labels_ws = watershed(-distance, markers, mask=mascara_nodulos)

# =========================================================
# 5. FILTRAGEM GEOMÉTRICA AVANÇADA (REDUÇÃO DE VASOS RESTANTES)
# =========================================================
print("[6/7] Aplicando restrições de esfericidade e persistência de fatias...")
mascara_final = np.zeros_like(mascara_nodulos, dtype=bool)
props = measure.regionprops(labels_ws)

for prop in props:
    volume = prop.area
    if volume < MIN_VOLUME or volume > MAX_VOLUME:
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
    esfericidade = menor / maior

    # Critérios combinados aplicados à forma geométrica tridimensional
    if razao > MAX_RAZAO or esfericidade < MIN_ESFERICIDADE:
        continue
        
    # Restrição física de fatias para estruturas cilíndricas que correm no eixo vertical
    if dz > MAX_SLICES:
        continue

    # Validação do diâmetro métrico real (em milímetros)
    diametro_max = max(dx * spacing_x, dy * spacing_y, dz * spacing_z)
    if diametro_max > MAX_DIAMETRO_MM:
        continue

    # Se a região resistiu a todas as restrições tubulares, é confirmada
    mascara_final[labels_ws == prop.label] = True

# =========================================================
# 6. MAPEAMENTO DO PADRÃO OURO (MASCARA GOLD XML)
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

# =========================================================
# 7. CÁLCULO DE MÉTRICAS COMPLEMENTARES E GRAVAÇÃO
# =========================================================
inter = np.logical_and(mascara_final, mascara_gold).sum()
dice = (2 * inter) / (mascara_final.sum() + mascara_gold.sum() + 1e-8)
print(f"\n📊 [RESULTADO] Coeficiente Dice alcançado: {dice:.4f}")

# Preserva o modelo refinado gerado em disco
np.save("mascara_test v3.npy", mascara_final)

# =========================================================
# GRÁFICOS DE ANÁLISE COMPLEMENTAR
# =========================================================
if MOSTRAR_SLICE:
    slice_id = volume_hu.shape[0] // 2
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))

    ax[0].imshow(volume_hu[slice_id], cmap="gray")
    ax[0].set_title("Original")

    ax[1].imshow(mascara_pulmao_limpa[slice_id], cmap="gray")
    ax[1].set_title("Pulmão (Sem Vias Aéreas)")

    ax[2].imshow(mascara_final[slice_id], cmap="gray")
    ax[2].set_title("Algoritmo Refinado")

    ax[3].imshow(mascara_gold[slice_id], cmap="gray")
    ax[3].set_title("Padrão Ouro (XML)")

    ax[4].imshow(volume_hu[slice_id], cmap="gray")
    ax[4].imshow(mascara_gold[slice_id], cmap="Greens", alpha=0.4)
    ax[4].imshow(mascara_final[slice_id], cmap="autumn", alpha=0.4)
    ax[4].set_title("Verde=Gold | Vermelho=Algoritmo")
    plt.show()

if MOSTRAR_3D:
    print("\n📦 Construindo ambiente 3D interativo...")
    mesh_pulmao = utils.criar_mesh(mascara_pulmao_limpa.astype(np.uint8))
    mesh_nodulos = utils.criar_mesh(mascara_final.astype(np.uint8))
    mesh_gold = utils.criar_mesh(mascara_gold.astype(np.uint8))

    plotter = pv.Plotter()
    if mesh_pulmao is not None:
        plotter.add_mesh(mesh_pulmao, color="lightblue", opacity=0.12)
    if mesh_nodulos is not None:
        plotter.add_mesh(mesh_nodulos, color="red", label="Predição Algoritmo")
    if mesh_gold is not None:
        plotter.add_mesh(mesh_gold, color="green", label="Gold Standard")

    plotter.add_legend(bcolor='grey')
    plotter.set_background("black")
    plotter.show()