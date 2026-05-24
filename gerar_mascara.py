import os
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import morphology, measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# Importa o arquivo central de utilidades
import utils

# =========================================================
# CONFIGURAÇÕES E CAMINHOS
# =========================================================
ROOT = r"C:\exames_dos_pacientes"
PACIENTE = "LIDC-IDRI-0068"  # ID do paciente de teste

# # Parâmetros Geométricos Otimizados (Mais tolerantes para não excluir nódulos fracos)
# MIN_VOLUME = 20          # Permite nódulos menores
# MAX_VOLUME = 4000
# MIN_ESFERICIDADE = 0.3   # Mais tolerante a nódulos ovais ou justavasculares
# MAX_RAZAO = 3.5          # Não tão rígido, evita descartar nódulos alongados
# MAX_SLICES = 15
# MAX_DIAMETRO_MM = 35
# Parâmetros Geométricos Otimizados (Mais tolerantes para não excluir nódulos fracos)
MIN_VOLUME = 20          # Permite nódulos menores
MAX_VOLUME = 4000
MIN_ESFERICIDADE = 0.1   # Mais tolerante a nódulos ovais ou justavasculares
MAX_RAZAO = 5.5          # Não tão rígido, evita descartar nódulos alongados
MAX_SLICES = 20
MAX_DIAMETRO_MM = 35

# =========================================================
# PIPELINE DE CARREGAMENTO
# =========================================================
pasta_paciente = os.path.join(ROOT, PACIENTE)
serie_path = utils.encontrar_serie_dicom(pasta_paciente)
slices = utils.carregar_slices(serie_path)

# Metadados de espaçamento físico
spacing_x = float(slices[0].PixelSpacing[1])
spacing_y = float(slices[0].PixelSpacing[0])
spacing_z = float(slices[0].SliceThickness)

# Processamento HU e Janelamento Pulmonar
volume_hu = utils.converter_para_hu(slices)
volume_window = utils.aplicar_window(volume_hu)
volume_suave = gaussian_filter(volume_window, sigma=1)

# =========================================================
# 1. SEGMENTAÇÃO COMPLETA DO PULMÃO
# =========================================================
mascara_pulmao = np.logical_and(volume_suave > -1000, volume_suave < -400)
mascara_pulmao = morphology.remove_small_objects(mascara_pulmao, min_size=1000)
mascara_pulmao = ndimage.binary_fill_holes(mascara_pulmao)

labels, num = ndimage.label(mascara_pulmao)
sizes = ndimage.sum(mascara_pulmao, labels, range(1, num + 1))
indices = np.argsort(sizes)[-2:] + 1  # Isola os dois maiores lobos pulmonares
mascara_pulmao = np.isin(labels, indices)

# =========================================================
# 2. PRÉ-PROCESSAMENTO: REMOÇÃO ATIVA DE VIAS AÉREAS (BRÔNQUIOS)
# =========================================================
# As vias aéreas internas têm densidade de ar (muito baixa, próximo a -950 HU)
vias_aereas = np.logical_and(volume_suave > -1050, volume_suave < -850)
vias_aereas = np.logical_and(vias_aereas, mascara_pulmao)

# Uma abertura morfológica desconecta os finos filamentos das vias aéreas dos lobos do pulmão
vias_aereas = morphology.binary_opening(vias_aereas, morphology.ball(2))

# Subtrai as árvores brônquicas detectadas da nossa área de busca de nódulos
mascara_pulmao_limpa = np.logical_and(mascara_pulmao, ~vias_aereas)

# =========================================================
# 3. EXTRAÇÃO DA REGIAO DE INTERESSE (ROI) SEM VIAS AÉREAS
# =========================================================
volume_roi = volume_suave.copy()
volume_roi[~mascara_pulmao_limpa] = -1000

# Segmentação inicial dos tecidos moles internos (Nódulos + Vasos restantes)
mascara_nodulos = np.logical_and(volume_roi > -250, volume_roi < 150)
mascara_nodulos = morphology.binary_closing(mascara_nodulos, morphology.ball(1))
mascara_nodulos = ndimage.binary_fill_holes(mascara_nodulos)
mascara_nodulos = morphology.remove_small_objects(mascara_nodulos, min_size=15)

# =========================================================
# 4. WATERSHED TRIDIMENSIONAL PARA DESCONECTAR VASOS DE NÓDULOS
# =========================================================
distance = ndimage.distance_transform_edt(mascara_nodulos)
coords = peak_local_max(distance, footprint=np.ones((3, 3, 3)), labels=mascara_nodulos)

mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndimage.label(mask)
labels_ws = watershed(-distance, markers, mask=mascara_nodulos)

# =========================================================
# 5. FILTRAGEM GEOMÉTRICA APÓS DESCONEXÃO E PRÉ-LIMPEZA
# =========================================================
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

    # Critérios combinados com os novos parâmetros flexíveis
    if razao > MAX_RAZAO or esfericidade < MIN_ESFERICIDADE:
        continue
    if dz > MAX_SLICES:
        continue

    # Validação do diâmetro métrico real (em mm)
    diametro_max = max(dx * spacing_x, dy * spacing_y, dz * spacing_z)
    if diametro_max > MAX_DIAMETRO_MM:
        continue

    # Se passou por todos os filtros após a remoção das vias aéreas, armazena
    mascara_final[labels_ws == prop.label] = True

# Salva o resultado final para ser aberto imediatamente na versão 3.7
np.save("mascara_test v3.npy", mascara_final)
print("\n[SUCESSO] Matriz 'mascara_final.npy' gerada e limpa de vias aéreas!")