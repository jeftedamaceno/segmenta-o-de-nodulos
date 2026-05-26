import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from skimage import morphology, measure
from skimage.draw import polygon
from scipy.ndimage import binary_fill_holes
from collections import deque
import pyvista as pv

import utils


ROOT = r"C:\exames_dos_pacientes"
# Agora mapeamos a pasta raiz que contém todas as subpastas numéricas de XMLs
XML_ROOT = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\segmentação de nodulos\padrao ouro\tcia-lidc-xml"

# Lista de pacientes para processamento em lote
LISTA_PACIENTES = ["LIDC-IDRI-0088"]
#  "LIDC-IDRI-0164"
EXIBIR_PULMAO_OPACO = True    
EXIBIR_APENAS_MASCARA = False  

# Parâmetros Geométricos Restritivos do Seu Método
MIN_VOLUME = 10          
MAX_VOLUME = 4000
MIN_ESFERICIDADE = 0.2   
MAX_RAZAO = 9.5          
MAX_SLICES = 40
MAX_DIAMETRO_MM = 35

RAIO_MIN_TUNEL_VOXELS = 1.6       
MAX_SLICES_TUNEL = 42             
MIN_GRADIENTE_CONTINUIDADE = 4.0 

# =========================================================
# FUNÇÕES AUXILIARES DA ARQUITETURA
# =========================================================
def buscar_xml_exclusivo(xml_root, cod_paciente):
    """
    Busca de forma inteligente e recursiva o arquivo XML do paciente.
    Converte o padrão de ID 'LIDC-IDRI-0068' para o nome do arquivo '068.xml'.
    """
    # Extrai os últimos 3 caracteres numéricos do ID do paciente (ex: '0068' -> '068')
    id_3_digitos = cod_paciente.split("-")[-1][-3:]
    nome_procurado = f"{id_3_digitos}.xml"
    
    # Faz uma varredura em todas as subpastas procurando pelo arquivo específico
    padrao_busca = os.path.join(xml_root, "**", nome_procurado)
    arquivos_encontrados = glob.glob(padrao_busca, recursive=True)
    
    if arquivos_encontrados:
        return arquivos_encontrados[0] # Retorna o caminho completo do arquivo encontrado
    return None

def extrair_padrao_ouro_puro(xml_path, shape_referencia):
    """Extrai os contornos do XML sem aplicar pré-processamentos ou filtros."""
    if xml_path is None or not os.path.exists(xml_path):
        print("⚠️ XML correspondente não fornecido ou inválido. Retornando padrão ouro vazio.")
        return np.zeros(shape_referencia, dtype=bool)
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {'ns': root.tag.split('}')[0].strip('{')}

    fatias_nodulo = []
    for nodule in root.findall(".//ns:unblindedReadNodule", ns):
        for roi in nodule.findall(".//ns:roi", ns):
            xs = [int(e.find("ns:xCoord", ns).text) for e in roi.findall(".//ns:edgeMap", ns)]
            ys = [int(e.find("ns:yCoord", ns).text) for e in roi.findall(".//ns:edgeMap", ns)]
            
            if len(xs) >= 3:
                matriz_fatia = np.zeros((512, 512), dtype=bool)
                rr, cc = polygon(ys, xs, shape=matriz_fatia.shape)
                matriz_fatia[rr, cc] = True
                matriz_fatia = binary_fill_holes(matriz_fatia)
                fatias_nodulo.append(matriz_fatia)

    mascara_gold = np.zeros(shape_referencia, dtype=bool)
    if len(fatias_nodulo) > 0:
        z_centro = shape_referencia[0] // 2
        z_inicio = max(0, z_centro - (len(fatias_nodulo) // 2))
        for i, fatia in enumerate(fatias_nodulo):
            if z_inicio + i < mascara_gold.shape[0]:
                mascara_gold[z_inicio + i] = fatia
    return mascara_gold

def calcular_magnitude_gradiente_3d(volume):
    s_z = sobel(volume, axis=0)
    s_y = sobel(volume, axis=1)
    s_x = sobel(volume, axis=2)
    return np.sqrt(s_z**2 + s_y**2 + s_x**2)

def crescimento_regioes_condicional(mascara_nodulos, mapa_gradiente):
    mapa_dist = ndimage.distance_transform_edt(mascara_nodulos)
    from skimage.feature import peak_local_max
    coords_sementes = peak_local_max(mapa_dist, footprint=np.ones((3, 3, 3)), labels=mascara_nodulos)
    
    mascara_crescida = np.zeros_like(mascara_nodulos, dtype=bool)
    visitados_global = np.zeros_like(mascara_nodulos, dtype=bool)
    
    vizinhos_3d = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0: continue
                vizinhos_3d.append((dz, dy, dx))
    
    for semente in coords_sementes:
        sz, sy, sx = semente
        if visitados_global[sz, sy, sx]: continue
            
        fila = deque([(sz, sy, sx)])
        pixels_da_regiao = []
        abortar_crescimento = False
        z_inicial = sz
        
        while fila:
            cz, cy, cx = fila.popleft()
            if visitados_global[cz, cy, cx]: continue
                
            if mapa_gradiente[cz, cy, cx] < MIN_GRADIENTE_CONTINUIDADE and mapa_dist[cz, cy, cx] < 1.8:
                continue
                
            distancia_fatias_z = abs(cz - z_inicial)
            raio_local = mapa_dist[cz, cy, cx]
            if distancia_fatias_z >= MAX_SLICES_TUNEL and raio_local <= RAIO_MIN_TUNEL_VOXELS:
                abortar_crescimento = True
                break
                
            visitados_global[cz, cy, cx] = True
            pixels_da_regiao.append((cz, cy, cx))
            
            for dz, dy, dx in vizinhos_3d:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if 0 <= nz < mascara_nodulos.shape[0] and 0 <= ny < mascara_nodulos.shape[1] and 0 <= nx < mascara_nodulos.shape[2]:
                    if mascara_nodulos[nz, ny, nx] and not visitados_global[nz, ny, nx]:
                        fila.append((nz, ny, nx))
                        
        if not abortar_crescimento and len(pixels_da_regiao) >= MIN_VOLUME:
            for cz, cy, cx in pixels_da_regiao:
                mascara_crescida[cz, cy, cx] = True
                
    return mascara_crescida

def criar_mesh(binario):
    if np.max(binario) == 0 or np.sum(binario) < 5: return None
    try:
        verts, faces, _, _ = measure.marching_cubes(binario.astype(np.uint8), level=0.5)
        faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
        return pv.PolyData(verts, faces)
    except: return None

# =========================================================
# LAÇO DE PROCESSAMENTO DOS PACIENTES
# =========================================================
for idx_paciente, cod_paciente in enumerate(LISTA_PACIENTES):
    print("\n" + "="*70)
    print(f"PROCESSANDO PACIENTE [{idx_paciente + 1}/{len(LISTA_PACIENTES)}]: {cod_paciente}")
    print("="*70)
    
    # 1. LOCALIZAÇÃO DINÂMICA DO XML
    xml_exclusivo = buscar_xml_exclusivo(XML_ROOT, cod_paciente)
    if xml_exclusivo:
        print(f" XML Exclusivo Encontrado: {os.path.basename(xml_exclusivo)}")
    else:
        print(f" Alerta: Nenhum XML de 3 dígitos correspondente foi achado para {cod_paciente}.")
    
    pasta_paciente = os.path.join(ROOT, cod_paciente)
    serie_path = utils.encontrar_serie_dicom(pasta_paciente)
    if serie_path is None:
        print(f" Erro: Diretório DICOM válido não encontrado para {cod_paciente}. Pulando...")
        continue
        
    slices = utils.carregar_slices(serie_path)
    spacing_x = float(slices[0].PixelSpacing[1])
    spacing_y = float(slices[0].PixelSpacing[0])
    spacing_z = float(slices[0].SliceThickness)

    # Pré-processamento Exclusivo do Seu Método
    volume_hu = utils.converter_para_hu(slices)
    volume_window = utils.aplicar_window(volume_hu)
    volume_suave = gaussian_filter(volume_window, sigma=1)
    mapa_gradiente = calcular_magnitude_gradiente_3d(volume_suave)

    # Segmentação do Pulmão
    mascara_pulmao = np.logical_and(volume_suave > -1000, volume_suave < -400)
    mascara_pulmao = morphology.remove_small_objects(mascara_pulmao, min_size=1000)
    mascara_pulmao = ndimage.binary_fill_holes(mascara_pulmao)
    labels, num = ndimage.label(mascara_pulmao)
    mascara_pulmao = np.isin(labels, np.argsort(ndimage.sum(mascara_pulmao, labels, range(1, num + 1)))[-2:] + 1)

    vias_aereas = np.logical_and(volume_suave > -1050, volume_suave < -850) & mascara_pulmao
    vias_aereas = morphology.binary_opening(vias_aereas, morphology.ball(2))
    mascara_pulmao_limpa = np.logical_and(mascara_pulmao, ~vias_aereas)

    volume_roi = volume_suave.copy()
    volume_roi[~mascara_pulmao_limpa] = -1000

    # Candidatos Iniciais e Crescimento Condicional
    mascara_nodulos = np.logical_and(volume_roi > -250, volume_roi < 150)
    mascara_nodulos = morphology.binary_closing(mascara_nodulos, morphology.ball(1))
    mascara_nodulos = ndimage.binary_fill_holes(mascara_nodulos)
    mascara_nodulos = morphology.remove_small_objects(mascara_nodulos, min_size=15)

    candidatos_crescidos = crescimento_regioes_condicional(mascara_nodulos, mapa_gradiente)

    # Filtragem Geométrica Final da Sua Segmentação
    mascara_segmentacao = np.zeros_like(mascara_nodulos, dtype=bool)
    props = measure.regionprops(measure.label(candidatos_crescidos))

    for prop in props:
        if prop.area < MIN_VOLUME or prop.area > MAX_VOLUME: continue
        bbox = prop.bbox
        dz, dy, dx = bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]
        maior, menor = max(dx, dy, dz), min(dx, dy, dz)
        if menor == 0: continue
        
        if (maior / menor) > MAX_RAZAO or (menor / maior) < MIN_ESFERICIDADE: continue
        if dz > MAX_SLICES: continue
        if max(dx * spacing_x, dy * spacing_y, dz * spacing_z) > MAX_DIAMETRO_MM: continue
        
        mascara_segmentacao[measure.label(candidatos_crescidos) == prop.label] = True

    # =========================================================
    # EXTRAÇÃO DO XML DINÂMICO E LÓGICA DE CONJUNTOS 3D
    # =========================================================
    mascara_gold = extrair_padrao_ouro_puro(xml_exclusivo, mascara_segmentacao.shape)

    # Isola matematicamente as zonas exclusivas e de toque (Sem conflito de voxels)
    intersecao_laranja = np.logical_and(mascara_segmentacao, mascara_gold)
    algoritmo_vermelho_puro = np.logical_and(mascara_segmentacao, ~intersecao_laranja)
    gold_verde_puro = np.logical_and(mascara_gold, ~intersecao_laranja)

    # =========================================================
    # RENDERIZAÇÃO DE CENAS TRIDIMENSIONAIS (PYVISTA)
    # =========================================================
    print(f"Gerando cenário interativo para {cod_paciente}...")
    plotter = pv.Plotter()
    
    mesh_pulmao = criar_mesh(mascara_pulmao)
    mesh_vermelho = criar_mesh(algoritmo_vermelho_puro)
    mesh_verde = criar_mesh(gold_verde_puro)
    mesh_laranja = criar_mesh(intersecao_laranja)

    # Adiciona a anatomia pulmonar transparente ao fundo
    if not EXIBIR_APENAS_MASCARA and EXIBIR_PULMAO_OPACO and mesh_pulmao is not None:
        plotter.add_mesh(mesh_pulmao, color="lightblue", opacity=0.07, label="Parênquima Pulmonar")

    # Vermelho: O que apenas o seu algoritmo marcou
    if mesh_vermelho is not None:
        plotter.add_mesh(mesh_vermelho, color="red", opacity=0.85, label="Seu Método (Exclusivo)")

    # Verde: O que apenas o Padrão Ouro puro marcou
    if mesh_verde is not None:
        plotter.add_mesh(mesh_verde, color="green", opacity=0.85, label="Padrão Ouro (Exclusivo)")

    # Laranja: Região exata de toque/interseção entre ambos
    if mesh_laranja is not None:
        plotter.add_mesh(mesh_laranja, color="orange", opacity=1.0, show_edges=True, edge_color="darkorange", label="Interseção (Toque)")

    # Configuração e exibição do ambiente de visualização médico
    plotter.add_legend(bcolor='grey', border=True)
    plotter.add_axes()
    plotter.set_background("black")
    plotter.window_size = [1024, 768]
    
    print(f"Janela aberta para {cod_paciente}. Feche-a para avançar ao próximo paciente.")
    plotter.show()

print("\n[FIM] Todos os pacientes da lista foram processados com sucesso de forma dinâmica!")