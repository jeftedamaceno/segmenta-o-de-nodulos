# import os
# import numpy as np
# from scipy.ndimage import gaussian_filter, binary_fill_holes
# from skimage import measure
# from skimage.morphology import binary_closing, binary_opening, ball, remove_small_objects

# import utils

# PACIENTE_PATH = r"C:\exames_dos_pacientes\LIDC-IDRI-0068"

# def segmentar_pulmao(volume):
#     mascara = volume < -400
#     return binary_fill_holes(binary_closing(mascara, ball(2)))

# def detectar_nodulos(volume, mascara_pulmao):
#     candidatos = np.logical_and(volume > -300, volume < 300)
#     candidatos = np.logical_and(candidatos, mascara_pulmao)
#     candidatos = gaussian_filter(candidatos.astype(float), sigma=1) > 0.2
#     candidatos = binary_opening(candidatos, ball(1))
#     candidatos = binary_closing(candidatos, ball(1))
#     return remove_small_objects(binary_fill_holes(candidatos), min_size=20)

# def filtrar_nodulos(binario):
#     labels = measure.label(binario)
#     props = measure.regionprops(labels)
#     final = np.zeros_like(binario)
#     for prop in props:
#         if prop.area < 30 or prop.area > 4000:
#             continue
#         bbox = prop.bbox
#         dz, dy, dx = bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]
#         if (max(dx, dy, dz) / max(1, min(dx, dy, dz))) > 4:
#             continue
#         final[labels == prop.label] = 1
#     return final.astype(np.uint8)

# # Execução limpa do fluxo principal
# print("\nCarregando DICOMs...")
# volume, slices = utils.carregar_dicom_recursivo(PACIENTE_PATH)

# print("Convertendo e filtrando em HU...")
# hu = utils.converter_para_hu(slices)
# hu_windowed = utils.aplicar_window(hu)

# mascara_pulmao = segmentar_pulmao(hu_windowed)
# candidatos = detectar_nodulos(hu_windowed, mascara_pulmao)
# mascara_final = filtrar_nodulos(candidatos)

# np.save("mascara_beta v4.npy", mascara_final)
# print("Arquivo mascara_final.npy salvo com sucesso!")

import os
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, distance_transform_edt
from skimage import measure
from skimage.morphology import binary_closing, binary_opening, ball, remove_small_objects
from collections import deque
import pyvista as pv

import utils

# =========================================================
# CONFIGURAÇÕES DE INTERAÇÃO E VISUALIZAÇÃO
# =========================================================
PACIENTE_PATH = r"C:\exames_dos_pacientes\LIDC-IDRI-0088"

EXIBIR_PULMAO_OPACO = True    # Se True, adiciona o pulmão translúcido de fundo
EXIBIR_APENAS_MASCARA = False  # Se True, foca estritamente nos nódulos gerados

# =========================================================
# SEU ESPAÇO DE TESTE DE PARÂMETROS (TÚNEL E GEOMETRIA)
# =========================================================
# Parâmetros de Crescimento por Túnel (Sua estratégia contra veias longas)
RAIO_MIN_TUNEL_VOXELS = 2.8   # Se o raio local for menor que isso, considera canal fino
MAX_SLICES_TUNEL = 20         # Quantas fatias Z o túnel pode persistir antes de ser cortado

# Filtros Geométricos Finais
MIN_VOLUME_FINAL = 10
MAX_VOLUME_FINAL = 2000
MAX_RAZAO_GEOMETRICA = 4    # Razão máxima entre o maior e menor eixo (comprimento / largura)

# =========================================================
# FUNÇÕES DO PIPELINE
# =========================================================
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

# =========================================================
# ESTRATÉGIA: CRESCIMENTO DE REGIÕES CONDICIONAL POR TÚNEL
# =========================================================
def crescimento_regioes_por_tunel(mascara_candidatos):
    """
    Executa o crescimento de regiões a partir dos centros dos objetos (sementes).
    Bloqueia e descarta o crescimento caso ele caminhe por muitas fatias Z
    mantendo uma espessura fina (perfil característico de veias longas).
    """
    mapa_distancia = distance_transform_edt(mascara_candidatos)
    from skimage.feature import peak_local_max
    coords_sementes = peak_local_max(mapa_distancia, footprint=np.ones((3, 3, 3)), labels=mascara_candidatos)
    
    mascara_crescida = np.zeros_like(mascara_candidatos, dtype=bool)
    visitados_global = np.zeros_like(mascara_candidatos, dtype=bool)
    
    # Vizinhança 3D (26 conectividades)
    vizinhos_3d = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0: continue
                vizinhos_3d.append((dz, dy, dx))
                
    print(f"🔬 Analisando {len(coords_sementes)} sementes com filtro de persistência de túnel...")
    
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
                
            visitados_global[cz, cy, cx] = True
            pixels_da_regiao.append((cz, cy, cx))
            
            raio_local = mapa_distancia[cz, cy, cx]
            distancia_fatias_z = abs(cz - z_inicial)
            
            # Condicional de Túnel: se esticou no eixo Z além do limite sendo estreito, aborta
            if distancia_fatias_z >= MAX_SLICES_TUNEL and raio_local <= RAIO_MIN_TUNEL_VOXELS:
                abortar_crescimento = True
                break
                
            # Expansão para os vizinhos válidos
            for dz, dy, dx in vizinhos_3d:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if 0 <= nz < mascara_candidatos.shape[0] and 0 <= ny < mascara_candidatos.shape[1] and 0 <= nx < mascara_candidatos.shape[2]:
                    if mascara_candidatos[nz, ny, nx] and not visitados_global[nz, ny, nx]:
                        fila.append((nz, ny, nx))
                        
        # Se a estrutura não se comportou como um túnel longo de veia, ela é adicionada à máscara
        if not abortar_crescimento and len(pixels_da_regiao) >= MIN_VOLUME_FINAL:
            for cz, cy, cx in pixels_da_regiao:
                mascara_crescida[cz, cy, cx] = True
                
    return mascara_crescida

def filtrar_nodulos(binario):
    labels = measure.label(binario)
    props = measure.regionprops(labels)
    final = np.zeros_like(binario)
    
    for prop in props:
        if prop.area < MIN_VOLUME_FINAL or prop.area > MAX_VOLUME_FINAL:
            continue
        bbox = prop.bbox
        dz, dy, dx = bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]
        
        # Filtro de razão para eliminar veias isoladas remanescentes que sejam muito compridas
        if (max(dx, dy, dz) / max(1, min(dx, dy, dz))) > MAX_RAZAO_GEOMETRICA:
            continue
            
        final[labels == prop.label] = 1
    return final.astype(np.uint8)

def criar_mesh(binario):
    if np.max(binario) == 0 or np.sum(binario) < 5: return None
    try:
        verts, faces, _, _ = measure.marching_cubes(binario.astype(np.uint8), level=0.5)
        faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
        return pv.PolyData(verts, faces)
    except: return None

# =========================================================
# EXECUÇÃO DO FLUXO PRINCIPAL
# =========================================================
print("\n[1/5] Carregando DICOMs...")
volume, slices = utils.carregar_dicom_recursivo(PACIENTE_PATH)

print("[2/5] Convertendo e filtrando em HU...")
hu = utils.converter_para_hu(slices)
hu_windowed = utils.aplicar_window(hu)

print("[3/5] Segmentando pulmão e isolando candidatos iniciais...")
mascara_pulmao = segmentar_pulmao(hu_windowed)
candidatos = detectar_nodulos(hu_windowed, mascara_pulmao)

print("[4/5] Aplicando crescimento condicional com restrição de túnel...")
candidatos_sem_veias = crescimento_regioes_por_tunel(candidatos)

print("[5/5] Lapidando filtros geométricos finais...")
mascara_final = filtrar_nodulos(candidatos_sem_veias)

# =========================================================
# VISUALIZAÇÃO INTERATIVA COM PYVISTA
# =========================================================
print("\n📦 Construindo cenário 3D para avaliação de parâmetros...")
plotter = pv.Plotter()

mesh_nodulos = criar_mesh(mascara_final)

if not EXIBIR_APENAS_MASCARA and EXIBIR_PULMAO_OPACO:
    mesh_pulmao = criar_mesh(mascara_pulmao)
    if mesh_pulmao is not None:
        plotter.add_mesh(mesh_pulmao, color="lightblue", opacity=0.15, label="Estrutura Pulmonar")

if mesh_nodulos is not None:
    plotter.add_mesh(mesh_nodulos, color="red", opacity=0.9, label="Nódulos (Filtro por Túnel)")
else:
    print("⚠️ Nenhuma estrutura sobreviveu aos parâmetros atuais.")

plotter.add_legend(bcolor='grey', border=True)
plotter.set_background("black")

print("🖥️ Janela gráfica aberta. Inspecione o modelo e FECHE a janela para decidir se quer salvar.")
plotter.show()

# =========================================================
# SISTEMA DE INTERAÇÃO DE SALVAMENTO
# =========================================================
print("\n" + "="*60)
opcao = input("❓ Gostou do resultado gerado por estes parâmetros? Salvar arquivo? (s/n): ").strip().lower()
print("="*60)

if opcao == 's':
    np.save("mascara_beta v4.npy", mascara_final)
    print("💾 [SALVO] 'mascara_beta v4.npy' gravado com sucesso! Pronto para métricas.")
else:
    print("❌ [CANCELADO] Modificações descartadas. Altere os valores no painel de controle e reexecute.")