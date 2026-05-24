import os
import numpy as np
from scipy.ndimage import (
    gaussian_filter, 
    binary_fill_holes, 
    label, 
    sobel
)
from skimage import measure
from skimage.morphology import binary_closing, binary_opening, ball, remove_small_objects
from collections import deque
import pyvista as pv

import utils

# =========================================================
# CONFIGURAÇÕES DE INTERAÇÃO E VISUALIZAÇÃO
# =========================================================
PACIENTE_PATH = r"C:\exames_dos_pacientes\LIDC-IDRI-0068"

EXIBIR_PULMAO_OPACO = True    # Se True, adiciona o pulmão translúcido ao fundo
EXIBIR_APENAS_MASCARA = False  # Se True, foca estritamente nos nódulos gerados

# =========================================================
# PARÂMETROS DAS METODOLOGIAS CIENTÍFICAS (ALTERE PARA TESTAR)
# =========================================================
# [Abordagem A] Limiar de Gradiente Mínimo para continuar crescendo
# Se a variação de densidade cair abaixo disso ao longo de uma estrutura, assume-se que entrou num tubo homogêneo (vaso)
MIN_GRADIENTE_CONTINUIDADE = 8.0  

# [Abordagem B] Sensibilidade de Vesselness (Matriz Hessiana)
# Valores menores (ex: 0.2) bloqueiam mais vasos. Valores maiores (ex: 0.6) são mais tolerantes
LIMIAR_VESSELNESS_BLOQUEIO = 0.45  

MIN_VOLUME_CANDIDATO = 15     
MAX_RAZAO_GEOMETRICA = 5.0    

# =========================================================
# CÁLCULOS AVANÇADOS DE GEOMETRIA (HESSIANA E GRADIENTE)
# =========================================================
def calcular_vesselness_3d(volume, sigma=1.0):
    """
    Abordagem B: Calcula a probabilidade de cada voxel pertencer a uma estrutura 
    tubular (vaso/via aérea) usando os autovalores da Matriz Hessiana 3D.
    """
    print("🧮 Calculando Matriz Hessiana 3D para filtragem de formas tubulares...")
    vol_suave = gaussian_filter(volume, sigma=sigma)
    
    # Gradientes de segunda ordem (Derivadas parciais)
    dz, dy, dx = np.gradient(vol_suave)
    dzz, dzy, dzx = np.gradient(dz)
    _, dyy, dyx = np.gradient(dy)
    _, _, dxx = np.gradient(dx)
    
    vesselness = np.zeros_like(volume, dtype=float)
    
    # Para cada voxel (processado em fatias para otimizar memória)
    for z in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            # Filtro rápido onde há estruturas candidatas (evita processar fundo preto)
            valid_x = np.where((volume[z, y] > -300) & (volume[z, y] < 300))[0]
            for x in valid_x:
                # Monta a matriz Hessiana local 3x3
                H = np.array([
                    [dzz[z,y,x], dzy[z,y,x], dzx[z,y,x]],
                    [dzy[z,y,x], dyy[z,y,x], dyx[z,y,x]],
                    [dzx[z,y,x], dyx[z,y,x], dxx[z,y,x]]
                ])
                
                # Calcula e ordena os autovalores por magnitude: |L1| <= |L2| <= |L3|
                autovalores = np.linalg.eigvalsh(H)
                idx = np.argsort(np.abs(autovalores))
                l1, l2, l3 = autovalores[idx]
                
                # Critério matemático de Frangi para Tubos em imagens médicas (3D):
                # Para um tubo brilhante: L1 próximo de 0, L2 e L3 muito negativos
                if l2 >= 0 or l3 >= 0:
                    continue
                    
                Rb = np.abs(l1) / (np.sqrt(np.abs(l2 * l3)) + 1e-5) # Indicador de formato de placa
                Ra = np.abs(l2) / (np.abs(l3) + 1e-5)               # Indicador de tubo vs esfera
                S = np.sqrt(l1**2 + l2**2 + l3**2)                  # Intensidade/Contraste estrutural
                
                # Equação de Vesselness simplificada
                c = 50.0  # Fator de escala do desvio
                vessel = (1.0 - np.exp(-(Ra**2) / 0.5)) * np.exp(-(Rb**2) / 0.5) * (1.0 - np.exp(-(S**2) / (2 * c**2)))
                vesselness[z, y, x] = vessel
                
    return vesselness

def calcular_magnitude_gradiente_3d(volume):
    """
    Abordagem A: Calcula a magnitude do gradiente tridimensional usando Sobel
    """
    print("📈 Calculando mapas de Gradiente de Borda 3D (Sobel)...")
    s_z = sobel(volume, axis=0)
    s_y = sobel(volume, axis=1)
    s_x = sobel(volume, axis=2)
    return np.sqrt(s_z**2 + s_y**2 + s_x**2)

# =========================================================
# PIPELINE DE SEGMENTAÇÃO
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
    return remove_small_objects(binary_fill_holes(candidatos), min_size=MIN_VOLUME_CANDIDATO)

# =========================================================
# CRESCIMENTO DE REGIÕES CIENTÍFICO (CONDICIONAL AVANÇADO)
# =========================================================
def crescimento_regioes_cientifico(mascara_nodulos, mapa_gradiente, mapa_vesselness):
    # Usamos os picos de distância como sementes centrais estáveis
    from scipy.ndimage import distance_transform_edt
    from skimage.feature import peak_local_max
    
    mapa_dist = distance_transform_edt(mascara_nodulos)
    coords_sementes = peak_local_max(mapa_dist, footprint=np.ones((3, 3, 3)), labels=mascara_nodulos)
    
    mascara_crescida = np.zeros_like(mascara_nodulos, dtype=bool)
    visitados_global = np.zeros_like(mascara_nodulos, dtype=bool)
    
    vizinhos_3d = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0: continue
                vizinhos_3d.append((dz, dy, dx))
                
    print(f"🔬 Analisando {len(coords_sementes)} sementes com restrições de Borda e Tubularidade...")
    
    for semente in coords_sementes:
        sz, sy, sx = semente
        if visitados_global[sz, sy, sx]: continue
            
        fila = deque([(sz, sy, sx)])
        pixels_da_regiao = []
        
        while fila:
            cz, cy, cx = fila.popleft()
            if visitados_global[cz, cy, cx]: continue
                
            # --- VERIFICAÇÃO DAS CONDIÇÕES DE ARTIGOS ---
            # 1. Condição B (Hessiana): Bloqueia se o voxel atual possuir forte característica estrutural de duto/tubo
            if mapa_vesselness[cz, cy, cx] > LIMIAR_VESSELNESS_BLOQUEIO:
                continue # Não adiciona este voxel e não expande a partir dele
                
            # 2. Condição A (Gradiente): Se o gradiente for extremamente plano/baixo em região de avanço periférico,
            # significa continuidade homogênea cilíndrica (vessel filling)
            if mapa_gradiente[cz, cy, cx] < MIN_GRADIENTE_CONTINUIDADE and mapa_dist[cz, cy, cx] < 2.0:
                continue
                
            visitados_global[cz, cy, cx] = True
            pixels_da_regiao.append((cz, cy, cx))
            
            # Expansão espacial
            for dz, dy, dx in vizinhos_3d:
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if 0 <= nz < mascara_nodulos.shape[0] and 0 <= ny < mascara_nodulos.shape[1] and 0 <= nx < mascara_nodulos.shape[2]:
                    if mascara_nodulos[nz, ny, nx] and not visitados_global[nz, ny, nx]:
                        fila.append((nz, ny, nx))
                        
        if len(pixels_da_regiao) >= MIN_VOLUME_CANDIDATO:
            for cz, cy, cx in pixels_da_regiao:
                mascara_crescida[cz, cy, cx] = True
                
    return mascara_crescida

def filtrar_nodulos(binario):
    labels_pos, num_pos = label(binario)
    props = measure.regionprops(labels_pos)
    final = np.zeros_like(binario)
    
    for prop in props:
        if prop.area < 20 or prop.area > 4500: continue
        bbox = prop.bbox
        dz, dy, dx = bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]
        
        maior = max(dx, dy, dz)
        menor = max(1, min(dx, dy, dz))
        
        if (maior / menor) > MAX_RAZAO_GEOMETRICA: continue
        final[labels_pos == prop.label] = 1
    return final.astype(np.uint8)

def criar_mesh(binario):
    if np.max(binario) == 0 or np.sum(binario) < 5: return None
    try:
        verts, faces, _, _ = measure.marching_cubes(binario.astype(np.uint8), level=0.5)
        faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
        return pv.PolyData(verts, faces)
    except: return None

# =========================================================
# EXECUÇÃO DO FLUXO PRINCIPAL COM ABORDAGENS DA LITERATURA
# =========================================================
print("\n[1/6] Carregando volume DICOM do paciente...")
volume, slices = utils.carregar_dicom_recursivo(PACIENTE_PATH)
hu = utils.converter_para_hu(slices)
hu_windowed = utils.aplicar_window(hu)

print("[2/6] Processando mapas matemáticos auxiliares (A & B)...")
mapa_gradiente = calcular_magnitude_gradiente_3d(hu_windowed)
mapa_vesselness = calcular_vesselness_3d(hu_windowed, sigma=1.0)

print("[3/6] Segmentando região interna do pulmão...")
mascara_pulmao = segmentar_pulmao(hu_windowed)

print("[4/6] Isolando candidatos iniciais de densidade...")
candidatos = detectar_nodulos(hu_windowed, mascara_pulmao)

print("[5/6] Executando Crescimento de Regiões Condicional Científico...")
candidatos_filtrados = crescimento_regioes_cientifico(candidatos, mapa_gradiente, mapa_vesselness)

print("[6/6] Filtrando geometria tridimensional residual...")
mascara_final = filtrar_nodulos(candidatos_filtrados)

# =========================================================
# RENDERIZAÇÃO E INTERAÇÃO
# =========================================================
print("\n📦 Renderizando ambiente de análise espacial 3D...")
plotter = pv.Plotter()

mesh_nodulos = criar_mesh(mascara_final)

if not EXIBIR_APENAS_MASCARA and EXIBIR_PULMAO_OPACO:
    mesh_pulmao = criar_mesh(mascara_pulmao)
    if mesh_pulmao is not None:
        plotter.add_mesh(mesh_pulmao, color="lightblue", opacity=0.12, label="Parênquima Pulmonar")

if mesh_nodulos is not None:
    plotter.add_mesh(mesh_nodulos, color="red", opacity=0.9, label="Nódulos (Hessiana + Sobel)")
else:
    print("⚠️ Nenhuma estrutura resistiu aos filtros matemáticos aplicados.")

plotter.add_legend(bcolor='grey', border=True)
plotter.set_background("black")

print("🖥️ Janela de exibição aberta. Avalie a eficácia do filtro científico e feche para decidir.")
plotter.show()

# Decisão de salvamento
print("\n" + "="*60)
opcao = input("❓ Deseja salvar esta versão refinada cientificamente? (s/n): ").strip().lower()
print("="*60)

if opcao == 's':
    np.save("mascara_final.npy", mascara_final)
    print("💾 [SUCESSO] 'mascara_final.npy' gravada com os novos filtros de artigos!")
else:
    print("❌ [CANCELADO] Modificações descartadas. Altere as sensibilidades e teste novamente.")