import os
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, sobel
from skimage import morphology, measure
from collections import deque
import pyvista as pv

import utils

ROOT = r"C:\exames_dos_pacientes"
# PACIENTE = "LIDC-IDRI-0068"
PACIENTE = "LIDC-IDRI-0164"
EXIBIR_PULMAO_OPACO = True   
EXIBIR_APENAS_MASCARA = False  

MIN_VOLUME = 20          
MAX_VOLUME = 4000
MIN_ESFERICIDADE = 0.1   
MAX_RAZAO = 9.5          
MAX_SLICES = 40
MAX_DIAMETRO_MM = 35

# Parâmetros de Crescimento Condicional Otimizados (Sem Hessiana)
RAIO_MIN_TUNEL_VOXELS = 1.6       

MAX_SLICES_TUNEL = 42             

MIN_GRADIENTE_CONTINUIDADE = 4.0 




def calcular_magnitude_gradiente_3d(volume):
    print("[Sobel] Mapeando variações abruptas de borda de forma otimizada...")
    s_z = sobel(volume, axis=0)
    s_y = sobel(volume, axis=1)
    s_x = sobel(volume, axis=2)
    return np.sqrt(s_z**2 + s_y**2 + s_x**2)

# CRESCIMENTO DE REGIÕES CONDICIONAL RESTRITO
def crescimento_regioes_condicional(mascara_nodulos, mapa_gradiente):
    mapa_dist = ndimage.distance_transform_edt(mascara_nodulos)
    from skimage.feature import peak_local_max
    coords_sementes = peak_local_max(mapa_dist, footprint=np.ones((3, 3, 3)), labels=mascara_nodulos)
    
    mascara_crescida = np.zeros_like(mascara_nodulos, dtype=bool)
    visitados_global = np.zeros_like(mascara_nodulos, dtype=bool)
    
    # Conectividade-26
    vizinhos_3d = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0: continue
                vizinhos_3d.append((dz, dy, dx))
                
    print(f" Processando crescimento veloz para {len(coords_sementes)} sementes...")
    
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
                
            # --- CRITÉRIOS DE INTERRUPÇÃO CIENTÍFICOS REMANESCENTES ---
            # 1. Critério de Artigo (Gradiente): Se o gradiente for plano na borda, indica vaso homogêneo
            if mapa_gradiente[cz, cy, cx] < MIN_GRADIENTE_CONTINUIDADE and mapa_dist[cz, cy, cx] < 1.8:
                continue
                
            # 2. Sua Estratégia (Túnel): Crescimento esticado e fino ao longo de fatias Z
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

pasta_paciente = os.path.join(ROOT, PACIENTE)
serie_path = utils.encontrar_serie_dicom(pasta_paciente)
slices = utils.carregar_slices(serie_path)

spacing_x, spacing_y, spacing_z = float(slices[0].PixelSpacing[1]), float(slices[0].PixelSpacing[0]), float(slices[0].SliceThickness)

volume_hu = utils.converter_para_hu(slices)
volume_window = utils.aplicar_window(volume_hu)
volume_suave = gaussian_filter(volume_window, sigma=1)

# Processa apenas o mapa de gradiente (convolução paralela rápida)
mapa_gradiente = calcular_magnitude_gradiente_3d(volume_suave)

# Segmentação de Pulmão
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

# Candidatos iniciais
mascara_nodulos = np.logical_and(volume_roi > -250, volume_roi < 150)
mascara_nodulos = morphology.binary_closing(mascara_nodulos, morphology.ball(1))
mascara_nodulos = ndimage.binary_fill_holes(mascara_nodulos)
mascara_nodulos = morphology.remove_small_objects(mascara_nodulos, min_size=15)

# Execução do crescimento condicional híbrido otimizado
candidatos_crescidos = crescimento_regioes_condicional(mascara_nodulos, mapa_gradiente)

# Filtragem Geométrica Final 
mascara_final = np.zeros_like(mascara_nodulos, dtype=bool)
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
    
    mascara_final[measure.label(candidatos_crescidos) == prop.label] = True

print("\nAbrindo ambiente PyVista...")
plotter = pv.Plotter()

mesh_nodulos = criar_mesh(mascara_final)

if not EXIBIR_APENAS_MASCARA and EXIBIR_PULMAO_OPACO:
    mesh_pulmao = criar_mesh(mascara_pulmao)
    if mesh_pulmao is not None:
        plotter.add_mesh(mesh_pulmao, color="lightblue", opacity=0.12, label="Volume Pulmonar")

if mesh_nodulos is not None:
    plotter.add_mesh(mesh_nodulos, color="red", opacity=0.9, label="Nódulos Detectados")
else:
    print("Nenhuma estrutura resistiu com as configurações atuais.")

plotter.add_legend(bcolor='grey', border=True)
plotter.set_background("black")
plotter.show()

print("\n" + "="*60)
opcao = input("❓ Deseja salvar este resultado em 'mascara_test v3.npy'? (s/n): ").strip().lower()
print("="*60)

if opcao == 's':
    np.save("mascara_test v3.npy", mascara_final)
    print("[SUCESSO] 'mascara_test v3.npy' gerada e gravada com sucesso!")
else:
    print("[CANCELADO] Processamento finalizado sem sobrescrever o arquivo.")