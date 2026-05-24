import os
import glob
import numpy as np
import pydicom
import xml.etree.ElementTree as ET
from skimage.draw import polygon
from skimage.measure import marching_cubes
from scipy.ndimage import binary_fill_holes
import pyvista as pv

# =========================================================
# CONFIGURAÇÕES DE CAMINHOS
# =========================================================
ROOT_EXAMES = r"C:\exames_dos_pacientes"
XML_PATH = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\segmentação de nodulos\padrao ouro\tcia-lidc-xml\185\068.xml"
MASCARA_USUARIO_PATH = "mascara_final.npy"

# =========================================================
# FUNÇÃO CORRIGIDA PARA ENCONTRAR A PASTA REAL DO PACIENTE
# =========================================================
def encontrar_pasta_dicom_por_xml(xml_path, root_exames):
    print("📋 Lendo metadados estruturais do XML...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Solução robusta: Remove/ignora o namespace para encontrar a tag independente da versão do XML
    target_uid = None
    for elem in root.iter():
        if elem.tag.endswith('SeriesInstanceUid'):
            target_uid = elem.text.strip()
            break
            
    if not target_uid:
        raise ValueError("❌ Não foi possível encontrar a tag SeriesInstanceUid no XML fornecido.")
    
    print(f"🔍 UID Alvo extraído do XML: {target_uid}")
    print("📂 Buscando na sua pasta de exames... (Isso pode levar alguns segundos)")
    
    # Varre as pastas de exames procurando arquivos DICOM correspondentes
    for raiz, dirs, arquivos in os.walk(root_exames):
        dcm_files = [f for f in arquivos if f.endswith(".dcm")]
        if len(dcm_files) > 20:
            try:
                amostra_path = os.path.join(raiz, dcm_files[0])
                ds = pydicom.dcmread(amostra_path, stop_before_pixels=True)
                
                if hasattr(ds, "SeriesInstanceUID") and ds.SeriesInstanceUID == target_uid:
                    # Confirmação real de identidade extraída de dentro dos metadados do arquivo .dcm
                    paciente_id = getattr(ds, "PatientID", "Desconhecido")
                    print("\n" + "="*50)
                    print(f"✨ COMPATIBILIDADE CONFIRMADA!")
                    print(f"   • ID do Paciente no DICOM: {paciente_id}")
                    print(f"   • Pasta Física Encontrada: {raiz}")
                    print("="*50 + "\n")
                    return raiz
            except Exception as e:
                pass
                
    raise FileNotFoundError(f"❌ Nenhuma pasta de exames correspondente ao UID {target_uid} foi encontrada em: {root_exames}")

# =========================================================
# RECONSTRUÇÃO DO PADRÃO OURO E EXTRAÇÃO DAS MAIORES ROIs
# =========================================================
def carregar_volume_e_mapear_uids(pasta_dicom):
    arquivos = glob.glob(os.path.join(pasta_dicom, "**", "*.dcm"), recursive=True)
    slices = []
    for arq in arquivos:
        try:
            ds = pydicom.dcmread(arq)
            if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "SOPInstanceUID"):
                slices.append(ds)
        except:
            pass
            
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array for s in slices]).astype(np.int16)
    uid_to_index = {s.SOPInstanceUID: i for i, s in enumerate(slices)}
    return volume.shape, uid_to_index

def criar_mascara_padrao_ouro(xml_path, volume_shape, uid_to_index):
    mascara_gt = np.zeros(volume_shape, dtype=bool)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Mapeia todas as tags ignorando o prefixo do namespace
    for session in root:
        if session.tag.endswith('readingSession'):
            for nodule in session:
                if nodule.tag.endswith('unblindedReadNodule'):
                    for roi in nodule:
                        if roi.tag.endswith('roi'):
                            uid_node = None
                            x_coords, y_coords = [], []
                            
                            for child in roi:
                                if child.tag.endswith('imageSOP_UID'):
                                    uid_node = child.text.strip()
                                if child.tag.endswith('edgeMap'):
                                    for coord in child:
                                        if coord.tag.endswith('xCoord'):
                                            x_coords.append(int(coord.text))
                                        if coord.tag.endswith('yCoord'):
                                            y_coords.append(int(coord.text))
                                            
                            if uid_node in uid_to_index and len(x_coords) > 2:
                                slice_idx = uid_to_index[uid_node]
                                rr, cc = polygon(y_coords, x_coords, shape=(volume_shape[1], volume_shape[2]))
                                mascara_gt[slice_idx, rr, cc] = True
                                
    for i in range(mascara_gt.shape[0]):
        if np.any(mascara_gt[i]):
            mascara_gt[i] = binary_fill_holes(mascara_gt[i])
    return mascara_gt

# =========================================================
# VALIDAÇÃO MATEMÁTICA E RENDERIZAÇÃO
# =========================================================
def calcular_metricas(mask_gt, mask_user):
    intersection = np.logical_and(mask_gt, mask_user).sum()
    total_pixels = mask_gt.sum() + mask_user.sum()
    union = np.logical_or(mask_gt, mask_user).sum()
    dice = (2.0 * intersection) / total_pixels if total_pixels > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return dice, iou

def criar_mesh(binario):
    if np.max(binario) == 0 or np.sum(binario) < 5:
        return None
    try:
        verts, faces, normals, values = marching_cubes(binario.astype(np.uint8), level=0.5)
        faces = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
        return pv.PolyData(verts, faces)
    except:
        return None

# Execution
try:
    pasta_paciente_real = encontrar_pasta_dicom_por_xml(XML_PATH, ROOT_EXAMES)
    shape_volume, dicionario_uids = carregar_volume_e_mapear_uids(pasta_paciente_real)
    
    print("🟢 Reconstruindo Padrão Ouro (Verde)...")
    mascara_gt = criar_mascara_padrao_ouro(XML_PATH, shape_volume, dicionario_uids)
    
    print("🔴 Carregando sua segmentação (Vermelho)...")
    if not os.path.exists(MASCARA_USUARIO_PATH):
        raise FileNotFoundError(f"❌ Arquivo '{MASCARA_USUARIO_PATH}' não encontrado.")
    mascara_usuario = np.load(MASCARA_USUARIO_PATH).astype(bool)
    
    if mascara_usuario.shape != mascara_gt.shape:
        min_z = min(mascara_usuario.shape[0], mascara_gt.shape[0])
        mascara_usuario = mascara_usuario[:min_z, :, :]
        mascara_gt = mascara_gt[:min_z, :, :]

    dice, iou = calcular_metricas(mascara_gt, mascara_usuario)
    print(f"\n📊 ACURÁCIA ALCANCE:")
    print(f"   • Dice Score: {dice:.4f}")
    print(f"   • IoU Score:  {iou:.4f}")
    
    mesh_gt = criar_mesh(mascara_gt)
    mesh_pred = criar_mesh(mascara_usuario)
    
    plotter = pv.Plotter()
    if mesh_gt is not None:
        plotter.add_mesh(mesh_gt, color="green", opacity=0.5, label="Padrao Ouro (Radiologistas)")
    if mesh_pred is not None:
        plotter.add_mesh(mesh_pred, color="red", opacity=0.6, label="Sua Segmentacao")
        
    plotter.add_legend(bcolor='grey', border=True)
    plotter.set_background("black")
    plotter.show()

except Exception as e:
    print(f"\n❗ Falha: {e}")