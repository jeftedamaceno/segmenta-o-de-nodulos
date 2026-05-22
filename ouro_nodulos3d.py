import os
import re
import numpy as np
import pydicom
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from skimage.draw import polygon
from skimage import measure
import pyvista as pv

# ==========================================
# CONFIGURAÇÕES
# ==========================================

PACIENTE_PATH = r"C:\exames_dos_pacientes\LIDC-IDRI-0068"

PASTA_XML = r"C:\Users\jefte\projetos em python\ufc 2025 a 2026\segmentação de nodulos\padrao ouro\tcia-lidc-xml\189"

# ==========================================
# EXTRAIR ID DO PACIENTE
# ==========================================

nome_paciente = os.path.basename(PACIENTE_PATH)

match = re.search(r'(\d+)', nome_paciente)

if match is None:
    raise Exception("Não foi possível extrair ID do paciente")

id_paciente = match.group(1)

id_xml = id_paciente[-3:]

print("\nPaciente:", nome_paciente)
print("ID XML:", id_xml)

# ==========================================
# ENCONTRAR XML
# ==========================================

xml_path = None

for raiz, dirs, arquivos in os.walk(PASTA_XML):

    for arq in arquivos:

        if arq.lower() == f"{id_xml}.xml":

            xml_path = os.path.join(raiz, arq)
            break

    if xml_path:
        break

print("\nXML encontrado:")
print(xml_path)

if xml_path is None:
    raise Exception("XML NÃO ENCONTRADO")

# ==========================================
# ENCONTRAR DICOMS
# ==========================================

arquivos_dcm = []

for raiz, dirs, arquivos in os.walk(PACIENTE_PATH):

    for arq in arquivos:

        if arq.endswith(".dcm"):

            arquivos_dcm.append(
                os.path.join(raiz, arq)
            )

print(f"\nDICOM encontrados: {len(arquivos_dcm)}")

# ==========================================
# CARREGAR SLICES
# ==========================================

slices = []

for arquivo in arquivos_dcm:

    try:

        ds = pydicom.dcmread(arquivo)

        if hasattr(ds, "pixel_array"):

            shape = ds.pixel_array.shape

            if len(shape) == 2 and shape == (512, 512):

                slices.append(ds)

    except:
        pass

print(f"Slices válidos: {len(slices)}")

# ==========================================
# ORDENAÇÃO
# ==========================================

def chave(ds):

    if hasattr(ds, "ImagePositionPatient"):

        return float(ds.ImagePositionPatient[2])

    elif hasattr(ds, "SliceLocation"):

        return float(ds.SliceLocation)

    elif hasattr(ds, "InstanceNumber"):

        return int(ds.InstanceNumber)

    return 0

slices.sort(key=chave)

# ==========================================
# CONVERTER PARA HU
# ==========================================

def converter_para_hu(slices):

    imagens = []

    for s in slices:

        try:
            imagens.append(s.pixel_array)

        except:
            pass

    volume = np.stack(imagens).astype(np.int16)

    volume[volume == -2000] = 0

    for i in range(len(slices)):

        intercept = slices[i].RescaleIntercept
        slope = slices[i].RescaleSlope

        if slope != 1:

            volume[i] = slope * volume[i].astype(np.float64)
            volume[i] = volume[i].astype(np.int16)

        volume[i] += np.int16(intercept)

    return volume

print("\nConvertendo HU...")

volume = converter_para_hu(slices)

print("Volume:", volume.shape)

# ==========================================
# MÁSCARA 3D
# ==========================================

mascara = np.zeros(volume.shape, dtype=np.uint8)

# ==========================================
# LER XML
# ==========================================

print("\nLendo XML...")

tree = ET.parse(xml_path)

root = tree.getroot()

namespace = {
    'ns': 'http://www.nih.gov'
}

nodulos = root.findall('.//ns:unblindedReadNodule', namespace)

print(f"Nódulos encontrados: {len(nodulos)}")

# ==========================================
# MAPEAR NÓDULOS NOS SLICES
# ==========================================

for nodulo in nodulos:

    roi_list = nodulo.findall('.//ns:roi', namespace)

    for roi in roi_list:

        z_element = roi.find('ns:imageZposition', namespace)

        if z_element is None:
            continue

        z_xml = float(z_element.text)

        indice_slice = None
        menor_erro = 999999

        for i, s in enumerate(slices):

            try:

                z_slice = float(
                    s.ImagePositionPatient[2]
                )

                erro = abs(z_slice - z_xml)

                if erro < menor_erro:

                    menor_erro = erro
                    indice_slice = i

            except:
                pass

        if indice_slice is None:
            continue

        pontos_x = []
        pontos_y = []

        edges = roi.findall('.//ns:edgeMap', namespace)

        for edge in edges:

            x = edge.find('ns:xCoord', namespace)
            y = edge.find('ns:yCoord', namespace)

            if x is not None and y is not None:

                pontos_x.append(int(x.text))
                pontos_y.append(int(y.text))

        if len(pontos_x) > 2:

            rr, cc = polygon(
                pontos_y,
                pontos_x,
                shape=mascara[indice_slice].shape
            )

            mascara[indice_slice, rr, cc] = 1

# ==========================================
# VISUALIZAÇÃO 2D
# ==========================================

slice_index = len(volume) // 2

fig, ax = plt.subplots(figsize=(8,8))

overlay = np.ma.masked_where(
    mascara[slice_index] == 0,
    mascara[slice_index]
)

ax.imshow(volume[slice_index], cmap='gray')
ax.imshow(overlay, cmap='autumn', alpha=0.7)

ax.set_title(f"Slice {slice_index}")
ax.axis('off')

plt.show()

# ==========================================
# MODELO 3D DOS NÓDULOS
# ==========================================

print("\nGerando modelo 3D dos nódulos...")

if mascara.max() == 0:

    print("Nenhum voxel encontrado.")

else:

    verts, faces, normals, values = measure.marching_cubes(
        mascara,
        level=0.5
    )

    faces_pv = np.hstack([
        np.full((faces.shape[0], 1), 3),
        faces
    ]).astype(np.int32)

    mesh = pv.PolyData(
        verts,
        faces_pv
    )

    plotter = pv.Plotter()

    plotter.add_mesh(
        mesh,
        opacity=1.0
    )

    plotter.add_axes()

    plotter.show_grid()

    plotter.show()