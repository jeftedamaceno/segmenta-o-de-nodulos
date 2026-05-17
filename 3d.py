import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt

from skimage import measure
import pyvista as pv


# ==========================================
# ENCONTRAR SÉRIES AUTOMATICAMENTE
# ==========================================

ROOT = r"C:\LIDC"

series_encontradas = []

for raiz, dirs, arquivos in os.walk(ROOT):

    arquivos_dcm = [
        f for f in arquivos
        if f.endswith(".dcm")
    ]

    # séries válidas
    if len(arquivos_dcm) > 20:

        series_encontradas.append({
            "path": raiz,
            "quantidade": len(arquivos_dcm)
        })

print(f"\nTotal de séries encontradas: {len(series_encontradas)}")


# ==========================================
# ESCOLHER PACIENTE
# ==========================================

indice_paciente = 0

path = series_encontradas[indice_paciente]["path"]

print("\nPaciente selecionado:")
print(path)


# ==========================================
# CARREGAR DICOM
# ==========================================

slices = []

arquivos = [
    os.path.join(path, f)
    for f in os.listdir(path)
    if f.endswith(".dcm")
]

for arquivo in arquivos:

    try:

        ds = pydicom.dcmread(arquivo)

        if hasattr(ds, "pixel_array"):
            slices.append(ds)

    except Exception as e:
        print(f"Erro: {arquivo}")
        print(e)

print(f"\nSlices carregados: {len(slices)}")


# ==========================================
# ORDENAÇÃO SEGURA
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

    volume = np.stack([
        s.pixel_array for s in slices
    ]).astype(np.int16)

    # valores inválidos
    volume[volume == -2000] = 0

    for i in range(len(slices)):

        intercept = slices[i].RescaleIntercept
        slope = slices[i].RescaleSlope

        # aplicar slope
        if slope != 1:

            volume[i] = slope * volume[i].astype(np.float64)
            volume[i] = volume[i].astype(np.int16)

        # aplicar intercept
        volume[i] += np.int16(intercept)

    return volume


volume_hu = converter_para_hu(slices)

print("\nFormato do volume:", volume_hu.shape)

print("HU mínimo:", volume_hu.min())
print("HU máximo:", volume_hu.max())


# ==========================================
# WINDOWING (JANELA PULMONAR)
# ==========================================

# pulmão geralmente:
# centro = -600
# largura = 1500

window_center = -600
window_width = 1500

window_min = window_center - (window_width // 2)
window_max = window_center + (window_width // 2)

volume_window = np.clip(
    volume_hu,
    window_min,
    window_max
)

# normalização 0-1
volume_norm = volume_window.astype(np.float32)

volume_norm -= volume_norm.min()
volume_norm /= volume_norm.max()


# ==========================================
# VISUALIZADOR 2D
# ==========================================

indice = len(volume_norm) // 2

fig, ax = plt.subplots(figsize=(8,8))

img_plot = ax.imshow(
    volume_norm[indice],
    cmap='gray'
)

ax.set_title(f"Slice {indice}")
ax.axis('off')


def tecla(event):

    global indice

    if event.key == 'right':
        indice = min(indice + 1, len(volume_norm)-1)

    elif event.key == 'left':
        indice = max(indice - 1, 0)

    img_plot.set_data(volume_norm[indice])

    ax.set_title(f"Slice {indice}")

    fig.canvas.draw()


fig.canvas.mpl_connect(
    'key_press_event',
    tecla
)

plt.show()


# ==========================================
# SEGMENTAÇÃO USANDO HU
# ==========================================

"""
Pulmão:
aproximadamente entre -1000 e -400 HU

Aqui usamos HU real em vez
de threshold arbitrário.
"""

mascara = volume_hu < -400


# ==========================================
# MODELO 3D
# ==========================================

print("\nGerando modelo 3D...")

verts, faces, normals, values = measure.marching_cubes(
    mascara,
    level=0.5
)


# ==========================================
# CONVERTER FACES PARA PYVISTA
# ==========================================

faces_pv = np.hstack([
    np.full((faces.shape[0], 1), 3),
    faces
]).astype(np.int32)


mesh = pv.PolyData(
    verts,
    faces_pv
)


# ==========================================
# VISUALIZAÇÃO 3D
# ==========================================

plotter = pv.Plotter()

plotter.add_mesh(
    mesh,
    opacity=0.7
)

plotter.add_axes()

plotter.show_grid()

plotter.show()