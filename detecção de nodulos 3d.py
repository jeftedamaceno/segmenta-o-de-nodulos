import os
import numpy as np
import pyvista as pv
from scipy import ndimage
from skimage import morphology
import matplotlib.pyplot as plt

# Importa as utilidades compartilhadas
import utils

ROOT = r"C:\exames_dos_pacientes"
PACIENTE = "LIDC-IDRI-0088"

# Pipeline simplificado
pasta_paciente = os.path.join(ROOT, PACIENTE)
serie_path = utils.encontrar_serie_dicom(pasta_paciente)
print(f"Série encontrada: {serie_path}")

slices = utils.carregar_slices(serie_path)
volume_hu = utils.converter_para_hu(slices)
print(f"Volume HU: {volume_hu.shape}")

# Segmentação baseada na versão original
mascara_pulmao = np.logical_and(volume_hu > -1000, volume_hu < -400)
mascara_pulmao = morphology.remove_small_objects(mascara_pulmao, min_size=1000)
mascara_pulmao = ndimage.binary_fill_holes(mascara_pulmao)

labels, num = ndimage.label(mascara_pulmao)
sizes = ndimage.sum(mascara_pulmao, labels, range(1, num + 1))
indices = np.argsort(sizes)[-2:] + 1
mascara_pulmao = morphology.binary_erosion(np.isin(labels, indices), morphology.ball(3))

mascara_nodulos = np.logical_and(volume_hu > -300, volume_hu < 100)
mascara_nodulos = np.logical_and(mascara_nodulos, mascara_pulmao)
mascara_nodulos = morphology.remove_small_objects(mascara_nodulos, min_size=20)

# Renderização 3D utilizando a biblioteca centralizada
mesh_pulmao = utils.criar_mesh(mascara_pulmao.astype(np.uint8))
mesh_nodulos = utils.criar_mesh(mascara_nodulos.astype(np.uint8))

plotter = pv.Plotter()
if mesh_pulmao:
    plotter.add_mesh(mesh_pulmao, color="lightblue", opacity=0.1)
if mesh_nodulos:
    plotter.add_mesh(mesh_nodulos, color="red")
plotter.set_background("black")
plotter.show()