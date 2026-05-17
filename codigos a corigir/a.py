from utils import *

import os
import numpy as np

# pasta da série DICOM
path = r"C:\LIDC\pacientes\pacientes_estado_avançado\LIDC-IDRI-0068\1.3.6.1.4.1.14519.5.2.1.6279.6001.272402179924508785386397167107\1.3.6.1.4.1.14519.5.2.1.6279.6001.195674076148062852502601112560"

# =========================
# PEGAR TODOS OS DCM
# =========================

arquivos = [
    os.path.join(path, f)
    for f in os.listdir(path)
    if f.endswith(".dcm")
]

# =========================
# LER DICOMS
# =========================

slices = []

for arquivo in arquivos:

    image, dicom = load_dicom(arquivo)

    slices.append((image, dicom))

print(f"Slices: {len(slices)}")
slices.sort(
    key=lambda x: int(x[1].InstanceNumber)
)
volume = []

for image, dicom in slices:

    hu = convert_to_hu(image, dicom)

    hu = apply_window(
        hu,
        min_hu=-1000,
        max_hu=400
    )

    volume.append(hu)

volume = np.stack(volume)

indice = len(volume) // 2

show_image(
    volume[indice],
    "Slice em HU"
)
mask = threshold_segmentation(
    volume[indice],
    threshold=0.35
)

mask = remove_noise(mask)

mask = fill_holes(mask)

show_image(mask, "Máscara")