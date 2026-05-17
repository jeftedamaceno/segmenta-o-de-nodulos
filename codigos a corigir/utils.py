import numpy as np
import cv2
import pydicom
import matplotlib.pyplot as plt


# =========================
# DICOM E HU
# =========================

def load_dicom(path):

    dicom = pydicom.dcmread(path)

    image = dicom.pixel_array.astype(np.int16)

    return image, dicom


def convert_to_hu(image, dicom):

    intercept = dicom.RescaleIntercept
    slope = dicom.RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return image


def apply_window(image, min_hu=-1000, max_hu=400):

    image = np.clip(image, min_hu, max_hu)

    image = (image - min_hu) / (max_hu - min_hu)

    return image


# =========================
# PROCESSAMENTO
# =========================

def resize_image(image, size=(256, 256)):

    return cv2.resize(image, size)


def normalize_image(image):

    image = image.astype(np.float32)

    image = image / np.max(image)

    return image


def threshold_segmentation(image, threshold=0.5):

    binary = (image > threshold).astype(np.uint8)

    return binary


# =========================
# MORFOLOGIA
# =========================

def remove_noise(mask):

    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return opening


def fill_holes(mask):

    kernel = np.ones((5,5), np.uint8)

    closing = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return closing


# =========================
# VISUALIZAÇÃO
# =========================

def show_image(image, title="Imagem"):

    plt.figure(figsize=(6,6))

    plt.imshow(image, cmap="gray")

    plt.title(title)

    plt.axis("off")

    plt.show()


def show_3d_slice(image):

    plt.figure(figsize=(8,8))

    plt.imshow(image, cmap='gray')

    plt.colorbar()

    plt.title("Slice CT")

    plt.show()


# =========================
# MÉTRICAS CLÁSSICAS
# =========================

def calculate_area(mask):

    return np.sum(mask)


def calculate_perimeter(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    perimeter = 0

    for cnt in contours:
        perimeter += cv2.arcLength(cnt, True)

    return perimeter


def circularity(area, perimeter):

    if perimeter == 0:
        return 0

    return (4 * np.pi * area) / (perimeter ** 2)