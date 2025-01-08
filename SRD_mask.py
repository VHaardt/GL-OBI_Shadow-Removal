#!/usr/bin/env python3
import os
import numpy as np
import random
import torch.utils.data as data
import cv2
import skimage
from tqdm import tqdm

from utils.utils_io import read_image, process_mask

# Funzione per calcolare la shadow mask (Placeholder)
def compute_shadow_mask(shadow_image, shadow_free_image):
        # convert to grayscale
        shadow_image = skimage.color.rgb2gray(shadow_image)
        shadow_free_image = skimage.color.rgb2gray(shadow_free_image)

        # Blur the images
        shadow_image = skimage.filters.gaussian(shadow_image, sigma=5)
        shadow_free_image = skimage.filters.gaussian(shadow_free_image, sigma=5)

        shadow_mask = np.abs(shadow_image - shadow_free_image)

        # Binarize the mask using otsu thresholding
        otsu_threshold = skimage.filters.threshold_otsu(shadow_mask)
        shadow_mask = np.where(shadow_mask > otsu_threshold, 1, 0).astype(np.float32)
        
        return shadow_mask

# Funzione per eliminare i file da tutte le cartelle
def delete_images(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
            print(f"Deleted {path}")

# Funzione principale per processare le immagini
def process_images(root_dir):
    for subdir in ['train', 'test']:
        # Percorsi delle cartelle
        path_A = os.path.join(root_dir, subdir, f'{subdir}_A')  # Shadow images
        path_B = os.path.join(root_dir, subdir, f'{subdir}_B')  # Masks
        path_C = os.path.join(root_dir, subdir, f'{subdir}_C')  # Shadow-free images

        # Lista dei file in test_B
        file_list = os.listdir(path_B)

        # Usa tqdm per mostrare la barra di progresso
        for index, filename in tqdm(enumerate(file_list), total=len(file_list), desc=f"Processing {subdir}"):
            file_path_B = os.path.join(path_B, filename)

            if os.path.isfile(file_path_B):
                # Leggi il shadow mask come immagine in scala di grigi
                shadow_mask = np.round(read_image(file_path_B))

                # Aggiungi una dimensione al mask (se necessario)
                mask_to_check = np.expand_dims(shadow_mask, axis=2)

                # Se il mask non rispetta il controllo, calcola una nuova shadow mask
                if np.all(process_mask(mask_to_check, min_size=100, max_size=100) == 0):
                    print(f"Mask not valid, recalculating...{filename}")

                    # Percorsi per le immagini shadow e shadow-free
                    shadow_image_path = os.path.join(path_A, filename)
                    shadow_free_image_path = os.path.join(path_C, filename)

                    # Carica le immagini necessarie per calcolare la shadow mask
                    shadow_image = read_image(shadow_image_path)
                    shadow_free_image = read_image(shadow_free_image_path)
                    
                    # Calcola una nuova shadow mask
                    shadow_mask = compute_shadow_mask(shadow_image, shadow_free_image)

                    # Assicurati che la maschera abbia un singolo canale prima di salvarla
                    if shadow_mask.ndim > 2:
                        shadow_mask = cv2.cvtColor(shadow_mask, cv2.COLOR_BGR2GRAY)

                    shadow_mask = (shadow_mask * 255).astype(np.uint8)

                    # Sovrascrivi l'immagine con il nuovo mask (salvato come singolo canale)
                    cv2.imwrite(file_path_B, shadow_mask)

                    # Controlla nuovamente la nuova maschera
                    mask_to_check = np.expand_dims(shadow_mask, axis=2)
                    if np.all(process_mask(mask_to_check, min_size=100, max_size=100) == 0):
                        print(f"Mask still not valid, deleting images for {filename}...")
                        # Se la nuova maschera non rispetta ancora il controllo, elimina le immagini
                        delete_images([shadow_image_path, file_path_B, shadow_free_image_path])

if __name__ == "__main__":
    # Esegui il controllo sulle cartelle (train e test)
    root_dir = "/home/vhaardt/Desktop/ShadowRemoval/data/SRD"
    process_images(root_dir)
