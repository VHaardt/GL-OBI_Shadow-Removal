# Experiment Log

## RESNET
* Esp_A: 
    - loss MSE(shadow) + 0.1 LPIPS(all)
* Esp_B: 
    - Decay exponential
    - Dropout 0.5
    - loss: L1(shadow) + 0.5 LPIPS(all)
    - LR: 0.001 (prima 0.0002)
    - blur
* Esp_C: come Esp_B
    - loss: L1(shadow) + 0.5 LPIPS(crop)
* Esp_B2: Esp_B con blur giousto
* Esp_C2: Esp_C con blur giousto
* Esp_D: Decay corretto (gamma .98)
    - loss: L1(shadow)
* Esp_E: Decay corretto 
    -loss: L1(shadow) + 0.5 LPIPS(crop) con blur giusto

## UNet
* UNET_esp_A:
    - input: blur_img + mask
    - out: 3ch da sommare all'input
    - loss: L1(shadow) + 0.5 LPIPS(crop)
* UNET_esp_B:
    - input: blur_img + mask
    - out: 6ch nuova mappa per input
    - loss: L1(shadow) + 0.5 LPIPS(crop)
* UNET_esp_C: come UNET_esp_C
    - input: blur_img + mask
    - out: 6ch nuova mappa per input
    - loss: L1(shadow) + 0.5 LPIPS(crop) + 0.1 PSNR(all)
* UNET_esp_C2:
    - input: blur_img + mask    
    - out: 6ch nuova mappa per input senza blur
    - loss: L1(shadow) + 0.5 LPIPS(crop) + 0.1 PSNR(all)
* UNET_esp_D:
    - input: img
    - out: 6ch nuova mappa per input
    - loss: L1(shadow) + 0.5 LPIPS(crop) + 0.1 MSE(light)
* UNET_3_A:
    - input: img + mask
    - out: 3ch non vincolati, residui da applicare
    - loss: L1(shadow) + 0.5 LPIPS(crop) + 0.1 MSE(all)
* UNET_3_B:
    - input: img + mask(expanded)
    - out: 3ch non vincolati, residui da applicare
    - loss: L1(shadow) + 0.5 LPIPS(all)
* UNET_3_C:
    - input: img + mask
    - out: 3ch non vincolati, residui da applicare
    - loss: L1(shadow) + 0.5 MSE(crop) + 0.1 LPIPS(all)
* UNET_3_D:
    - input: img + mask
    - out: 3ch non vincolati, residui da applicare
    - loss: L1(shadow) + 0.5 L1(crop) + 0.2 LPIPS(all)
