import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image =Image.open('Couleurs.jpg')
plt.imshow(image)
plt.show()

order = 1

def svd_image(image, order): #stockage de la matrice compressée
    compressed = np.zeros(image.shape)

    U, S, V = np.linalg.svd(image)

#Reconstruction de la matrice compressée
    for i in range(order):
        Ui=U[:,i].reshape(-1,1)
        Vi=V[i,:].reshape(1,-1)
        Si=S[i]
        compressed += (Ui * Si * Vi)
    return


image = Image.open('Couleurs.jpg').convert("L")
imagearray = np.array(image, dtype = float)

compresssion = svd_image(image, 20)

plt.subplot(1,2,1)
plt.imshow(image)

plt.subplot(1,2,2)
plt.imshow(compresssion)
plt.show()

