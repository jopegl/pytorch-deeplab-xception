import numpy as np
import math
import os

caminho = 'leaf\masks\leaf_001_002_segmentation.raw'

tamanho_bytes = os.path.getsize(caminho)
lado = int(math.sqrt(tamanho_bytes))  # assumindo imagem quadrada
print(f'A imagem é aproximadamente {lado}x{lado}')
