import pandas as pd
import matplotlib.pyplot as plt
from parameters import *

paths = get_paths()
path = '{}/10epochs_noWmap_loss.txt'.format(paths['model_dir'])

df = pd.read_csv(path, sep='\s')

losses = list(df['loss'])
patches = list(df['patch_counter'])

plt.plot(patches, losses)
plt.xlabel('patch_counter')
plt.ylabel('loss')
plt.show()

