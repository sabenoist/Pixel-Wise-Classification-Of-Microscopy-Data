import pandas as pd
import matplotlib.pyplot as plt
from parameters import *

paths = get_paths()
path = '{}/Final_25_6_lr-9_stock_loss.txt'.format(paths['model_dir'])

df = pd.read_csv(path, sep='\s')

losses = list(df['loss'])
patches = list(df['patch_counter'])

plt.plot(patches, losses)
plt.xlabel('number of patches trained')
plt.ylabel('loss')
plt.title('Cross-entropy Loss over time with learning rate: 10^-9')
plt.show()

