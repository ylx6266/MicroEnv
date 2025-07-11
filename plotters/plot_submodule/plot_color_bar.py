import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

df = pd.read_csv('merged_color.csv')
colors = df['Color'].tolist()

rgb_colors = [mcolors.hex2color(color) for color in colors]

fig, ax = plt.subplots(figsize=(5, 1))

ax.imshow([rgb_colors], aspect='auto')
ax.set_xticks([]) 
ax.set_yticks([]) 

ax.set_title('Color Bar')

plt.show()
