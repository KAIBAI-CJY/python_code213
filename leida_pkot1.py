import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)

ax.set_ylim(0.8, 1.0)
ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])

ax.tick_params(axis='both', which='major', direction='in', length=8, width=2, colors='black', labelsize=14)
ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1, colors='black')
ax.minorticks_on()

plt.show()
