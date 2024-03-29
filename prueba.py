#%%
import matplotlib
#matplotlib.use("tkagg")
print(matplotlib.get_backend())

import matplotlib.pyplot as plt
import numpy as np

# Create a list of evenly-spaced numbers over the range
x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(2*x))       # Plot the sine of each x point
plt.show()                   # Display the plot


# %%

# %%
