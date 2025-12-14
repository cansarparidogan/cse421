import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

w = tf.keras.initializers.Constant([.5, -0.5])
b = tf.keras.initializers.Constant(0.)
l0=tf.keras.layers.Dense(units=1, 
                         input_shape=[2], 
                         kernel_initializer=w, 
                         bias_initializer=b, 
                         activation='sigmoid')

model = tf.keras.Sequential([l0])

model.compile(optimizer='adam', loss='binary_crossentropy')  # Compile the model

x_grid, y_grid = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))

x_gr_ravel = x_grid.ravel()
y_gr_ravel = y_grid.ravel()

Z = model(np.c_[x_gr_ravel, y_gr_ravel], training=False).numpy()
Z1 = model.predict(np.c_[x_gr_ravel, y_gr_ravel], verbose=0)

# Plot the model output
Z_reshaped = Z.reshape(x_grid.shape)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x_grid, y_grid, Z_reshaped, cmap=cm.coolwarm)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Output')
ax.set_title('Model Output Visualization')
plt.colorbar(surf)
plt.show()
