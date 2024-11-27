import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

'''
# Example data
X = np.random.rand(1000, 1000)  # 1000 spectra, 1000 wavelength bins
y_redshift = np.random.uniform(0, 10, 1000)  # Redshifts between 0 and 3

# Split data
X_train, X_test, y_z_train, y_z_test = train_test_split(
    X, y_redshift, test_size=0.2, random_state=42
)

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# Define the model
input_layer = tf.keras.layers.Input(shape=(1000,))
x = tf.keras.layers.Conv1D(32, 3, activation='relu')(tf.expand_dims(input_layer, -1))
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

# Outputs
redshift_output = tf.keras.layers.Dense(1, activation='linear', name='redshift_output')(x)

model = tf.keras.Model(inputs=input_layer, outputs=[redshift_output])

# Compile the model
model.compile(
    optimizer='adam',
    loss={'redshift_output': 'mse'},
    metrics={'redshift_output': 'mae'}
)

# Train the model
history = model.fit(
    X_train, {'redshift_output': y_z_train},
    validation_data=(X_test, {'redshift_output': y_z_test}),
    epochs=20,
    batch_size=32
)

# Evaluate the model
redshift_loss, redshift_mae = model.evaluate(
    X_test, {'redshift_output': y_z_test}
)
'''
#making the redshift arrays
#z_grids = np.arange(1., 12, .01)

#getting the rest_frame wavelength and model spectrum 
#wavelength_arr, flux_arr = make_model(1e-20)

#making the wavelengths into a 2D array
#wavelength_array = np.tile(wavelength_arr, (z_grids.shape[0], 1))

#convert wavelengths to observed frame
#obs_wave = wavelength_array * (1 + z_grids[:, np.newaxis])

##fluxes = np.zeros((z_grids.shape[0], wavelength_arr.shape[0]))

#for i in range(z_grids.shape[0]):
#    fluxes[i] = flux_arr


fluxes = np.loadtxt('TEST_500.txt')
wavelength_arr = np.loadtxt('TEST_500_Wavelengths_Microns.txt')

wavelength_array = np.tile(wavelength_arr, (fluxes.shape[0], 1))
redshift_array = np.loadtxt('TEST_500_Predictions.txt')


flux_input = tf.keras.layers.Input(shape=(fluxes.shape[1],), name='flux_input')
wavelength_input = tf.keras.layers.Input(shape=(fluxes.shape[1],), name='wavelength_input')

combined = tf.keras.layers.Concatenate()([tf.expand_dims(flux_input, -1), tf.expand_dims(wavelength_input, -1)])
x = tf.keras.layers.Conv1D(32, 3, activation='relu')(combined)
x = tf.keras.layers.MaxPooling1D(2)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

redshift_output = tf.keras.layers.Dense(1, activation='linear', name='redshift_output')(x)

model = tf.keras.Model(inputs=[flux_input, wavelength_input], outputs=[redshift_output])

model.compile(
    optimizer='adam',
    loss={ 'redshift_output': 'mse'},
    metrics={'redshift_output': 'mae'}
)

hist = model.fit(
    {'flux_input': fluxes, 'wavelength_input': wavelength_array},
    {'redshift_output': redshift_array},
    epochs=10,
    batch_size=32
)