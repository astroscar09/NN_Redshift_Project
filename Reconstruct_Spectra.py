import tensorflow as tf
from tensorflow.keras import layers, Model


# Multi-Head Attention Block
def attention_block(inputs, num_heads, dff, dropout_rate=0.1):
    # Multi-Head Attention
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = layers.Dropout(dropout_rate)(attention_output)
    attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feedforward Network
    ff_output = layers.Dense(dff, activation='relu')(attention_output)
    ff_output = layers.Dense(inputs.shape[-1])(ff_output)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    
    # Residual Connection
    output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ff_output)
    return output

# Autoencoder Block
def autoencoder_block(inputs, num_heads, dff, dropout_rate=0.1):
    # Encoder
    attention_output = attention_block(inputs, num_heads, dff, dropout_rate)
    
    # Downsample
    downsampled = layers.Conv1D(filters=inputs.shape[-1] * 2, kernel_size=3, strides=2, padding='same')(attention_output)
    
    # Decoder
    upsampled = layers.Conv1DTranspose(filters=inputs.shape[-1], kernel_size=3, strides=2, padding='same')(downsampled)
    
    # Residual Connection
    output = layers.Add()([attention_output, upsampled])
    return output

# Decoder Block with Attention and Feedforward Layers
def decoder_block(inputs, encoder_output, num_heads, dff, dropout_rate=0.1):
    # Multi-Head Self-Attention
    self_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    self_attention = layers.Dropout(dropout_rate)(self_attention)
    self_attention = layers.LayerNormalization(epsilon=1e-6)(inputs + self_attention)
    
    # Encoder-Decoder Attention
    enc_dec_attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(
        self_attention, encoder_output
    )
    enc_dec_attention = layers.Dropout(dropout_rate)(enc_dec_attention)
    enc_dec_attention = layers.LayerNormalization(epsilon=1e-6)(self_attention + enc_dec_attention)
    
    # Feedforward Layers
    ff_output = layers.Dense(dff, activation='relu')(enc_dec_attention)
    ff_output = layers.Dense(inputs.shape[-1])(ff_output)
    ff_output = layers.Dropout(dropout_rate)(ff_output)
    
    # Residual Connection
    output = layers.LayerNormalization(epsilon=1e-6)(enc_dec_attention + ff_output)
    return output



# Input Spectrum
input_spectra = tf.keras.Input(shape=(N, 1))  # N is the length of the 1D spectrum

# CNN Layers for Feature Extraction
small_kernel = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_spectra)
medium_kernel = layers.Conv1D(filters=32, kernel_size=7, activation='relu', padding='same')(input_spectra)
large_kernel = layers.Conv1D(filters=32, kernel_size=15, activation='relu', padding='same')(input_spectra)

concatenated = layers.Concatenate()([small_kernel, medium_kernel, large_kernel])

# Pass Through Autoencoder Blocks (Encoder)
x = concatenated
for _ in range(3):  # Add 3 autoencoder blocks
    x = autoencoder_block(x, num_heads=4, dff=128, dropout_rate=0.1)

# Latent Space
latent_space = layers.GlobalAveragePooling1D()(x)
latent_space = layers.Dense(128, activation='relu', name='latent_space')(latent_space)

# Decoder Blocks
decoder_input = layers.Dense(N // 2, activation='relu')(latent_space)
decoder_input = layers.Reshape((N // 2, 1))(decoder_input)

x = decoder_input
for _ in range(3):  # Add 3 decoder blocks
    x = decoder_block(x, encoder_output=concatenated, num_heads=4, dff=128, dropout_rate=0.1)

# Final Reconstruction
reconstruction_256 = layers.Dense(256, activation='relu')(x)
reconstruction_128 = layers.Dense(128, activation='relu')(reconstruction_256)
reconstruction_64 = layers.Dense(64, activation='relu')(reconstruction_128)

final_reconstruction = layers.Conv1D(filters=1, kernel_size=3, activation='linear', padding='same')(reconstruction_64)

# Build the Model
model = Model(inputs=input_spectra, outputs=final_reconstruction)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
