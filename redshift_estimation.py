import tensorflow as tf
from tensorflow.keras import layers, Model

def nmad_loss(y_true, y_pred):
    """
    Custom loss function based on NMAD.

    Args:
        y_true: Tensor of true redshift values.
        y_pred: Tensor of predicted redshift values.

    Returns:
        Tensor representing the NMAD loss.
    """
    # Compute the absolute differences between predicted and true values
    absolute_differences = tf.abs(y_pred - y_true)
    
    # Compute the standard deviation of the true values
    true_std = tf.math.reduce_std(y_true)
    
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-8
    true_std = tf.maximum(true_std, epsilon)
    
    # Compute NMAD
    nmad = tf.math.reduce_mean(absolute_differences / true_std)
    return nmad

# Input Spectrum
input_spectra = tf.keras.Input(shape=(N, 1))  # N is the length of the input spectrum

# Encoder Block
conv1 = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_spectra)
conv1 = layers.MaxPooling1D(pool_size=2)(conv1)

conv2 = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv1)
conv2 = layers.MaxPooling1D(pool_size=2)(conv2)

conv3 = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(conv2)
conv3 = layers.MaxPooling1D(pool_size=2)(conv3)

# Transformer Encoder
def transformer_encoder(inputs, num_heads, dff, dropout_rate=0.1):
    # Multi-Head Attention
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_output = layers.Dropout(dropout_rate)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    
    # Feedforward
    ffn = layers.Dense(dff, activation='relu')(out1)
    ffn = layers.Dense(inputs.shape[-1])(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)

x = conv3
for _ in range(3):  # Add 3 transformer encoder layers
    x = transformer_encoder(x, num_heads=8, dff=512, dropout_rate=0.1)

# Latent Space
latent_space = layers.GlobalAveragePooling1D()(x)
latent_space = layers.Dense(512, activation='relu', name='latent_space')(latent_space)

# Decoder/MLP Blocks
def mlp_block(inputs, units, dropout_rate=0.2):
    x = layers.Dense(units, activation='swish')(inputs)
    x = layers.Dropout(dropout_rate)(x)
    return x

x = latent_space
for _ in range(5):  # Add 5 MLP blocks
    x = mlp_block(x, units=512, dropout_rate=0.2)

# Linear Layers
linear1 = layers.Dense(512, activation='swish')(x)
linear2 = layers.Dense(256, activation='swish')(linear1)

# Output Layer
output = layers.Dense(1, activation='softplus')(linear2)

# Build the Model
model = Model(inputs=input_spectra, outputs=output)

# Compile the Model
model.compile(optimizer='adam', loss=nmad_loss, metrics=['mae'])
model.summary()
