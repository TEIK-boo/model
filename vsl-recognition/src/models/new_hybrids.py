import tensorflow as tf
from tensorflow.keras import layers, Model
try:
    from .components import create_hand_branch, create_face_branch, create_pose_branch
    from .transformer_utils import PositionalEncoding, TransformerBlock
    from .hybrid import create_hybrid_multistream_model
except ImportError:
    from components import create_hand_branch, create_face_branch, create_pose_branch
    from transformer_utils import PositionalEncoding, TransformerBlock
    from hybrid import create_hybrid_multistream_model

def get_input_layer(sequence_length, input_dim=1662):
    return layers.Input(shape=(sequence_length, input_dim), name='sequence_input')

# 1. Baseline MLP + LSTM (Wrapper)
def create_baseline_lstm(num_classes, sequence_length):
    return create_hybrid_multistream_model(num_classes, sequence_length)

# 2. CNN + LSTM
def create_cnn_lstm(num_classes, sequence_length):
    inputs = get_input_layer(sequence_length)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # 1D CNN for temporal/spatial feature extraction
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # LSTM for sequence modeling
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(64)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='CNN_LSTM')

# 3. Simple Transformer
def create_simple_transformer(num_classes, sequence_length):
    inputs = get_input_layer(sequence_length)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # Projection to d_model
    x = layers.Dense(128, activation='relu')(x)
    
    # Transformer
    x = PositionalEncoding(sequence_length, 128)(x)
    x = TransformerBlock(128, 4, 256)(x)
    x = TransformerBlock(128, 4, 256)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='Simple_Transformer')

# 4. CNN + Transformer
def create_cnn_transformer(num_classes, sequence_length):
    inputs = get_input_layer(sequence_length)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # CNN Feature Extractor
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    
    # Transformer
    x = PositionalEncoding(sequence_length, 128)(x)
    x = TransformerBlock(128, 4, 256)(x)
    x = TransformerBlock(128, 4, 256)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='CNN_Transformer')

# 5. LSTM + Transformer
def create_lstm_transformer(num_classes, sequence_length):
    inputs = get_input_layer(sequence_length)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # LSTM to process local temporal dependencies
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LayerNormalization()(x)
    
    # Transformer for global dependencies
    x = PositionalEncoding(sequence_length, 128)(x) # Optional as LSTM already encodes order, but safe to add
    x = TransformerBlock(128, 4, 256)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='LSTM_Transformer')

# 6. BiLSTM + Attention (Self-Attention)
def create_bilstm_attention(num_classes, sequence_length):
    inputs = get_input_layer(sequence_length)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    # Attention Layer (Dot product self-attention)
    # Query, Key, Value all from x
    attention = layers.Attention()([x, x])
    
    # Combine
    x = layers.GlobalAveragePooling1D()(attention)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='BiLSTM_Attention')

# 7. Dual Stream Transformer (Specialized for Sign Language)
def create_dual_stream_transformer(num_classes, sequence_length):
    inputs = get_input_layer(sequence_length)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # Split: Hands vs Body+Face
    # Hands: 1536:1662
    # Body+Face: 0:1536
    hand_seq = layers.Lambda(lambda x: x[:, :, 1536:])(x)
    body_seq = layers.Lambda(lambda x: x[:, :, :1536])(x)
    
    # Stream 1: Hands
    h = layers.Dense(64, activation='relu')(hand_seq)
    h = PositionalEncoding(sequence_length, 64)(h)
    h = TransformerBlock(64, 4, 128)(h)
    h = layers.GlobalAveragePooling1D()(h)
    
    # Stream 2: Body/Face
    b = layers.Dense(64, activation='relu')(body_seq)
    b = PositionalEncoding(sequence_length, 64)(b)
    b = TransformerBlock(64, 4, 128)(b)
    b = layers.GlobalAveragePooling1D()(b)
    
    # Fusion
    combined = layers.Concatenate()([h, b])
    x = layers.Dense(64, activation='relu')(combined)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='Dual_Stream_Transformer')

# 8. Residual MLP + Transformer
def create_residual_mlp_transformer(num_classes, sequence_length):
    inputs = get_input_layer(sequence_length)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # Residual MLP Block per frame
    def res_block(x, units):
        sc = layers.Dense(units)(x)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dense(units)(x)
        return layers.Add()([x, sc])
    
    x = layers.TimeDistributed(layers.Dense(128))(x)
    x = layers.TimeDistributed(layers.Lambda(lambda x: res_block(x, 128)))(x)
    
    # Transformer
    x = PositionalEncoding(sequence_length, 128)(x)
    x = TransformerBlock(128, 4, 256)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='Residual_MLP_Transformer')

# 9. Separable CNN + Transformer (Lightweight)
def create_sep_cnn_transformer(num_classes, sequence_length):
    inputs = get_input_layer(sequence_length)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    x = layers.SeparableConv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv1D(128, 3, padding='same', activation='relu')(x)
    
    x = PositionalEncoding(sequence_length, 128)(x)
    x = TransformerBlock(128, 4, 256)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='Sep_CNN_Transformer')

# 10. Transformer + LSTM Hybrid (Encoder-Decoder style thought)
def create_transformer_lstm_hybrid(num_classes, sequence_length):
    inputs = get_input_layer(sequence_length)
    x = layers.Masking(mask_value=0.0)(inputs)
    
    # Project
    x = layers.Dense(128)(x)
    
    # Transformer Encoder features
    x = PositionalEncoding(sequence_length, 128)(x)
    x = TransformerBlock(128, 4, 256)(x)
    
    # LSTM to aggregate over time
    x = layers.LSTM(64, return_sequences=False)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='Transformer_LSTM_Hybrid')
