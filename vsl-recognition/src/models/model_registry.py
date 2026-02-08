from .new_hybrids import (
    create_baseline_lstm,
    create_cnn_lstm,
    create_simple_transformer,
    create_cnn_transformer,
    create_lstm_transformer,
    create_bilstm_attention,
    create_dual_stream_transformer,
    create_residual_mlp_transformer,
    create_sep_cnn_transformer,
    create_transformer_lstm_hybrid
)

MODEL_REGISTRY = {
    'Baseline_LSTM': create_baseline_lstm,
    'CNN_LSTM': create_cnn_lstm,
    'Simple_Transformer': create_simple_transformer,
    'CNN_Transformer': create_cnn_transformer,
    'LSTM_Transformer': create_lstm_transformer,
    'BiLSTM_Attention': create_bilstm_attention,
    'Dual_Stream_Transformer': create_dual_stream_transformer,
    'Residual_MLP_Transformer': create_residual_mlp_transformer,
    'Sep_CNN_Transformer': create_sep_cnn_transformer,
    'Transformer_LSTM_Hybrid': create_transformer_lstm_hybrid
}

def get_model(name, num_classes, sequence_length):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    
    print(f"Creating model: {name}")
    return MODEL_REGISTRY[name](num_classes, sequence_length)

def list_models():
    return list(MODEL_REGISTRY.keys())
