import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Masking
from tensorflow.keras.optimizers import Adam

class SequentialModel:
    def __init__(self, num_items, embedding_size=64, lstm_units=64):
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.lstm_units = lstm_units

    def get_model(self):
        model = Sequential([
            # Embedding layer to convert item indices into dense vectors of fixed size
            Embedding(input_dim=self.num_items, output_dim=self.embedding_size, mask_zero=True, name='embedding_layer'),
            
            # Masking layer to ignore the padding (mask_zero in Embedding)
            Masking(mask_value=0, name='masking_layer'),
            
            # LSTM layer for capturing temporal dynamics
            LSTM(self.lstm_units, name='lstm_layer'),
            
            # Dropout for regularization
            Dropout(0.2, name='dropout_layer'),
            
            # Dense layer for output prediction
            Dense(self.num_items, activation='softmax', name='output_layer')
        ])
        
        return model

def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Example usage
    NUM_ITEMS = 1700  # Placeholder, adjust according to your dataset's unique items
    
    seq_model = SequentialModel(num_items=NUM_ITEMS)
    model = seq_model.get_model()
    model = compile_model(model)
    
    # Print model summary to verify the architecture
    model.summary()
