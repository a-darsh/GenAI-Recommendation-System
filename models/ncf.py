import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class NCF:
    def __init__(self, num_users, num_items, embedding_size=64, layers=[64, 32, 16, 8]):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.layers = layers

    def get_model(self):
        # User and Item Embedding Layers
        user_input = Input(shape=(1,), name='user_input')
        item_input = Input(shape=(1,), name='item_input')

        user_embedding = Embedding(input_dim=self.num_users, output_dim=self.embedding_size, name='user_embedding')(user_input)
        item_embedding = Embedding(input_dim=self.num_items, output_dim=self.embedding_size, name='item_embedding')(item_input)

        # Flatten the embedding vectors
        user_vector = Flatten(name='flatten_users')(user_embedding)
        item_vector = Flatten(name='flatten_items')(item_embedding)

        # Concatenate the flatten embeddings
        concat = Concatenate()([user_vector, item_vector])

        # Fully connected layers
        for idx, layer_size in enumerate(self.layers):
            layer = Dense(layer_size, activation='relu', name=f'layer{idx}')
            dropout = Dropout(0.2, name=f'dropout{idx}')
            concat = dropout(layer(concat))

        # Output layer
        output = Dense(1, activation='sigmoid', name='output')(concat)

        model = Model(inputs=[user_input, item_input], outputs=output)
        
        return model

def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Example usage
    NUM_USERS = 1000  # Placeholder, adjust to your dataset
    NUM_ITEMS = 1700  # Placeholder, adjust to your dataset

    ncf = NCF(num_users=NUM_USERS, num_items=NUM_ITEMS)
    model = ncf.get_model()
    model = compile_model(model)
    
    # Print model summary to verify the architecture
    model.summary()
