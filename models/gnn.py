import spektral
import tensorflow as tf
from spektral.layers import GraphConv
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class GNNModel:
    def __init__(self, num_features, num_classes):
        self.num_features = num_features  # Number of features per node
        self.num_classes = num_classes    # Number of output classes (items)

    def get_model(self):
        # Inputs
        X_in = Input(shape=(self.num_features, ), name='X_in')
        A_in = Input(shape=(None, ), sparse=True, name='A_in')

        # GNN layers
        graph_conv_1 = GraphConv(64, activation='relu')([X_in, A_in])
        dropout_1 = Dropout(0.5)(graph_conv_1)
        graph_conv_2 = GraphConv(self.num_classes, activation='softmax')([dropout_1, A_in])

        # Build the model
        model = Model(inputs=[X_in, A_in], outputs=graph_conv_2)
        return model

def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Example usage
    NUM_FEATURES = 10   # Example: Number of features for each user/item node
    NUM_CLASSES = 1700  # Placeholder: Number of unique items (or classes for prediction)

    gnn_model = GNNModel(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
    model = gnn_model.get_model()
    model = compile_model(model)
    
    # Print model summary to verify the architecture
    model.summary()
