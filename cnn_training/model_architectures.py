from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform  # Xavier initializer


def I5_architecture(input_shape=(32, 15)):
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(64, (5, 3), padding='same', kernel_initializer=GlorotUniform(), input_shape=(*input_shape, 1)),
        layers.BatchNormalization(),  # Batch normalization
        layers.LeakyReLU(alpha=0.1),  # Leaky ReLU with alpha=0.1
        layers.MaxPooling2D((2, 1)),  # Pooling with size (2, 1)

        # Second Convolutional Block
        layers.Conv2D(128, (5, 3), padding='same', kernel_initializer=GlorotUniform()),
        layers.BatchNormalization(),  # Batch normalization
        layers.LeakyReLU(alpha=0.1),  # Leaky ReLU with alpha=0.1
        layers.MaxPooling2D((2, 1)),  # Pooling with size (2, 1)

        # Flatten and Fully Connected Layers
        layers.Flatten(),
        layers.Dense(1, kernel_initializer=GlorotUniform()),  # Fully connected layer
        layers.Dropout(0.5),  # 50% dropout
        layers.Activation('sigmoid')  # Binary classification (True/False)
    ])

    # Use Adam optimizer with a learning rate of 0.001 --> Kelly uses 1 x 10^(-5) [0.00001]
    optimizer = Adam(learning_rate=0.01)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def I20_architecture(input_shape=(64, 60)):
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(64, (5, 3), padding='same', kernel_initializer=GlorotUniform(),
                      dilation_rate=(2, 1), input_shape=(*input_shape, 1)),  # Dilation only
        layers.BatchNormalization(),  # Batch normalization
        layers.LeakyReLU(alpha=0.1),  # Leaky ReLU with alpha=0.1
        layers.MaxPooling2D((3, 1)),  # Simulating vertical strides

        # Second Convolutional Block
        layers.Conv2D(128, (5, 3), padding='same', kernel_initializer=GlorotUniform(),
                      strides=(1, 1), dilation_rate=(1, 1)),  # Regular convolution
        layers.BatchNormalization(),  # Batch normalization
        layers.LeakyReLU(alpha=0.1),  # Leaky ReLU with alpha=0.1
        layers.MaxPooling2D((2, 1)),  # Pooling with size (2, 1)

        # Third Convolutional Block
        layers.Conv2D(256, (5, 3), padding='same', kernel_initializer=GlorotUniform(),
                      strides=(1, 1), dilation_rate=(1, 1)),  # Regular convolution
        layers.BatchNormalization(),  # Batch normalization
        layers.LeakyReLU(alpha=0.1),  # Leaky ReLU with alpha=0.1
        layers.MaxPooling2D((2, 1)),  # Pooling with size (2, 1)

        # Flatten and Fully Connected Layers
        layers.Flatten(),
        layers.Dense(1, kernel_initializer=GlorotUniform()),  # Fully connected layer
        layers.Dropout(0.5),  # 50% dropout
        layers.Activation('sigmoid')  # Binary classification (True/False)
    ])

    # Use Adam optimizer with a learning rate of 0.01
    optimizer = Adam(learning_rate=0.01)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def I60_architecture(input_shape=(96, 180)):
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(64, (5, 3), padding='same', kernel_initializer=GlorotUniform(),
                      dilation_rate=(2, 1), input_shape=(*input_shape, 1)),  # Dilation only
        layers.BatchNormalization(),  # Batch normalization
        layers.LeakyReLU(alpha=0.1),  # Leaky ReLU with alpha=0.1
        layers.MaxPooling2D((3, 1)),  # Simulating vertical strides

        # Second Convolutional Block
        layers.Conv2D(128, (5, 3), padding='same', kernel_initializer=GlorotUniform(),
                      strides=(1, 1), dilation_rate=(1, 1)),  # Regular convolution
        layers.BatchNormalization(),  # Batch normalization
        layers.LeakyReLU(alpha=0.1),  # Leaky ReLU with alpha=0.1
        layers.MaxPooling2D((2, 1)),  # Pooling with size (2, 1)

        # Third Convolutional Block
        layers.Conv2D(256, (5, 3), padding='same', kernel_initializer=GlorotUniform(),
                      strides=(1, 1), dilation_rate=(1, 1)),  # Regular convolution
        layers.BatchNormalization(),  # Batch normalization
        layers.LeakyReLU(alpha=0.1),  # Leaky ReLU with alpha=0.1
        layers.MaxPooling2D((2, 1)),  # Pooling with size (2, 1)

        # Third Convolutional Block
        layers.Conv2D(512, (5, 3), padding='same', kernel_initializer=GlorotUniform(),
                      strides=(1, 1), dilation_rate=(1, 1)),  # Regular convolution
        layers.BatchNormalization(),  # Batch normalization
        layers.LeakyReLU(alpha=0.1),  # Leaky ReLU with alpha=0.1
        layers.MaxPooling2D((2, 1)),  # Pooling with size (2, 1)

        # Flatten and Fully Connected Layers
        layers.Flatten(),
        layers.Dense(1, kernel_initializer=GlorotUniform()),  # Fully connected layer
        layers.Dropout(0.5),  # 50% dropout
        layers.Activation('sigmoid')  # Binary classification (True/False)
    ])

    # Use Adam optimizer with a learning rate of 0.01
    optimizer = Adam(learning_rate=0.01)

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model