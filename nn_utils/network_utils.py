from keras.models import Sequential
from keras.layers.core import TimeDistributedDense, Dense, Reshape, Dropout, Activation
# In TimeDistributedDense we apply the same dense layer for each time dimension. It's used when you want the entire O/P sequence
from keras.layers.recurrent import LSTM


def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions):
    model = Sequential()  # Sequential is a linear stack of layers
    # This layer converts frequency space to hidden space
    model.add(Dense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
    
    model.add(Reshape((-1, num_hidden_dimensions)))
    # return_sequences=True implies return the entire output sequence & not just the last output
    model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=False))
    model.add(Dropout(0.2))
    # This layer converts hidden space back to frequency space
    model.add(Dense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
    model.add(Activation('softmax'))
    # Done building the model.Now, configure it for the learning process
    # model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')
    return model
