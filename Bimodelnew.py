from keras.models import load_model, Model, model_from_json
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from tools import loadfasttext,preparation
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import numpy as np
from classifiers.tools import softmax,delete_weights,checkpoint,we_path,input_path,output_path,filepath,weights_path

class Bimodel(Model):
    def __init__(self):
        self.we_path = we_path
        self.input_path = input_path
        self.output_path = output_path
        self.opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)

        loadres = loadfasttext(self.we_path)
        prepres = preparation(self.input_path, self.output_path)
        
        Ty = 10
        n_a = 256
        n_s = 256       
        
        Tx = max([len(sentence) for sentence in prepres[0]])
        X = np.zeros((prepres[2], Tx, loadres[1]))
        onehot_encoder = OneHotEncoder(sparse=False)
        Y = onehot_encoder.fit_transform(prepres[1])
        classes = Y.shape[1]
        Y = Y.reshape((prepres[2], 1, classes))
        for i in range(prepres[2]):
            for j in range(len(prepres[0][i])):
                X[i, j, :] = loadres[0].wv[prepres[0][i][j]]
        X, Y = shuffle(X, Y, random_state=0)
        
        post_activation_LSTM_cell = LSTM(n_s, return_state=True)
        X = Input(shape=(Tx, loadres[1]))
        s0 = Input(shape=(n_s,), name='s0')
        c0 = Input(shape=(n_s,), name='c0')
        s = s0
        c = c0
        repeator = RepeatVector(Tx)
        concatenator = Concatenate(axis=-1)
        densor1 = Dense(10, activation="tanh")
        densor2 = Dense(1, activation="relu")
        activator = Activation(softmax, name='attention_weights')
        dotor = Dot(axes=1)

        def one_step_attention(a, s_prev):
            s_prev = repeator(s_prev)
            concat = concatenator([s_prev, a])
            e = densor1(concat)
            energies = densor2(e)
            alphas = activator(energies)
            context = dotor([alphas, a])
            return context

        global outputs,X,s0,c0
        outputs = []
        a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
        for t in range(Ty):
            context = one_step_attention(a, s)
            s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])
        output = Dense(classes, activation='softmax')(s)
        s0 = np.zeros((prepres[2], n_s))
        c0 = np.zeros((prepres[2], n_s))

        outputs = list(Y.swapaxes(0, 1))
        super().__init__(self,[X, s0, c0], output)

    def compile(self):
        super().compile(self.opt,loss='categorical_crossentropy',metrics=['accuracy'])

    def fit(self):
        delete_weights(weights_path)
        global epochs
        epochs= 100
        history = super().fit([X, s0, c0], outputs, epochs=epochs, batch_size=24, verbose=2, callbacks=checkpoint(filepath), shuffle=True)
        global rounded_loss
        rounded_loss = round(history.history['loss'][epochs - 1], 4)
        return history

    def load_weights(self):
        filename = r'C:\Users\yassinerh\PycharmProjects\chatbot\models\classificationmodels\weights-improvement-{}-{}.hdf5'.format(
            epochs, rounded_loss)
        super().fit(filename)
