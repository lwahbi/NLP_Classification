import os
from keras.models import load_model, Model, model_from_json
import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint
import gensim
import pandas as pd
from gensim.models import KeyedVectors
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from flask import Flask, render_template, request,jsonify

# Chemins absolus
we_path = r'C:\Users\yassinerh\PycharmProjects\chatbot\models\wordembeddingmodels\fast_cbow_300D'
input_path = r'C:\Users\yassinerh\PycharmProjects\chatbot\data\txtdata\input.txt'
output_path = r'C:\Users\yassinerh\PycharmProjects\chatbot\data\txtdata\output.txt'
weights_path = r'C:\Users\yassinerh\PycharmProjects\chatbot\models\classificationmodels'
filepath = r'C:\Users\yassinerh\PycharmProjects\chatbot\models\classificationmodels\weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'

# {Numero de la classe : Nom de la classe}
classes_names = {
    1: "Dépot Document",
    2: "Abs/Congés",
    3: "Abs/Congés",
    4: "Info perso",
    5: "Arkevia",
    6: "Griffe",
    7: "Abs/Congés",
    8: "Autres",
    9: "Télétravail",
    10: "Autres",
    11: "Info perso",
    12: "Info perso",
    13: "Abs/Congés",
    14: "Télétravail",
    15: "Manager",
    16: "Dépot Document",
    17: "Dépot Document",
    18: "Abs/Congés",
    19: "Abs/Congés",
    20: "Manager",
    21: "Manager",
    22: "Abs/Congés",
    23: "Acompte",
    24: "Abs/Congés",
    25: "Accès smart-rh",
    26: "Abs/Congés",
    27: "Arkevia",
    28: "Info perso",
    29: "Info perso",
    30: "Info perso",
    31: "Info perso",
    32: "Accès smart-rh",
    33: "Accès smart-rh",
    34: "Abs/Congés",
    35: "Abs/Congés",
    36: "Abs/Congés",
    37: "Abs/Congés"}

def read_sentences(path):
    tmp = []
    with open(path, encoding="utf-8") as file:
        for sentence in file.readlines():
            tmp.append(sentence.strip())
    return np.array(tmp)

########################## For Keras layers #################################

def softmax(x, axis=1):
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')

########################## For Keras layers #################################

def save_model_json(model, word_dim):
    model_json = model.to_json()

    with open("model_{}.json".format(word_dim), "w") as json_file:
        json_file.write(model_json)

    # Serialisation des poids vers HDF5
    model.save_weights("model_{}.h5".format(word_dim))
    print("Saved model to disk")

    file_name = 'model_{}'.format(word_dim)
    return file_name

def load_model_json(file_name):
    json_file = open(file_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # Charger les poids vers un nouveau model
    model.load_weights(file_name + ".h5")
    print("Loaded model from disk")
    return model

############################## Glove model  ################################

def similar_posneg(model, positive, negative, topn=10):
    """
    Doc is not available
    """
    mean_vecs = []

    for word in positive:
        mean_vecs.append(model.word_vectors[model.dictionary[word]])
    for word in negative:
        mean_vecs.append(-1 * model.word_vectors[model.dictionary[word]])

    mean = np.array(mean_vecs).mean(axis=0)
    mean /= np.linalg.norm(mean)

    dists = np.dot(model.word_vectors, mean)

    best = np.argsort(dists)[::-1]

    results = [(model.inverse_dictionary[i], dists[i]) for i in best if (model.inverse_dictionary[i] not in positive and
                                                                         model.inverse_dictionary[i] not in negative)][
              :topn]

    return results


############################### More utils #################################

def get_by_address(address):
    return [x for x in globals().values() if id(x) == address]

def delete_weights(models_path):
    for the_file in os.listdir(models_path):
        file_path = os.path.join(models_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def loadfasttext(path):
    model_w2v = gensim.models.fasttext.FastText.load(path)
    word_dim = model_w2v.wv['.'].shape[0]
    return model_w2v, word_dim

def preparation(input_path, output_path):
    data = read_sentences(path=input_path).astype(str)
    target = read_sentences(path=output_path).astype(int)
    # m nombre d'echantillants
    m = data.shape[0]
    print('Ce dataset contient {} échantillants.'.format(m))
    data = [data[i].split() for i in range(m)]
    target = target.reshape(len(target), 1)
    return data, target, m

def decode_sequence(input_seq, n_s, Tx, word_dim, model_w2v, model):
    state_value = np.zeros((1, n_s))

    input_buffer = np.zeros((1, Tx, word_dim))
    input_words = input_seq.split()

    for i in range(len(input_words)):
        input_buffer[0, i, :] = model_w2v.wv[input_words[i]]

    return model.predict([input_buffer, state_value, state_value])

def annuaire(i):
    path = r'/jsondata/listerep.json'
    df = pd.read_json(path)
    rep = df.values.tolist()
    x = rep[i]
    return x

def get_response(input_seq, n_s, Tx, word_dim, model_w2v, model):
    response = decode_sequence(input_seq, n_s, Tx, word_dim, model_w2v, model)
    classifier_output_class = classes_names[np.argmax(response) + 1]
    classifier_output_rep = annuaire(np.argmax(response))
    scores = np.transpose(response)
    dict_classifier = {"class": str(np.argmax(response) + 1) + ":" + classifier_output_class,
                       "reponse": classifier_output_rep[1],
                       "score": scores[np.argmax(response)][0], "jsondata": response}
    return dict_classifier

def get_stat(r):
    scores = np.transpose(r)
    lst = []
    for i in range(len(scores)):
        lst.append(scores[i][0])
    top_2_idx = np.argsort(lst)[-5:]
    rev_top_2 = np.flip(top_2_idx)
    top_2_values = [lst[i] for i in top_2_idx]
    rev_top_2_val = np.flip(top_2_values)
    x = np.transpose(rev_top_2)
    for i in range(len(x)):
        x[i] += 1
    y = np.transpose(rev_top_2_val)
    import pandas as pd
    dataset = pd.DataFrame({'classe': x, 'precision': y},
                           columns=['classe', 'precision'])
    return dataset

def checkpoint(filepath):
    checkpoint = ModelCheckpoint(filepath, monitor='loss',verbose=1,save_best_only=False,mode='min',save_weights_only=True)
    return [checkpoint]

def start():
    app = Flask(__name__)
    @app.route("/")
    def hello():
        return render_template('chat.html')
    
    @app.route("/ask", methods=['POST'])
    def ask():
        message = request.form['messageText'].encode('utf-8').strip()
        print(message)
        a = get_response(str(message))
        stat = get_stat(a["jsondata"])
        lst = []
        for i in range(len(stat)):
            lst.append(classes_names[int(stat.loc[i, "classe"])])
        dfclasse = pd.DataFrame(lst, columns=['libelle'])
        dfconc = pd.concat([stat, dfclasse], axis=1, sort=False)
        dfconc = dfconc[['classe', 'libelle', 'precision']]
        repk = dfconc.to_dict('records')
        if float(a["score"]) > 0.7:
            return jsonify(
                {'status': 'OK', 'answer': a["reponse"], 'infos': repk})
        else:
            return jsonify({'status': 'OK',
                            'answer': 'Oops désolé ,je ne peux pas vous répondre.',
                            'infos': repk})
    if __name__ == '__main__':
        app.run()