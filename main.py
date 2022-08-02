import os
import random

import matplotlib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from helper import calculate_results
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.utils import plot_model
import time
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import pickle
from nltk.corpus import stopwords
import string
import re


class Data_loader:
    __slots__ = 'data_dir', 'train_df', 'test_df', 'val_df', 'data_facter', 'train_sentences', \
                'val_sentences', 'test_sentences', 'train_labels_encoded', 'val_labels_encoded', \
                'test_labels_encoded', 'num_classes', 'class_names', 'train_labels_one_hot', 'val_labels_one_hot', \
                'test_labels_one_hot', 'token_model_history_file', 'sw_nltk', 'text_output_seq_len', 'text_vocab', 'text_vectorizer', \
                'output_seq_char_len', 'num_char_tokens', 'char_vectorizer', 'train_chars', 'val_chars', 'test_chars', \
                'token_model', 'char_model', 'hybrid_embedding_model', 'hybrid_model_history_file', 'triple_model_history_file', 'triple_input_model', \
                'steps_per_epoch', 'epoch'

    def __init__(self):
        # self.data_dir = "small_data" + os.path.sep
        self.data_dir = "large_data" + os.path.sep
        self.token_model_history_file = 'model1_history_file1'
        self.hybrid_model_history_file = 'hybrid_model_history_file'
        self.triple_model_history_file = 'token_position_model_history_file'
        self.sw_nltk = stopwords.words('english')
        self.text_vectorizer = None
        self.char_vectorizer = None
        self.data_facter = 1
        self.steps_per_epoch = 0.033
        self.epoch = 1

    def build(self):
        self.parse_files()
        self.one_hot_label()
        self.numeric_encoded_label()
        self.create_vectorization()
        self.create_token_model()
        self.create_character_model()
        self.create_hybrid_embedding_model()
        self.create_triple_input_model()

    def read_file(self, file_name):
        """
        :param file_name: file path
        :return: a list of maps, the list members represent the sentences.
        """
        lens = []

        f = open(file_name, 'r')
        input_lines = f.readlines()
        abstract_lines = []
        abstract_data = []
        for line in input_lines:
            if line.startswith("###"):
                if abstract_lines != []:
                    lens.append(len(abstract_lines))
                abstract_lines = []
            elif line.isspace():
                for abstract_line_number, abstract_line in enumerate(abstract_lines):
                    line_data = {}
                    target_text_split = abstract_line.split("\t")
                    line_data["label"] = target_text_split[0]
                    line_data["text"] = target_text_split[1].lower()
                    line_data[
                        "line_number"] = abstract_line_number
                    line_data["total_lines"] = len(
                        abstract_lines) - 1
                    abstract_data.append(line_data)

            else:
                abstract_lines.append(line)

        if file_name.count( 'train')  > 0:
            print('======================', file_name)
            # self.analyze_sentence_per_abstract(lens)
        return abstract_data * self.data_facter

    def analyze_sentence_per_abstract(self, data1):
        print('analyze_sentence_per_abstract')

        lens1 = [i for i in data1 if i <= 30]
        font = {'size': 20}
        plt.figure(figsize=(11, 9))
        matplotlib.rc('font', **font)
        # plt.rcParams['figure.dpi'] = 300
        plt.hist(lens1, len(set(lens1)))

        # data = {}
        # for i in range(30):
        #     data[i] = 0
        #
        # for len1 in lens1:
        #     if len1 in data:
        #         data[len1] += 1
        #
        # courses = list(data.keys())
        # values = list(data.values())
        # plt.bar(courses, values,width=0.4)

        plt.xlabel("Number of sentences per abstract")
        plt.ylabel("Number of abstracts")
        plt.savefig('test.png')
        plt.show()
        plt.close()

    def analyze_sentence_length(self, data1):
        import matplotlib.pyplot as plt
        lens = [len(sentence.split())  for sentence in data1]

        lens1 = [i for i in lens if i <= 30]

        font = {'size': 14}
        # plt.figure(figsize=(10, 10))
        matplotlib.rc('font', **font)
        plt.rcParams['figure.dpi'] = 300
        plt.hist(lens1, len(set(lens1)))
        plt.xlabel("Number of characters per sentence")
        plt.ylabel("Number of sentences")
        plt.savefig('test.png')
        plt.show()
        plt.close()

    def parse_files(self):
        """
        parse the files, create pandas frames
        :return:
        """
        train_data = self.read_file(self.data_dir + "train.txt")
        # self.analyze_sentence_length(pd.DataFrame(train_data)['text'].tolist())

        train_data = self.balance_data(train_data)

        val_data = self.read_file(self.data_dir + "val.txt")
        test_data = self.read_file(self.data_dir + "test.txt")
        print('test data length', len(test_data))

        self.train_df = pd.DataFrame(train_data)
        self.val_df = pd.DataFrame(val_data)
        self.test_df = pd.DataFrame(test_data)
        print(self.train_df.label.value_counts())

        self.train_sentences = self.train_df["text"].tolist()
        self.val_sentences = self.val_df["text"].tolist()
        self.test_sentences = self.test_df["text"].tolist()
        print(len(self.train_sentences), len(self.val_sentences), len(self.test_sentences))

    def balance_data(self, data):
        label_map = {}
        for line in data:
            if line["label"] not in label_map:
                label_map[line["label"]] = []
            else:
                label_map[line["label"]].append(line)
        labels = label_map.keys()
        total_count = sum([len(label_map[t]) for t in labels])
        max_count = max([len(label_map[t]) for t in labels]) * 0.26
        new_data = []
        for label in labels:
            print(label, len(label_map[label]))
            ratio = max_count / len(label_map[label])
            # new_data += (label_map[label] * int(ratio))
            new_data += label_map[label][0: int((1000000 / total_count) * len(label_map[label]))]
            # new_data += label_map[label][0: int((ratio - int(ratio)) * len(label_map[label]))]
        random.shuffle(new_data)
        return new_data

    def one_hot_label(self):
        """
        create one hot encoded labels
        :return:
        """
        one_hot_encoder = OneHotEncoder(sparse=False)
        self.train_labels_one_hot = one_hot_encoder.fit_transform(self.train_df["label"].to_numpy().reshape(-1, 1))
        self.val_labels_one_hot = one_hot_encoder.transform(self.val_df["label"].to_numpy().reshape(-1, 1))
        self.test_labels_one_hot = one_hot_encoder.transform(self.test_df["label"].to_numpy().reshape(-1, 1))

    def numeric_encoded_label(self):
        """
        create numeric encoded labels
        :return:
        """
        label_encoder = LabelEncoder()
        self.train_labels_encoded = label_encoder.fit_transform(self.train_df["label"].to_numpy())
        self.val_labels_encoded = label_encoder.transform(self.val_df["label"].to_numpy())
        self.test_labels_encoded = label_encoder.transform(self.test_df["label"].to_numpy())

        self.num_classes = len(label_encoder.classes_)
        self.class_names = label_encoder.classes_

        plt.close()

    def get_vectorization(self, data, file_name, max_tokens, output_seq_len):
        start = time.time()

        if os.path.exists(file_name):
            print('found vectorization')
            from_disk = pickle.load(open(file_name, "rb"))
            text_vectorizer = TextVectorization.from_config(from_disk['config'])
            print('config', from_disk['config'])
            from_disk['config']['output_sequence_length'] = output_seq_len
            pickle.dump({'config': from_disk['config'],
                         'weights': from_disk['weights']}
                        , open(file_name, "wb"))
            text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
            text_vectorizer.set_weights(from_disk['weights'])

        else:
            print('vectorizing')
            print(file_name, 'not exist')
            text_vectorizer = TextVectorization(
                max_tokens=max_tokens,
                output_sequence_length=output_seq_len)
            text_vectorizer.adapt(data)

            pickle.dump({'config': text_vectorizer.get_config(),
                         'weights': text_vectorizer.get_weights()}
                        , open(file_name, "wb"))
        end = time.time()
        print('time:     ', end - start)
        return text_vectorizer

    def create_vectorization(self):
        if self.text_vectorizer is None:
            print('creating text vectorization')
            sent_lens = [len(sentence.split()) for sentence in self.train_sentences]
            avg_sent_len = np.mean(sent_lens)
            print('avg_sent_len', avg_sent_len)  # return average sentence length (in tokens)
            # plt.hist(sent_lens, bins=7)
            self.text_output_seq_len = int(np.percentile(sent_lens, 95))
            print('output_seq_len', self.text_output_seq_len)
            max_tokens = 68000
            self.text_vectorizer = self.get_vectorization(self.train_sentences,
                                                          self.data_dir.strip(os.path.sep) + '_file',
                                                          max_tokens, self.text_output_seq_len)

            self.text_vocab = self.text_vectorizer.get_vocabulary()

        if self.char_vectorizer is None:
            self.train_chars = [" ".join(list(sentence)) for sentence in self.train_sentences]
            self.val_chars = [" ".join(list(sentence)) for sentence in self.val_sentences]
            self.test_chars = [" ".join(list(sentence)) for sentence in self.test_sentences]
            char_lens = [len(sentence) for sentence in self.train_sentences]
            mean_char_len = np.mean(char_lens)
            self.output_seq_char_len = int(np.percentile(char_lens, 95))
            alphabet = string.ascii_lowercase + string.digits + string.punctuation
            self.num_char_tokens = len(alphabet) + 2
            print('output_seq_char_len', self.output_seq_char_len, 'num_char_tokens', self.num_char_tokens)

            # self.char_vectorizer = self.get_vectorization(self.train_chars,
            #                                               self.data_dir.strip(os.path.sep) + 'char_file',
            #                                               self.num_char_tokens, self.output_seq_char_len)

            self.char_vectorizer = self.get_vectorization(self.train_chars,
                                                          self.data_dir.strip(os.path.sep) + 'char_file',
                                                          self.num_char_tokens, self.output_seq_char_len)

    def create_token_model(self):
        self.token_model = tf.keras.models.Sequential([
            self.text_vectorizer,
            layers.Embedding(input_dim=len(self.text_vocab),  # length of vocabulary
                             output_dim=200,
                             name="token_embed"
                             ),
            layers.Bidirectional(layers.LSTM(80, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(80, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(80)),

        ])

        self.token_model.compile(loss="categorical_crossentropy",
                                 optimizer=tf.keras.optimizers.Adam(),
                                 metrics=["accuracy"])

    def create_character_model(self):
        char_embed = layers.Embedding(input_dim=self.num_char_tokens,  # number of different characters
                                      output_dim=256,
                                      mask_zero=False,
                                      name="char_embed")

        self.char_model = tf.keras.models.Sequential([
            self.char_vectorizer,
            char_embed,

            layers.Bidirectional(layers.LSTM(25, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(25, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(25)),
        ])

        # Compile model
        self.char_model.compile(loss="categorical_crossentropy",
                                optimizer=tf.keras.optimizers.Adam(),
                                metrics=["accuracy"])

    def create_hybrid_embedding_model(self):

        token_char_concat = layers.Concatenate()([self.token_model.output, self.char_model.output])

        combined_dropout = layers.Dropout(0.5)(token_char_concat)
        # combined_dense = layers.Bidirectional(layers.LSTM(80, return_sequences=True))(
        #     combined_dropout)
        # combined_dense = layers.Bidirectional(layers.LSTM(80, return_sequences=True))(
        #     combined_dense)
        # combined_dense = layers.Bidirectional(layers.LSTM(80))(
        #     combined_dense)
        combined_dense = layers.Dense(80)(combined_dropout)
        combined_dense = layers.Dense(80)(combined_dense)
        combined_dense = layers.Dense(80)(combined_dense)

        final_dropout = layers.Dropout(0.5)(combined_dense)
        output_layer = layers.Dense(5, activation="softmax")(final_dropout)
        self.hybrid_embedding_model = tf.keras.Model(inputs=[self.token_model.input, self.char_model.input],
                                                     outputs=output_layer,
                                                     name="model_4_token_and_char_embeddings")

    def create_triple_input_model(self):
        # 3. Line numbers inputs
        line_number_inputs = layers.Input(shape=(15,), dtype=tf.int32, name="line_number_input")
        x = layers.Dense(32, activation="relu")(line_number_inputs)
        line_number_model = tf.keras.Model(inputs=line_number_inputs,
                                           outputs=x)

        # 4. Total lines inputs
        total_lines_inputs = layers.Input(shape=(20,), dtype=tf.int32, name="total_lines_input")
        y = layers.Dense(32, activation="relu")(total_lines_inputs)
        total_line_model = tf.keras.Model(inputs=total_lines_inputs,
                                          outputs=y)

        # 5. Combine token and char embeddings into a hybrid embedding
        combined_embeddings = layers.Concatenate(name="token_char_hybrid_embedding")([self.token_model.output,
                                                                                      self.char_model.output])
        z = layers.Dense(168, activation="relu")(self.token_model.output)
        z = layers.Dense(168, activation="relu")(z)
        z = layers.Dense(168, activation="relu")(z)

        # 6. Combine positional embeddings with combined token and char embeddings into a tribrid embedding
        z = layers.Concatenate(name="token_char_positional_embedding")([line_number_model.output,
                                                                        total_line_model.output,
                                                                        z])
        z = layers.Dense(128, activation="relu")(z)
        z = layers.Dense(128, activation="relu")(z)
        z = layers.Dense(128, activation="relu")(z)
        z = layers.Dense(128, activation="relu")(z)

        # 7. Create output layer
        output_layer = layers.Dense(5, activation="softmax", name="output_layer")(z)

        # 8. Put together model
        self.triple_input_model = tf.keras.Model(inputs=[line_number_model.input,
                                                         total_line_model.input,
                                                         self.token_model.input,
                                                         # self.char_model.input
                                                         ],
                                                 outputs=output_layer)

        self.triple_input_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                                        # add label smoothing (examples which are really confident get smoothed a little)
                                        optimizer=tf.keras.optimizers.Adam(),
                                        metrics=["accuracy"])

    def train_token_model(self):
        self.token_model.add(layers.Dense(5, activation="softmax"))
        model_path = 'token_model1_' + self.data_dir
        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_sentences, self.train_labels_one_hot))
        valid_dataset = tf.data.Dataset.from_tensor_slices((self.val_sentences, self.val_labels_one_hot))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.test_sentences, self.test_labels_one_hot))

        train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        if os.path.exists(model_path):
            print('found model', model_path)
            self.token_model.load_weights(model_path)
        else:
            print('training model', model_path)
            model_1_history = self.token_model.fit(train_dataset,
                                                   steps_per_epoch=int(self.steps_per_epoch * len(train_dataset)),
                                                   epochs=self.epoch,
                                                   validation_data=valid_dataset,
                                                   validation_steps=int(
                                                       self.steps_per_epoch * len(valid_dataset)))

            self.token_model.summary()
            with open(self.token_model_history_file, 'wb') as f:
                pickle.dump(model_1_history.history, f)
            self.token_model.save(model_path)

        with open(self.token_model_history_file, 'rb') as f:
            history = pickle.load(f)
            self.plot(history, model_path)
        self.token_model.evaluate(valid_dataset)
        self.test(self.token_model, test_dataset, self.test_labels_encoded, self.test_df["label"].to_numpy(),
                  'model1-test')

    def train_hybrid_embedding_model(self):
        model_path = 'hybrid_embedding2_model_' + self.data_dir.strip('\\')
        self.token_model.pop()
        self.char_model.pop()

        self.hybrid_embedding_model.compile(loss="categorical_crossentropy",
                                            optimizer=tf.keras.optimizers.Adam(),
                                            metrics=["accuracy"])
        train_char_token_data = tf.data.Dataset.from_tensor_slices(
            (self.train_sentences, self.train_chars))
        train_char_token_labels = tf.data.Dataset.from_tensor_slices(self.train_labels_one_hot)
        train_char_token_dataset = tf.data.Dataset.zip(
            (train_char_token_data, train_char_token_labels))

        train_char_token_dataset = train_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        val_char_token_data = tf.data.Dataset.from_tensor_slices((self.val_sentences, self.val_chars))
        val_char_token_labels = tf.data.Dataset.from_tensor_slices(self.val_labels_one_hot)
        val_char_token_dataset = tf.data.Dataset.zip((val_char_token_data, val_char_token_labels))
        val_char_token_dataset = val_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        test_char_token_data = tf.data.Dataset.from_tensor_slices((self.test_sentences, self.test_chars))
        test_char_token_labels = tf.data.Dataset.from_tensor_slices(self.test_labels_one_hot)
        test_char_token_dataset = tf.data.Dataset.zip((test_char_token_data, test_char_token_labels))
        test_char_token_dataset = test_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

        if os.path.exists(model_path):
            print('found model', model_path)
            self.hybrid_embedding_model.load_weights(model_path)
        else:
            print('training model', model_path)
            plot_model(self.hybrid_embedding_model, to_file='hybrid_model.png', show_shapes=True,
                       show_layer_names=True)
            model_history = self.hybrid_embedding_model.fit(train_char_token_dataset,
                                                            steps_per_epoch=int(
                                                                self.steps_per_epoch * len(train_char_token_dataset)),
                                                            epochs=self.epoch,
                                                            validation_data=val_char_token_dataset,
                                                            validation_steps=int(
                                                                self.steps_per_epoch * len(val_char_token_dataset)))

            with open(self.hybrid_model_history_file, 'wb') as f:
                pickle.dump(model_history.history, f)
            self.hybrid_embedding_model.save(model_path)
        self.hybrid_embedding_model.summary()

        with open(self.hybrid_model_history_file, 'rb') as f:
            history = pickle.load(f)
            self.plot(history, model_path)

        self.test(self.hybrid_embedding_model, test_char_token_dataset, self.test_labels_encoded,
                  self.test_df["label"].to_numpy(),
                  model_path)

    def train_triple_input_model(self):
        # model_path = 'triple_embedding_model_' + self.data_dir.strip('\\')
        model_path = 'token_position4_embedding_model_' + self.data_dir.strip('\\')
        train_line_numbers_one_hot = tf.one_hot(self.train_df["line_number"].to_numpy(), depth=15)
        val_line_numbers_one_hot = tf.one_hot(self.val_df["line_number"].to_numpy(), depth=15)
        test_line_numbers_one_hot = tf.one_hot(self.test_df["line_number"].to_numpy(), depth=15)

        train_total_lines_one_hot = tf.one_hot(self.train_df["total_lines"].to_numpy(), depth=20)
        val_total_lines_one_hot = tf.one_hot(self.val_df["total_lines"].to_numpy(), depth=20)
        test_total_lines_one_hot = tf.one_hot(self.test_df["total_lines"].to_numpy(), depth=20)

        train_pos_char_token_data = tf.data.Dataset.from_tensor_slices((train_line_numbers_one_hot,  # line numbers
                                                                        train_total_lines_one_hot,  # total lines
                                                                        self.train_sentences,  # train tokens
                                                                        # self.train_chars
                                                                        ))  # train chars
        train_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(self.train_labels_one_hot)  # train labels
        train_pos_char_token_dataset = tf.data.Dataset.zip(
            (train_pos_char_token_data, train_pos_char_token_labels))  # combine data and labels
        train_pos_char_token_dataset = train_pos_char_token_dataset.batch(32).prefetch(
            tf.data.AUTOTUNE)  # turn into batches and prefetch appropriately

        # Validation dataset
        val_pos_char_token_data = tf.data.Dataset.from_tensor_slices((val_line_numbers_one_hot,
                                                                      val_total_lines_one_hot,
                                                                      self.val_sentences,
                                                                      # self.val_chars
                                                                      ))
        val_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(self.val_labels_one_hot)
        val_pos_char_token_dataset = tf.data.Dataset.zip((val_pos_char_token_data, val_pos_char_token_labels))
        val_pos_char_token_dataset = val_pos_char_token_dataset.batch(32).prefetch(
            tf.data.AUTOTUNE)  # turn into batches and prefetch appropriately

        # Validation dataset
        test_pos_char_token_data = tf.data.Dataset.from_tensor_slices((test_line_numbers_one_hot,
                                                                       test_total_lines_one_hot,
                                                                       self.test_sentences,
                                                                       # self.test_chars
                                                                       ))
        test_pos_char_token_labels = tf.data.Dataset.from_tensor_slices(self.test_labels_one_hot)
        test_pos_char_token_dataset = tf.data.Dataset.zip((test_pos_char_token_data, test_pos_char_token_labels))
        test_pos_char_token_dataset = test_pos_char_token_dataset.batch(32).prefetch(
            tf.data.AUTOTUNE)  # turn into batches and prefetch appropriately

        if os.path.exists(model_path):
            print('found model', model_path)
            self.triple_input_model.load_weights(model_path)

            plot_model(self.triple_input_model, to_file='triple_embedding_model.png', show_shapes=True,
                       show_layer_names=True)
        else:
            print('training model', model_path)
            plot_model(self.triple_input_model, to_file='triple_embedding_model.png', show_shapes=True,
                       show_layer_names=True)
            model_history = self.triple_input_model.fit(train_pos_char_token_dataset,
                                                        steps_per_epoch=int(
                                                            self.steps_per_epoch * len(train_pos_char_token_dataset)),
                                                        epochs=self.epoch,
                                                        validation_data=val_pos_char_token_dataset,
                                                        validation_steps=int(
                                                            self.steps_per_epoch * len(val_pos_char_token_dataset)))

            plot_model(self.triple_input_model, to_file='triple_embedding_model.png', show_shapes=True,
                       show_layer_names=True)

            with open(self.triple_model_history_file, 'wb') as f:
                pickle.dump(model_history.history, f)
            self.triple_input_model.save(model_path)
        with open(self.triple_model_history_file, 'rb') as f:
            history = pickle.load(f)
            self.plot(history, model_path)

        self.triple_input_model.summary()

        self.test(self.triple_input_model, test_pos_char_token_dataset, self.test_labels_encoded,
                  self.test_df["label"].to_numpy(),
                  model_path)

    def test(self, model, x, y, word_label, name):
        pred_probs = model.predict(x)
        preds = tf.argmax(pred_probs, axis=1)
        index_name = {}
        for i in range(len(y)):
            if len(index_name) == 5:
                break
            key1 = int(y[i])
            if key1 not in index_name:
                index_name[key1] = word_label[i]

        self.plot_confusion_matrix(word_label, [index_name[int(i)] for i in preds.numpy()], name)

        results = calculate_results(y_true=y, y_pred=preds)
        print('testing ', results)
        return results

    def plot(self, history, name):
        print('plot accuracy and loss', name + '-accuracy.jpg', name + '-loss.jpg')
        plt.close()
        train_acc = history['accuracy']
        plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
        val_acc = history['val_accuracy']
        plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
        plt.ylim([0.5, 0.95])

        plt.title(name[0: name.index('model') + 5].replace('_', ' ') + ' accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(name + '-accuracy.jpg')
        plt.close()

        train_loss = history['loss']
        plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
        val_loss = history['val_loss']
        plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Accuracy')
        plt.title(name[0:name.index('model') + 5].replace('_', ' ') + ' loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(name + '-loss.jpg')
        plt.close()

    def plot_confusion_matrix(self, true_label, predict_label, name):
        print('plot confusion matrix')
        cmp = ConfusionMatrixDisplay.from_predictions(true_label,
                                                      predict_label)
        font = {'size': 14}
        # plt.figure(figsize=(10, 10))
        matplotlib.rc('font', **font)

        fig, ax = plt.subplots(figsize=(10, 10))
        cmp.plot(ax=ax)
        plt.title('Confusion Matrix')
        plt.savefig(name + '-confusion-matrix.jpg')
        plt.close()
        report = classification_report(np.array(true_label), np.array(predict_label))
        with open(name + '-report', 'w') as f:
            f.write('\n')
            f.write(report)


if __name__ == '__main__':
    '''
    Epoch 19/20
    281/281 [==============================] - 4s 14ms/step - loss: 0.5355 - accuracy: 0.8097 - val_loss: 0.5061 - val_accuracy: 0.8195
    Epoch 20/20
    281/281 [==============================] - 4s 14ms/step - loss: 0.5065 - accuracy: 0.8175 - val_loss: 0.5109 - val_accuracy: 0.8178
     loss: 0.5109 - accuracy: 0.8178
    '''

    data_loader = Data_loader()
    data_loader.build()
    data_loader.train_triple_input_model()


