import os
import markdown2

import json
import pickle
import random
from datetime import datetime, timedelta

import nltk
import numpy
import tensorflow
import tflearn
from nltk.stem.lancaster import LancasterStemmer

import langid

DIR_INTENTS = "files/intents.json"
DIR_PICKLE = "files/training_models/data.pickle"
DIR_MODEL = "files/training_models/model.tflearn"


class ChatBOT:
    def __init__(self, train=False, accuracy=0.9):
        self.accuracy = accuracy
        self.words = []
        self.labels = []
        self.docs_x = []
        self.docs_y = []
        self.training = []
        self.output = []
        self.model = None

        self.stemmer = LancasterStemmer()

        if os.path.isfile(DIR_INTENTS):
            with open(DIR_INTENTS) as file:
                self.data = json.load(file)

            if train:
                self.train()
            else:
                try:
                    self.load()
                except:
                    self.train()

    def load(self):
        with open(DIR_PICKLE, "rb") as f:
            self.words, self.labels, self.training, self.output = pickle.load(f)

        self._load_model()
        self.model.load(DIR_MODEL)

    def _load_model(self):
        tensorflow.compat.v1.reset_default_graph()

        net = tflearn.input_data(shape=[None, len(self.training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.output[0]), activation="softmax")
        net = tflearn.regression(net)

        self.model = tflearn.DNN(net)

    def train(self):
        for intent in self.data["intents"]["main"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                self.words.extend(wrds)
                self.docs_x.append(wrds)
                self.docs_y.append(intent["tag"])

            if intent["tag"] not in self.labels:
                self.labels.append(intent["tag"])

        self.words = [self.stemmer.stem(w.lower()) for w in self.words if w not in "?"]
        self.words = sorted(list(set(self.words)))

        self.labels = sorted(self.labels)

        out_empty = [0 for _ in range(len(self.labels))]

        for x, doc in enumerate(self.docs_x):
            bag = []

            wrds = [self.stemmer.stem(w) for w in doc]

            for w in self.words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[self.labels.index(self.docs_y[x])] = 1

            self.training.append(bag)
            self.output.append(output_row)

        self.training = numpy.array(self.training)
        self.output = numpy.array(self.output)

        ## Buatkan Folder nya
        folder = os.path.dirname(DIR_PICKLE)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(DIR_PICKLE, "wb") as f:
            pickle.dump((self.words, self.labels, self.training, self.output), f)

        self._load_model()

        self.model.fit(self.training, self.output, n_epoch=1000, batch_size=8, show_metric=True)
        self.model.save(DIR_MODEL)

    def bag_of_words(self, s, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1

        return numpy.array(bag)

    def _time_day_part(self, lang="en"):
        sekarang = datetime.now()
        jam = sekarang.hour

        if lang == 'id':
            if 4 <= jam < 11:
                return "pagi"
            elif 11 <= jam < 16:
                return "siang"
            elif 16 <= jam < 19:
                return "sore"
            else:
                return "malam"
        else:
            if 4 <= jam < 12:
                return "morning"
            elif 12 <= jam < 18:
                return "afternoon"
            else:
                return "evening"

    def _answer_fill(self, message, lang="id"):
        message = message.replace("[time_day_part]", self._time_day_part(lang))
        message = message.replace("[info_funfact]", random.choice(self.data["extra"]["info_funfact"][lang]))
        message = message.replace("[ref_job_experience]", self._load_file_md("files/reference/en/exp_job.md"))
        message = message.replace("[ref_study_experience]", self._load_file_md("files/reference/en/exp_study.md"))
        message = message.replace("[ref_project_experience]", self._load_file_md("files/reference/en/exp_project.md"))
        message = message.replace("[ref_code_lang]", self._load_file_md("files/reference/en/code_language.md"))
        message = message.replace("[info_myself]", self._load_file_md(f"files/reference/{lang}/info_myself.md"))
        return message
    
    def _language_detect(self, text):
        try:
            language, _ = langid.classify(text)
            return language
        except Exception as e:
            print(f"Error: {e}")
            return False
        
    def _load_file_md(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as md_file:
            # Membaca file Markdown
            md_text = md_file.read()
        # Konversi Markdown ke HTML menggunakan markdown2
        html_text = markdown2.markdown(md_text)
        
        return html_text

    def ask(self, message, need_accuracy=False):
        message = message.replace("your", "")
        message = message.replace("you", "")
        message = message.replace("harlan", "")
        message = message.replace("kamu", "")
        message = message.replace("kmu", "")
        message = message.replace("mu", "")
        message = message.replace("me", "")

        results = self.model.predict([self.bag_of_words(message, self.words)])[0]
        results_index = numpy.argmax(results)
        results_max = results[results_index]
        tag = self.labels[results_index]

        if results_max > self.accuracy:
            language = 'id'
            language_d = self._language_detect(message) 
            if language_d == 'en':
                language = 'en'

            for tg in self.data["intents"]["main"]:
                if tg['tag'] == tag:
                    responses = tg['responses'][language]

            if need_accuracy:
                return self._answer_fill(random.choice(responses),lang=language), float(results_max)
            else:
                return self._answer_fill(random.choice(responses),lang=language)

        messageUnknown = "Maaf saya tidak mengerti"
        if need_accuracy:
            return messageUnknown, float(0)
        else:
            return messageUnknown

    def _clean_text(self, text):
        text = text.replace("!", "")
        text = text.replace("?", "")
        return text.lower()

    def run_loop(self):
        print("Start talking with the bot!")
        while True:
            inp = input("You: ")
            if inp.lower() == "quit":
                break

            answer = self.ask(inp, True)
            print(f"Bot: {answer[0]} => {answer[1]}")

    # -------------------------- Extra --------------------------
    def _find_key(self, array, value):
        for key, val in array.items():
            if val.lower() == value.lower():
                return key
        return None

    def predict_extra(self, sentence, type):
        tag = {}
        sen_before = ""
        arr = self.str_to_array(sentence.lower())
        for key, item in enumerate(arr):
            for place_code in self.data["intents"]["extra"][type]:
                if f"{sen_before} {item}" in place_code['patterns'] or item in place_code['patterns']:
                    tag[key] = place_code['tag']
            sen_before = item
        return tag



    def predict_date(self, sentence):
        plus_day = 0
        data_place_find = self.predict_extra(sentence, "time")
        have_time = len(data_place_find)>0
        for key, item in data_place_find.items():
            if item.startswith('date'):
                arr = self.str_to_array(sentence.lower())
                if len(arr)-1 > key and arr[key+1].isdigit():
                    date = datetime.now()
                    date_new = int(arr[key+1])

                    if date.day > date_new:
                        date = date.replace(day=1)
                        if date.month == 12:
                            date = date.replace(month=1)
                            date = date.replace(year=date.year + 1)
                        else:
                            date = date.replace(month=date.month + 1)
                    date = date.replace(day=date_new)
                    return date.date(), have_time

            elif item.startswith('plus_'):
                plus_day = int(item.split('_')[1])

            elif item.startswith('day_'):
                day_offset = int(item.split('_')[1]) + 1
                today_weekday = datetime.now().weekday()
                # target_weekday = (today_weekday + day_offset) % 7
                target_weekday = day_offset - 1
                plus_day = (target_weekday - today_weekday) % 7
                if plus_day == 0:
                    plus_day = 7
                print("++++++++++++++++++", target_weekday, "=",today_weekday,"+", day_offset,"|", plus_day)

        date = datetime.now() + timedelta(days=plus_day)
        
        return date.date(), have_time

    def predict_rout(self, sentence):
        from_code = None
        to_code = None

        data_place_find = self.predict_extra(sentence, "place_code")
        data_range_find = self.predict_extra(sentence, "range")

        range_from = self._find_key(data_range_find, "from")
        range_to = self._find_key(data_range_find, "to")

        for key, item in data_place_find.items():
            if range_from is not None and range_from == key - 1:
                if from_code is None:
                    from_code = item
                    continue
                else:
                    to_code = from_code
                    from_code = item
                    continue
            if range_to is not None and range_to == key - 1:
                if to_code is None:
                    to_code = item
                    continue
                else:
                    from_code = to_code
                    to_code = item
                    continue

            if from_code is None:
                from_code = item
            elif to_code is None:
                to_code = item

        return from_code, to_code

    def predict_time(self, sentence):

        return None

    def str_to_array(self, sentence):
        return nltk.word_tokenize(sentence)

    # ========================== Extra ==========================


if __name__ == "__main__":
    bot = ChatBOT(False, 0.1)
    bot.run_loop()
