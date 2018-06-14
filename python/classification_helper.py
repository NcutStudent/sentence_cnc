from python.util import *
import os
class Helper:
    def __init__(self, train_file_path, dictionary, encoding='utf-8'):
        self.train_file     = open(train_file_path, 'r', encoding=encoding)
        self.dictionary = dictionary
        
    def get_data(self, batch_size=10, pad_sequence_length=None):
        train_data_list = []
        for i in range(batch_size):
            train_data = self.train_file.readline()
            if train_data == '':
                self.train_file.seek(0)
                train_data = self.train_file.readline()

            train_data = train_data[:-1].split(' ')

            _train_word_list = []
            for word in train_data:
                if word == '':
                    continue
                if not word in self.dictionary:
                    _train_word_list.append(Constance.EMPTY_WORD_ID)
                else:
                    _train_word_list.append(self.dictionary[word])

            _train_word_list.append(Constance.EOS_WORD_ID)

            if (not pad_sequence_length == None) and pad_sequence_length - len(_train_word_list) > 0:
                _train_word_list += ([Constance.PAD_WORD_ID] * (pad_sequence_length - len(_train_word_list)))

            train_data_list.append(_train_word_list)

        return train_data_list

class Data_Helper:
    def __init__(self, train_file_path, dictionary_file_path, encoding='utf-8'):
        dictionary_file     = open(dictionary_file_path, 'r', encoding=encoding)

        print('work with dictionary...')
        self.dictionary = {}
        count = 0
        str_len = 0

        Constance.EMPTY_WORD_ID = 0
        Constance.EOS_WORD_ID   = 1
        Constance.PAD_WORD_ID   = 2

        i = 3
        for word in dictionary_file:
            print('\b' * str_len, end='')

            self.dictionary[word[:-1]] = i
            count += 1
            print(count, end='')
            str_len = len(str(count))
            i += 1
        print('\nwork finish\n')

        print("list train files")
        if not os.path.isdir(train_file_path):
            raise IOError("this is not a folder")

        index = 0
        train_file_path += '/'
        self.helper_list = []
        self.class_map = []
        for f in os.listdir(train_file_path):
            print(f + ' ' + str(index))
            self.class_map.append(f)
            self.helper_list.append((Helper(train_file_path + f, self.dictionary), index))
            index += 1
        print('init finish')

    def get_data(self, batch_size=10, pad_sequence_length=None):
        mini_batch = batch_size // len(self.helper_list)
        if mini_batch == 0:
            mini_batch = 1

        train_data = []
        label_data = []
        for helper, helper_id in self.helper_list:
            train_data += helper.get_data(mini_batch, pad_sequence_length)
            label_data += [helper_id] * mini_batch
        
        return train_data, label_data

    def get_vocab_size(self):
        return len(self.dictionary) + 3

    def get_class_count(self):
        return len(self.helper_list)

    def get_class_name(self, index):
        return self.class_map[index]

    def string_to_input_data(self, input_string_list, pad_sequence_length=None):
        processed_list = []
        source_list = []
        for input_string in input_string_list:
            processed_string = process_string_with_jieba(input_string)
            word_list = []
            source_word_list = []
            for word in processed_string:
                if not word in self.dictionary:
                    word_list.append(Constance.EMPTY_WORD_ID)
                else:
                    word_list.append(self.dictionary[word])
                source_word_list.append(word)

            word_list.append(Constance.EOS_WORD_ID)

            source_list.append(source_word_list)
            if not pad_sequence_length == None and pad_sequence_length - len(word_list) > 0:
                word_list += [Constance.PAD_WORD_ID] * (pad_sequence_length - len(word_list))

            processed_list.append(word_list)
        return source_list, processed_list
