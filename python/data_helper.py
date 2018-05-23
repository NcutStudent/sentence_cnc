from python.util import *
class Data_Helper:
    def __init__(self, train_file_path, label_file_path, dictionary_file_path, encoding='utf-8'):
        self.train_file     = open(train_file_path, 'r', encoding=encoding)
        self.label_file     = open(label_file_path, 'r', encoding=encoding)
        dictionary_file     = open(dictionary_file_path, 'r', encoding=encoding)

        print('work with dictionary...')
        self.dictionary = {}
        count = 0
        str_len = 0

        empty_word = dictionary_file.readline()
        if empty_word == '':
            print("this is a empty file")

        data = empty_word.split(' ')
        if len(data) != 2:
            raise ValueError('this may not a dictionary file')
        self.dictionary[data[0]] = int(data[1])

        self.empty_word    = data[0]
        self.empty_word_id = int(data[1])
            
        for word in dictionary_file:
            print('\b' * str_len, end='')

            data = word.split(' ')
            if len(data) != 2:
                raise ValueError('this may not a dictionary file')

            self.dictionary[data[0]] = int(data[1])
            count += 1
            print(count, end='')
            str_len = len(str(count))
        print('\nwork finish')       
        
    def get_data(self, batch_size=10, pad_sequence_length=None):
        train_data_list = []
        label_data_list = []
        for i in range(batch_size):
            train_data = self.train_file.readline()
            if train_data == '':
                self.train_file.seek(0)
                self.label_file.seek(0)
                train_data = self.train_file.readline()

            label_data = self.label_file.readline()
            if label_data == '':
                print("Warning, label dosen't match")
                self.train_file.seek(0)
                self.label_file.seek(0)
                train_data = self.train_file.readline()
                label_data = self.label_file.readline()

            train_data = train_data[:-1].split(' ')
            label_data = label_data[:-1].split(' ')

            _train_word_list = []
            _label_word_list = []
            for word in train_data:
                if word == '':
                    continue
                if not word in self.dictionary:
                    _train_word_list.append(self.empty_word_id)
                else:
                    _train_word_list.append(self.dictionary[word])

            for label in label_data:
                if label == '':
                    continue
                try:
                    _label_word_list.append(int(label))
                except ValueError:
                    raise ValueError("label file format error, number only")

            if (not pad_sequence_length == None) and pad_sequence_length - len(_train_word_list) > 0:
                _train_word_list += ([self.empty_word_id] * (pad_sequence_length - len(_train_word_list)))

            train_data_list.append(_train_word_list)
            label_data_list.append(_label_word_list)

        return train_data_list, label_data_list

    def get_vocab_size(self):
        return len(self.dictionary)

    def string_to_input_data(self, input_string_list, pad_sequence_length=None):
        processed_list = []
        source_list = []
        for input_string in input_string_list:
            processed_string = process_string_with_jieba(input_string)
            word_list = []
            source_word_list = []
            for word in processed_string:
                if not word in self.dictionary:
                    word_list.append(self.empty_word_id)
                else:
                    word_list.append(self.dictionary[word])
                source_word_list.append(word)

            source_list.append(source_word_list)
            if not pad_sequence_length == None and pad_sequence_length - len(word_list) > 0:
                word_list += [self.empty_word_id] * (pad_sequence_length - len(word_list))

            processed_list.append(word_list)
        return source_list, processed_list
