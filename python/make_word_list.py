import re
import jieba

jieba.load_userdict("../data/user_direct.txt")

data_dir = '../data/'

f = open(data_dir + 'sentence.txt', 'r', encoding='utf-8')
o = open(data_dir + 'word_list.txt', 'w', encoding='utf-8')
view = open(data_dir + 'out_sentence.txt', 'w', encoding='utf-8')

word_set = set()

for l in f:
    arr = re.sub("([\w\W\u4e00-\u9fff]-)", "", l) #chinese, english, and number
    arr = re.sub("[\u3002\uff1b\uff0c\uff1a\u201c\u201d\uff08\uff09\u3001\uff1f\u300a\u300b\n]", "", arr) # punctuation
    no_num_str = re.sub("[\d]+", "NUMBERFLAG", arr)
    out = jieba.cut(arr)
    no_num_out = jieba.cut(no_num_str)
    for s in no_num_out :
        view.write(s + " ")
        word_set.add(s)
    view.write('\n')

i = 3
o.write("INKNOWFLAG 0\n")
o.write("EOSFLAG 1\n")
o.write("PADFLAG 2\n")
for s in word_set:
    o.write(s + " " + str(i) + '\n')
    i += 1
