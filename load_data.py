import json
from nltk.corpus import stopwords
import joblib as ib
import pickle
def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    # borrowed this function from NeuralTalk
    print ('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold  and w not in stopwords.words('english')]
    print ('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    wordtoix = {}
    word_counts_new = {}
    for idx, w in enumerate(vocab):
        wordtoix[w] = idx
        ixtoword[idx] = w
        word_counts_new[w] = word_counts[w]
    '''
    word_counts['<pad>'] = nsents
    word_counts['<bos>'] = nsents
    word_counts['<eos>'] = nsents
    word_counts['<unk>'] = nsents
    '''
    return wordtoix, ixtoword, word_counts_new


def clean_string(string):
    return string.replace('.', '').replace(',', '').replace('"', '').replace('\n', '').replace('?', '').replace('!', '').replace('\\', '').replace('/', '')


def load_json(path):
    with open(path) as data_file:    
        path_file = json.load(data_file)
    return path_file



def load_caption(label_path, word_count_threshold = 5):
    train_label_dict={}    
    train_label = load_json(label_path)
    captions_corpus = []
    id_list = []
    for sample in train_label:
        id_list.append(sample['id'])
        cleaned_captions = [clean_string(sentence) for sentence in sample["caption"]]
        '''
        please put the functional word filter here
        '''
        #filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    
        captions_corpus += cleaned_captions
        train_label_dict[sample["id"]] = cleaned_captions
        #print(cleaned_captions)
    word2ix, ix2word, word_counts = preProBuildWordVocab(captions_corpus, word_count_threshold)
    return train_label_dict, word2ix, ix2word, word_counts
label_path = '/home/ubuntu/SCL/ADLxMLDS2017/hw2/MSRVTT_hw2_data/training_label.json'
word_count = 7
if __name__ == "__main__":
    trian_label, word2ix, ix2word, word_counts =  load_caption(label_path, word_count)
    ib.dump(ix2word, './ix2word.pkl')
    ib.dump(word2ix, './word2ix.pkl')
    ib.dump(word_counts, './word_counts.pkl')
    print(len(list(word2ix.keys())))
