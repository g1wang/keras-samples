from keras.preprocessing.text import Tokenizer

samples = ['The is sat on the nat.','The dog ate my homework.']

#分词器
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)

sequences = tokenizer.texts_to_sequences(samples)

print(sequences)
one_hot_results = tokenizer.texts_to_matrix(samples,mode='binary')
print(one_hot_results)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))