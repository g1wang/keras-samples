from keras.preprocessing.text import Tokenizer

samples = ['The sat on the nat a.','The a dog ate my a homework a a.']

#分词器
tokenizer = Tokenizer(num_words=4)
tokenizer.fit_on_texts(samples)
print(tokenizer.word_index)
sequences = tokenizer.texts_to_sequences(samples)

print(sequences)
one_hot_results = tokenizer.texts_to_matrix(samples,mode='binary')
print(one_hot_results)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))