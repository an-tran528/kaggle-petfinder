#Extract text features using fastText

def fastText_vectorize(X,col,textfile,dimension):
    clf = ft.train_unsupervised(input=textfile, \
        ws = 5, minCount=10, epoch=10, \
        minn = 3, maxn = 6, wordNgrams = 3, \
        dim=dimension,thread=1)

    fastText_text_features = []
    for description in col:
        #print(description)
        fastText_text_features.append(clf.get_sentence_vector(description))

    fastText_text_features = pd.DataFrame(fastText_text_features)
    fastText_text_features = fastText_text_features.add_prefix('fastText_Description_'.format(i))

    return fastText_text_features