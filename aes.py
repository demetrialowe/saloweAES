import nltk
#nltk.download('punkt')
import csv
import codecs
from sklearn.naive_bayes import MultinomialNB

'''
The essays are here:

https://docs.google.com/a/knox.edu/spreadsheets/d/1Vp1DScZXbBjxgyF7eSdUjYp3ZXmB5YAi1hOjFMwY9Qc/edit?usp=sharing

These essays are from a Kaggle contest from a few years ago. They were essays written
by students for an essay contest, I think in the UK. Each essay was scored by two 
different judges, rater1 and rater2.

The essays are also stored in the essays folder.

'''

def makewordindex(wordset):
    """
    Return a dictionary that maps each word from wordset to a unique index starting at 0
    and going up to N-1, where N is the len(wordset).
    """
    indexmap = {}
    sortwords = sorted(list(wordset))
    for i in range(len(sortwords)):
        word = sortwords[i]
        indexmap[word] = i
    return indexmap


def readessays(filename):
    # had to use codecs to open the file and ignore non-utf8 characters like the copyright symbol
    with codecs.open(filename, "r", encoding='ascii', errors='ignore') as csvfile:
        infile = csv.DictReader(csvfile, delimiter='\t', quotechar='"')
        wordlist = set()

        for row in infile:
            essay_id = row['essay_id']
            text = row['essay']
            score1 = row['rater1']
            score2 = row['rater2']
            totalscore = row['total']

            # using nltk (natural language toolkit, for natural language processing)
            # to help with parsing words out of each essay
            tokens = nltk.word_tokenize(text)
            text = nltk.Text(tokens)

            #print("essay {} has {} words".format(essay_id, len(text)))

            # use a regular foreach loop to go through text:
            for word in text:
                # print(word)
                if word not in wordlist:
                    wordlist.add(word)
    indexmap = makewordindex(wordlist)

    trainingdata = []
    labels = []
    with codecs.open(filename, "r", encoding='ascii', errors='ignore') as csvfile:
        infile = csv.DictReader(csvfile, delimiter='\t', quotechar='"')
        for row in infile:
            essay_id = row['essay_id']
            text = row['essay']
            score1 = row['rater1']
            score2 = row['rater2']
            totalscore = row['total']

            # using nltk (natural language toolkit, for natural language processing)
            # to help with parsing words out of each essay
            tokens = nltk.word_tokenize(text)
            text = nltk.Text(tokens)

            vector = [0] * len(indexmap)
            for word in text:
                index = indexmap[word]
                vector[index] += 1
            trainingdata.append(vector)
            labels.append(score1)

    clf = MultinomialNB()
    clf.fit(trainingdata,labels)

    testvectors = []
    testlabels = []
    with codecs.open('essays/validation1.tsv', "r", encoding='ascii', errors='ignore') as csvfile:
        infile = csv.DictReader(csvfile, delimiter='\t', quotechar='"')
        for row in infile:
            essay_id = row['essay_id']
            text = row['essay']
            score1 = row['rater1']
            score2 = row['rater2']
            totalscore = row['total']

            # using nltk (natural language toolkit, for natural language processing)
            # to help with parsing words out of each essay
            tokens = nltk.word_tokenize(text)
            text = nltk.Text(tokens)

            vector = [0] * len(indexmap)
            for word in text:
                if word not in indexmap:
                    continue
                index = indexmap[word]
                vector[index] += 1
            testvectors.append(vector)
            testlabels.append(score1)
    guess = clf.predict(testvectors)
    print(guess)
    print(testlabels)

    #get the % accuracy of the tests
    count = 0
    for index in range(len(guess)):
        if guess[index] == testlabels[index]:
            count += 1
    percentacc = float(count) / float(len(guess))
    print 100* percentacc, "% Accuracy"
if __name__ == '__main__':
    readessays('essays/essays1.tsv')
