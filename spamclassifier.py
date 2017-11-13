from sklearn.naive_bayes import MultinomialNB
import os
import codecs

"""
testspam() runs the Enron spam data set
"""


def wordifyfile(filename):
    words = []
    # had to use codecs to open the file and ignore non-utf8 characters like the copyright symbol
    # https://stackoverflow.com/questions/12468179/unicodedecodeerror-utf8-codec-cant-decode-byte-0x9c
    with codecs.open(filename, "r", encoding='utf-8', errors='ignore') as file:
        for line in file:
            line = line.rstrip()
            for word in line.split():
                words.append(word)
        return words


def filepaths(folder):
    """
    Return a list of paths to all the files contained in the given folder.
    """
    paths = []
    for f in os.listdir(folder):
        paths.append(os.path.join(folder, f))
    return paths


def readwordset(wordlist, wordset):
    """
    Add the words from the list wordlist into the given set wordset
    """
    for word in wordlist:
        wordset.add(word)


def readwordsetfolder(folder, wordset):
    """
    Return a set of all of the words contained in all of the files
    in the given folder.
    """
    for filename in filepaths(folder):
        readwordset(wordifyfile(filename), wordset)


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


def makevector(wordlist, windex):
    """
    Return a "feature vector" of the given wordlist, where each word in wordlist
    increments the count for its corresponding index in the feature vector.
    We use windex (short for word index) to decide the index for each word.
    """
    # really cool way of creating a list of a given number of zeroes
    vector = [0] * len(windex)
    for word in wordlist:
        # We ignore words we haven't seen.
        # This is a weakness of this approach; we have to ignore words we have never
        # seen before. To use this approach, we either need to update windex and retrain the
        # feature vector, or re-compute Bayes theorem directly for each new message.
        if word not in windex:
            continue
        index = windex[word]
        vector[index] += 1
    return vector


def makevectorfolder(folder, windex):
    """
    Return a list of feature vectors, one for each message in the given folder.
    Use the given windex (wordindex) for the indexes of each word in the
    feature vector.
    """
    vectorlist = []
    for filename in filepaths(folder):
        vector = makevector(wordifyfile(filename), windex)
        vectorlist.append(vector)
    return vectorlist


def testspam():
    # read each word from the spam and ham folders into a set of words
    wordset = set()
    readwordsetfolder('enron1/spam', wordset)
    readwordsetfolder('enron1/ham', wordset)

    # create the word index from the words we have found
    windex = makewordindex(wordset)

    # create the feature vectors
    spam = makevectorfolder('enron1/spam', windex)
    ham = makevectorfolder('enron1/ham', windex)

    # now stick the two feature vectors together to produce the training data
    # you can use + with two lists in Python
    trainingdata = spam + ham
    # produce the labels
    labels = ['spam'] * len(spam) + ['ham'] * len(ham)

    # load the feature vectors for the 4 files we will test on
    tests = makevectorfolder('enron1/hamtest', windex) + makevectorfolder('enron1/spamtest', windex)

    # create the classifier
    clf = MultinomialNB()

    # fit the training data to the labels
    clf.fit(trainingdata, labels)

    # now predict the labels for the test data
    # this part should print ['ham' 'ham' 'spam' 'spam']
    print(clf.predict(tests))


if __name__ == '__main__':
    testspam()

