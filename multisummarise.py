"multisummarise.py - Multi-document summarisation"

from xml.dom.minidom import parse
import os
import sys
import codecs
from nltk import sent_tokenize
import scipy.sparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

import summarise
from xml_abstract_retriever import getAbstract, getText

TEST_QUESTIONS_PATH = os.path.join("src", "pubmed", "devtestset-questions.txt")
ABSTRACTS_PATH = os.path.join("..", "..", "dev")
CORPUS_PATH = os.path.join("dev", "ClinicalInquiries.xml")

sys.path.append('/code/dev/')
sys.path.append('../../dev/')
sys.path.append(os.path.dirname(__file__))


def getFilePath(filename):
    module_dir = os.path.dirname(__file__)  # get current directory
    return os.path.join(module_dir, filename)


#with open(getFilePath('pickles/regression_multisummariser.pickle'), 'rb') as f:
#    (tfidf, regression) = pickle.load(f, encoding="ISO-8859-1")


def defaultsummarise(question, docstext, n=2):
    "Default summariser"
    return rtwostepsummarise(question, docstext, n)


def singlesummarise(question, docstext, n=2):
    """Return the output of the default single-document summariser
    >>> question = "my sentence"
    >>> text = ["This is one sentence.",\
                "This is another sentence.",\
                "This is yet another sentence.",\
                "And this is the final sentence."]
    >>> singlesummarise(question,text,n=2)
    ['This is another sentence.', 'This is yet another sentence.']
"""
    alltext = " ".join([t for t in docstext])
    if len(alltext.strip()) == 0:
        return '[No text to summarise]'
    offsets = summarise.summarise(question, alltext, n)
    if len(offsets) == 0:
        return '[No summary text found]'
    return [alltext[b:e] for ((b, e), score) in offsets]


def rsummarise(question, docstext, n=2, sentTokenised=False):
    """Regression summariser
    >>> question = "my sentence"
    >>> text = ["This is one sentence.",\
                "This is another sentence.",\
                "This is yet another sentence.",\
                "And this is the final sentence."]
    >>> rsummarise(question,text,n=2)
    ['This is one sentence.', 'And this is the final sentence.']
"""
    global tfidf, regression
    featuresquestion = tfidf.transform([question])
    sentences = []
    for d in docstext:
        if sentTokenised:
            sentences += d
        else:
            sentences += sent_tokenize(d)
    if len(sentences) == 0:
        return ["No text found for summarisation"]

    featuressentences = tfidf.transform(sentences)
    distances = pairwise_distances(featuresquestion,
                                   featuressentences, 'cosine').transpose()
    allfeatures = scipy.sparse.hstack((distances, featuressentences))
    predictions = regression.predict(allfeatures)
    scores = zip(predictions, range(len(predictions)))
    scores.sort()
    summary = scores[-n:]

    summary.sort(cmp=lambda x, y: cmp(x[1], y[1]))
    return [sentences[i] for (score, i) in summary]


def twostepsummarise(question, docstext, n=2):
    """Cascade single-document with another single-document summariser"""
    # First step
    firststeptext = []
    for abstract in docstext:
        if len(abstract.strip()) == 0:
            continue
        summary = [abstract[b:e] for ((b, e), score) in summarise.summarise(question, abstract)]
        firststeptext.append(" ".join(summary))

    # Second step
    return singlesummarise(question, firststeptext, n)


def qsummarise(question, text, n=3, sentTokenize=False):
    """simple query-based summarisation.

    Performs cosine similarity between tf.idf of the question and tf.idf of the candidate sentences.

    >>> question = "What is the best treatment for migraines?"
    >>> text = "Exome results are reported for two patients with connective tissue dysplasia, one refining a clinical diagnosis of Ehlers-Danlos to Marfan syndrome, the other suggesting arthrogryposis derived from maternofetal Stickler syndrome. Patient 1 had mutations in transthyretin (TTR), fibrillin (FBN1), and a calcium channel (CACNA1A) gene suggesting diagnoses of transthyretin amyloidosis, Marfan syndrome, and familial hemiplegic migraines, respectively. Patient 2 presented with arthrogryposis that was correlated with his mother's habitus and arthritis once COL2A1 mutations suggestive of Stickler syndrome were defined. Although DNA results often defy prediction by the best of clinicians, these patients illustrate needs for ongoing clinical scholarship (e.g., to delineate guidelines for management of mutations like that for hyperekplexia in Patient 2) and for interpretation of polygenic change that is optimized by clinical genetic/syndromology experience (e.g., suggesting acetazolamide therapy for Patient 1 and explaining arthrogryposis in Patient 2)."
    >>> for s in qsummarise(question,text,n=3):
    ...   print(s[0] + "," + round(s[1],4))
    (0, 229) 1.0
    (230, 449) 0.9018
    (618, 1057) 0.9252
"""
    # global tfidf #, pca
    # Obtain the question and sentence features
    sentences = text
    if not sentTokenize:
        sentences = sent_tokenize(text)
    # Re-creating the text summary if the text is tokenized as the character offsets require it
    else:
        text = ''.join(sentences)


    # # -- Uncomment this code to use tfidf of cluster information
    # sentence_words = [my_tokenize(s) for s in sentences+[question]]
    # sentence_clusters = []
    # for s in sentence_words:
    #     sentence_clusters.append([c for c in word2vec.clusters(s) if c])

    # tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
    # tfidf.fit(sentence_clusters)
    # tfidf_question = tfidf.transform(sentence_clusters[-1:])
    # tfidf_sentences = tfidf.transform(sentence_clusters[:-1])
    # # --

    # -- Uncomment this code to use tfidf of words
    tfidf = TfidfVectorizer(stop_words="english",
                            lowercase="true")
    tfidf.fit(sentences+[question])

    tfidf_question = tfidf.transform([question])
    tfidf_sentences = tfidf.transform(sentences)
    # --

    # pca_question = pca.transform(tfidf_question)
    # pca_sentences = pca.transform(tfidf_sentences)


    # Score each sentence
    distances = pairwise_distances(tfidf_question,
                                   tfidf_sentences,
                                   'cosine')[0,:]
    scores = [(distances[i],i) for i in range(len(sentences))]

    # Obtain the top n sentences
    scores.sort()

    summary = scores[:n]

    # Find the character offsets
    offsets = []
    begin = 0
    for s in sentences:
        b = text.find(s,begin)
        assert(b >= 0)
        offsets.append((b,b+len(s)))
        begin = offsets[-1][1]

    # Return the results
    summary.sort(key=lambda x: x[1])
    #summary.sort(cmp = lambda x,y: cmp(x[1],y[1]))
    return [(offsets[i],score) for (score,i) in summary]



def rtwostepsummarise(question, docstext, n=2):
    """Cascade single-document with regression multi-document summariser"""
    # First step
    firststeptext = []
    for abstract in docstext:
        if len(abstract.strip()) == 0:
            continue
        summary = [abstract[b:e] for ((b, e), score) in summarise.summarise(question, abstract)]
        firststeptext.append(summary)

    # Second step
    return rsummarise(question, firststeptext, n, sentTokenised=True)


def getSnips(q_path=TEST_QUESTIONS_PATH,
             a_path=ABSTRACTS_PATH,
             corpus_path=CORPUS_PATH, verbose=1):
    """Yield the question, abstract, and target summaries
    >>> snips = [l for l in get_snips(verbose=0)]
    Processing 78 questions
    >>> len(snips)
    236
    >>> print(snips[0][0] + "," + snips[0][1])
    6495 Is there much risk in using fluoroquinolones in children?
    >>> print(snips[0][2])
    No, the risks seem to be minimal. Arthralgias and myalgias have been observed clinically in children and adolescents exposed to fluoroquinolones, but they're transient, disappear when the drug is discontinued, and appear to be no more prevalent than with other antibiotics.
    >>> print(len(snips[1][3]))
    2
    >>> print(snips[1][3][0][:60])
    The use of quinolones in children and accumulation of data o
    >>> print(snips[1][3][1][:60])
    Quinolone-induced arthropathic toxicity in weight-bearing jo
"""
    # Build DOM dictionary of questions
    d = parse(corpus_path)
    q_dom = dict()
    for qd in d.getElementsByTagName('record'):
        q_dom[qd.getAttribute('id')] = qd

    # Obtain questions IDs
    with open(q_path) as f:
        question_ids = [l.split()[0] for l in f.readlines()]
    print("Processing " + str(len(question_ids)) + " questions")
    for q in question_ids:
        qd = q_dom[q]
        qtext = getText(qd.getElementsByTagName('question')[0].childNodes)
        for snip in qd.getElementsByTagName('snip'):
            sniptext = getText(snip.getElementsByTagName('sniptext')[0].childNodes)
            abstracts = []
            for r in snip.getElementsByTagName('ref'):
                filename = os.path.join(a_path, r.getAttribute('abstract'))
                abstracts.append(getAbstract(filename, verbose=verbose)[0])

            yield (q, qtext, sniptext, abstracts)


def bioasq_summarise(question, abstract, nnc, n=2):
    """Summariser returns list of tuples of top n sentences and ranked
     in format (sentence, sentence rank)
    >>> test_question = "my sentence"
    >>> test_text = ["This is one sentence.",\
                "This is another sentence.",\
                "This is yet another sentence.",\
                "And this is the final sentence."]
    >>> bioasq_single_summarise(test_question,test_text, None, n=2)
    [('This is one sentence.', 0.48373976), ('This is another sentence.', 0.45038313)]
"""
    # load model if not passed
    if not nnc:
        nnc = LSTMSimilarities(hidden_layer=50, build_model=False, positions=True)
        nnc.fit(None, None, None, restore_model=True, verbose=0, savepath="task8b_nnc_model_1024")

    # clean and tokenize abstract (currently assumes abstract is split into sentences)
    candidates_sentences = [nnc.cleantext(s) for s in abstract]
    candidates_sentences_ids = list(range(len(abstract)))
    predictions = [p[0] for p in nnc.predict(candidates_sentences,
                                             [nnc.cleantext(question)] * len(abstract),
                                             X_positions=candidates_sentences_ids)]
    predictions_unranked = zip(candidates_sentences_ids, predictions)
    predictions_ranked = sorted(predictions_unranked, key=lambda x: x[1], reverse=True)

    # tuple_sentences are in format of ([sentence], id, prediction)
    ranked_sentence_tuples = [(abstract[x[0]], x[0], x[1]) for x in predictions_ranked[0:n]]

    # re-sort to be in order of occurence, using sentence id
    ordered_top_sentences = sorted(ranked_sentence_tuples, key=lambda x: x[1], reverse=False)

    return ordered_top_sentences


def bioasq_multi_summarise(question, documents, nnc, n=2, sent_tokenized=False):
    """ Utilises qsummarize to get top n predictions from each document
    and bioasq_summarize results to get the models top n predicted summaries overall"""

    # load model once if not passed
    if not nnc:
        nnc = LSTMSimilarities(hidden_layer=50, build_model=False, positions=True)
        nnc.fit(None, None, None, restore_model=True, verbose=0, savepath="task8b_nnc_model_1024")

    # docs_tokenized = []
    # for d in documents:
    #     if not sent_tokenized:
    #         docs_tokenized.append(sent_tokenize(d))
    #     else:
    #         docs_tokenized.append(d)
    # if len(docs_tokenized) == 0:
    #     return ["No text found for summarisation"]

    q_summarize_results = []
    for doc in documents:
        # text should be as string of sentences since qsummarize uses both tokenized and string versions
        single_q_summarize = qsummarise(question, doc, n=n)
        # summary = [abstract[b:e] for ((b, e), score) in summarise.summarise(question, abstract)]
        summary = [doc[b:e] for ((b, e), score) in single_q_summarize]
        for result in summary:
            q_summarize_results.append(result)

    bioasq_results = bioasq_summarise(question, q_summarize_results, nnc, n=n)

    return bioasq_results


# def evaluate(testfile, summariser, xml_rouge_filename="rouge.xml"):
#     "Evaluate the multisummariser"
#     frouge = open(xml_rouge_filename, 'w')
#     frouge.write('<ROUGE-EVAL version="1.0">\n')
#     rouge_i = 0
#     for (qid, question, target, abstracts) in getSnips(q_path=TEST_QUESTIONS_PATH,
#                                                        a_path=ABSTRACTS_PATH,
#                                                        corpus_path=CORPUS_PATH,
#                                                        verbose=0):
#         rouge_i += 1

#         # XML data
#         frouge.write("""<EVAL ID="%i">
#  <MODEL-ROOT>
#  rouge/models
#  </MODEL-ROOT>
#  <PEER-ROOT>
#  rouge/peers
#  </PEER-ROOT>
#  <INPUT-FORMAT TYPE="SPL" />
#  <MODELS>
#    <M ID="1">model%i</M>
#  </MODELS>
#  <PEERS>
#    <P ID="A">peer%i</P>
#  </PEERS>
# </EVAL>
# """ % (rouge_i, rouge_i, rouge_i))

#         # Model
#         with codecs.open('rouge/models/model%i' % rouge_i, 'w', 'utf-8') as fout:
#             fout.write('\n'.join(sent_tokenize(target)) + '\n')

#         # Peer
#         with codecs.open('rouge/peers/peer%i' % rouge_i, 'w', 'utf-8') as fout:
#             if len(abstracts) == 0:
#                 fout.write('\n')
#             else:
#                 summaries = summariser(question, abstracts)
#                 if len(summaries) == 0:
#                     fout.write('\n')
#                 else:
#                     for sentence in summaries:
#                         fout.write(sentence + '\n')

#     frouge.write('</ROUGE-EVAL>\n')
#     frouge.close()
#     ROUGE_CMD = 'perl rouge/ROUGE-1.5.5.pl -e rouge/data -a -n 2 -2 4 -U %s' % (xml_rouge_filename)
#     print("Calling " + ROUGE_CMD)
#     os.system(ROUGE_CMD)


if __name__ == "__main__":
    # import doctest
    # doctest.testmod()

    question = "What is the best treatment for migraines?"
    text = [[
        "Exome results are reported for two patients with connective tissue dysplasia, one refining a clinical diagnosis of Ehlers-Danlos to Marfan syndrome, the other suggesting arthrogryposis derived from maternofetal Stickler syndrome.",
        "Patient 1 had mutations in transthyretin (TTR), fibrillin (FBN1), and a calcium channel (CACNA1A) gene suggesting diagnoses of transthyretin amyloidosis, Marfan syndrome, and familial hemiplegic migraines, respectively.",
        "this also has nothing to do with the question",
        "Patient 2 presented with arthrogryposis that was correlated with his mother's habitus and arthritis once COL2A1 mutations suggestive of Stickler syndrome were defined.",
        "Although DNA results often defy prediction by the best of clinicians, these patients illustrate needs for ongoing clinical scholarship (e.g., to delineate guidelines for management of mutations like that for hyperekplexia in Patient 2) and for interpretation of polygenic change that is optimized by clinical genetic/syndromology experience (e.g., suggesting acetazolamide therapy for Patient 1 and explaining arthrogryposis in Patient 2).",
        "this has nothing to do with the question"],
        ["Migraines are best treated with neurofen or panadol"]]
    results = bioasq_multi_summarise(question, text, nnc, 3, sent_tokenized=True)
    print(results)

    #    evaluate(TEST_QUESTIONS_PATH,singlesummarise)
    #    evaluate(TEST_QUESTIONS_PATH,twostepsummarise)
    #    evaluate(TEST_QUESTIONS_PATH,rsummarise)
    #    evaluate(TEST_QUESTIONS_PATH, rtwostepsummarise)
