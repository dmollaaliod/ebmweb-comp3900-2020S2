from bottle import route, run, template, get, post, request
import json

from nltk import sent_tokenize

from search import search
from multisummarise import bioasq_multi_summarise
from nnc import LSTMSimilarities


@post('/search')
def pubmed_search():
    search_query = request.forms.get('query')
    results = search(search_query, 10)

    summaries = []
    for document in results:
        if 'summary' in results[document]:
            summaries.append(results[document]['summary'])

    return summaries


@post('/searchandbioasqsummarise')
def summarise():
    search_query = request.forms.get('query')
    sentences = request.forms.get('sentences')
    results = search(search_query, 10)

    summaries = []
    for document in results:
        if 'summary' in results[document]:
            summaries.append(results[document]['summary'])

    if sentences:
        summariser_results = bioasq_multi_summarise(search_query, summaries, nnc, sentences)
    else:
        summariser_results = bioasq_multi_summarise(search_query, summaries, nnc, 3)

    # results are in tuples, which are non json serializable. We only need the summaries.
    return json.dumps([y[0] for y in summariser_results])


@post('/bioasqsummarise')
def summarise():
    search_query = request.forms.get('query')
    summaries = json.loads(request.forms.get('summaries'))
    sentences = request.forms.get('sentences')

    if sentences:
        sentences = int(sentences)
    else:
        sentences = 3

    #list_sentences = [s for abstract in summaries for s in sent_tokenize(abstract)]

    #return json.dumps(list_sentences[:sentences])


    summariser_results = bioasq_multi_summarise(search_query, summaries, nnc, sentences)

    # results are in tuples, which are non json serializable. We only need the summaries.
    return json.dumps([y[0] for y in summariser_results])


nnc = LSTMSimilarities(hidden_layer=50, build_model=False, positions=True)
nnc.fit(None, None, None, restore_model=True, verbose=0, savepath="task8b_nnc_model_1024")
run(host='localhost', port=8080)
