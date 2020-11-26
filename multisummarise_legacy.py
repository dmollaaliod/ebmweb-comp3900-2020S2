"multisummarise.py - Multi-document summarisation"

import requests
import json


def defaultsummarise(question, docstext, n=2):
    "Default summariser"
#    url = "http://localhost:8080/search"
    url = "http://localhost:8080/bioasqsummarise"
    payload = {"query": question,
               "summaries": json.dumps(docstext),
               "sentences": n}
    result = requests.post(url, data=payload)
    return json.loads(result.text)



if __name__ == "__main__":
    import doctest
    doctest.testmod()

    question = "What is the best treatment for migraines?"
    text = [
        "Exome results are reported for two patients with connective tissue dysplasia, one refining a clinical diagnosis of Ehlers-Danlos to Marfan syndrome, the other suggesting arthrogryposis derived from maternofetal Stickler syndrome. Patient 1 had mutations in transthyretin (TTR), fibrillin (FBN1), and a calcium channel (CACNA1A) gene suggesting diagnoses of transthyretin amyloidosis, Marfan syndrome, and familial hemiplegic migraines, respectively. This also has nothing to do with the question. Patient 2 presented with arthrogryposis that was correlated with his mother's habitus and arthritis once COL2A1 mutations suggestive of Stickler syndrome were defined. Although DNA results often defy prediction by the best of clinicians, these patients illustrate needs for ongoing clinical scholarship (e.g., to delineate guidelines for management of mutations like that for hyperekplexia in Patient 2) and for interpretation of polygenic change that is optimized by clinical genetic/syndromology experience (e.g., suggesting acetazolamide therapy for Patient 1 and explaining arthrogryposis in Patient 2). This has nothing to do with the question",
        "Migraines are best treated with neurofen or panadol"]
    results = defaultsummarise(question, text, 3)
    print(results)
