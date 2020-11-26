"""
Search API:
Private
Version 0.4, 06-OCT-2014

This is a search module which provides the ability to query external databases 
for medical based resources and publications. It makes use of the Biopython (Bio) 
Entrez module which is used to retrieve data from the PubMed database. 
"""

from Bio import Entrez

MAX_RESULTS = 50 
# default maximum number of results to return
   
Entrez.email = "jack.fenton@students.mq.edu.au"
# Biopython email setting


def search(query, max_results=MAX_RESULTS):
    """Performs search and retrieval of evidence based medical results.
    
    Return the dictionary of results. Each dictionary is defined by an ID
    (i.e. PubMed ID) which forms the key. The value of each PubMed ID is 
    another dictionary which contains the attributes of each result
    (id, title, authors, publication, publication year, content, summary and url).
    
    - query: a string containing the query to search for.
    - max_results (optional): the maximum number of results to return, defaults
        to the MAX_RESULTS constant if not provided.
        
    Example:
    >>> query = "how to treat migraines"
    >>> search(query, 2)
    { '123': { "title": 'Treating Migraines 101',
               "authors": 'John Smith',
               "publication": 'Dr. Suess',
               "publication_year": '1967',
               "content": 'Take a panadol',
               "summary": 'Panadol',
               "url": 'http://www.ncbi.nlm.nih.gov/pubmed/123'
             },
    '456': { "title": 'How to treat a headache',
               "authors": 'Mary Citizen',
               "publication": 'Dr. Hibbert',
               "publication_year": '2000',
               "content": 'Rehydrate, adjust sugar levels and rest.',
               "summary": "Adjust sugar levels",
               "url": 'http://www.ncbi.nlm.nih.gov/pubmed/456'
             }
    } 
    
    """
      
    query_results = {}
    
    # get entrez handle and list of result IDs
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    ids = Entrez.read(handle)["IdList"]
    
    if not ids:
        return query_results
    
    # fetch the results
    handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")    
    results = Entrez.read(handle)
    
    # process the results to extract relevant information
    for r in results['PubmedArticle']:
        if 'MedlineCitation' in r and 'Article' in r['MedlineCitation']:
            if 'ArticleTitle' in r['MedlineCitation']['Article'] and 'Abstract' in r['MedlineCitation']['Article']:
                authors = ""
                if 'AuthorList' in r['MedlineCitation']['Article']:
                    for author in r['MedlineCitation']['Article']['AuthorList']:
                        auth = ""                        
                        if 'LastName' in author:
                            auth = author['LastName']
                        if 'Initials' in author:
                            auth += " " + author['Initials']     
                        if authors:
                            authors += ", "                       
                        authors += auth
                
                publication = ""
                publication_year = ""
                if 'Journal' in r['MedlineCitation']['Article']:
                    journal_dict = r['MedlineCitation']['Article']['Journal']
                    if 'Title' in journal_dict:
                        publication += journal_dict['Title'] + "."
                    if 'JournalIssue' in journal_dict and 'PubDate' in journal_dict['JournalIssue']:
                        date = journal_dict['JournalIssue']['PubDate']
                        if 'Year' in date:
                            publication_year += " " + date['Year']
                        if 'Month' in date:
                            publication_year += " " + date['Month']
                        if 'Day' in date:
                            publication_year += " " + date['Day']                  
                      
                record = { str(r['MedlineCitation']['PMID']):
                              {
                        "title": r['MedlineCitation']['Article']['ArticleTitle'],
                        "authors": authors,
                        "publication": publication,
                        "publication_year": publication_year,
                        "summary": r['MedlineCitation']['Article']['Abstract']['AbstractText'][0],
                        "content": r['MedlineCitation']['Article']['Abstract']['AbstractText'][0],
                        "url": 'http://www.ncbi.nlm.nih.gov/pubmed/' + r['MedlineCitation']['PMID']
                              }
                          }
                query_results.update(record)
    
    return query_results


if __name__ == "__main__":
    import doctest
    doctest.testmod()
