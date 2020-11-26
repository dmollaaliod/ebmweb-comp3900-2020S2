# COMP3900 2020S2

Code that integrates Jack Fenton's work in COMP3900. This system incorporates BioASQ's deep learning query-focused summariser in the ebmweb summariser.

## Files needed

Besides the files available in this repository, you need to obtain the following files. They are not in the github repository because of the large size of some of them:

- `allMeSH_2016_100.vectors.txt` (1.9GB)
- `task8b_nnc_model_1024.data-00000-of-00002` (1.6GB)
- `task8b_nnc_model_1024.data-00001-of-00002` (4MB)
- `task8b_nnc_model_1024.index` (4KB)
- `ebmweb_comp3900_dockerimage.tar.gz` (394MB)

## To run the server for the multisummariser
```
$ conda env create -f environment.yml
$ conda activate comp3900
(comp3900) $ python start_server.py
```

## To run the web application
```
$ docker load -i ebmweb_comp3900_dockerimage.tar
$ docker run --network="host" -it ebmweb_comp3900
/code# python src/manage.py runserver 0.0.0.0:8000
```