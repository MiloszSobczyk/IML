# Introduction
Project carried out as part of the Introduction to Machine Learning course in the Computer Science program at the Faculty of Mathematics and Information Science, Warsaw University of Technology.

# Setup
### Generate your own files
To generate spectrograms extract contents of file from https://zenodo.org/records/4660670 into the *./resources* folder and remove non-wav files. You can do most of this by runnning:
```
cd ./resources 
python cleanup.py
python convert.py
```
Some files won't be deleted, since I was lazy and didn't want to write more code. I know you will manage.
You may need to install additional python packages using *pip*.

### Already generated files
Just put them in the resources folder.

Congrats!