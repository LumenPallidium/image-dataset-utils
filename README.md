# image-dataset-utils
Just a repo for me to store some image utility scripts I've made.

- duplicate_remover : Uses the imagehash library to build a dataframe of image hashes for a given folder, then moves all the duplicates (keeping the largest) to a new folder (specified). Works on many common image and video formats (for videos, the first frame is used). Very useful for cleaning up files gotten from webscraping.
