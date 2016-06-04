import gzip
import os
import urllib
import urllib.request
import csv
import zipfile

import numpy as np

DOWNLOADS_DIR = 'data/'


# Download data iff it's not already in DOWNLOADS_DIR
# return the file name of file in dir
def cached_download(url):
    # Split on the rightmost / and take everything on the right side of that
    name = url.rsplit('/', 1)[-1]

    # Combine the name and the downloads directory to get the local filename
    filename = os.path.join(DOWNLOADS_DIR, name)

    # Download the file if it does not exist
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)

    return filename


# non-strict json parse
def parse_amazon(infile):
    g = gzip.open(infile, 'r')
    for l in g:
        yield eval(l)


def movielens_unzip(infile):
    g = zipfile.ZipFile(infile)
    g.extractall(DOWNLOADS_DIR)


def parse_movieLens(path):
    train = []
    with open(os.path.join(path, 'u1.base')) as f:
        for line in f:
            uId, iId, r, timestamp = line.strip().split()
            train.append((int(uId), int(iId), int(r)))
    train = np.array(train, dtype=np.int32)

    test = []
    with open(os.path.join(path, 'u1.test')) as f:
        for line in f:
            uId, iId, r, timestamp = line.strip().split()
            test.append((int(uId), int(iId), int(r)))
    test = np.array(test, dtype=np.int32)

    return train, test


def out_data_movieLens(fnameTrain, fnameTest, train, test):
    np.savetxt(os.path.join(DOWNLOADS_DIR, fnameTrain), train, fmt='%d', delimiter=",")
    np.savetxt(os.path.join(DOWNLOADS_DIR, fnameTest), test, fmt='%d', delimiter=",")


def parse_amazon_csv(fname, delim):
    with open(fname, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delim)
        X = []
        y = []
        for line in csvreader:
            uId, iId, r, timestamp = line
            X.append((uId, iId, int(timestamp)))
            y.append(float(r))

    unique_users = list(set([d[0] for d in X]))
    unique_items = list(set([d[1] for d in X]))

    uid_map = dict(zip(unique_users, list(range(len(unique_users)))))
    iid_map = dict(zip(unique_items, list(range(len(unique_items)))))
    unique_users = list(uid_map.keys())
    unique_items = list(iid_map.keys())

    X = [[uid_map[uid], iid_map[iid], time] for [uid, iid, time] in X]

    return unique_users, uid_map, unique_items, iid_map, X, y
