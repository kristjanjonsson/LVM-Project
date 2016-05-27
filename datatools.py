import gzip
import os
import urllib
import urllib.request
import csv

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


