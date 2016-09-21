import urllib2

#uncomment the url for the respective glove file
url_6b = "http://nlp.stanford.edu/data/glove.6B.zip"
# url_42b = "http://nlp.stanford.edu/data/glove.42B.300d.zip"
# url_840b = "http://nlp.stanford.edu/data/glove.840B.300d.zip"

#url_'x'b for the respective word vectors
glove_file = url_6b.split('/')[-1]
u = urllib2.urlopen(url_6b)
f = open(glove_file, 'wb')
meta = u.info()
file_size = int(meta.getheaders("Content-Length")[0])
print "Downloading: %s Bytes: %s" % (glove_file, file_size)

size_dl = 0
block_sz = 8192
while True:
    buffer = u.read(block_sz)
    if not buffer:
        break

    size_dl += len(buffer)
    f.write(buffer)
    status = r"%10d  [%3.2f%%]" % (size_dl, size_dl * 100. / file_size)
    status = status + chr(8)*(len(status)+1)
    print status,

f.close()
