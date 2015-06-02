import cv2
from numpy import zeros, histogram, hstack, vstack, savetxt
import scipy.cluster.vq as vq
import svmutil
from cPickle import load

HISTOGRAMS_FILE = 'testdata.svm'
CODEBOOK_FILE = 'dataset_tinycodebook.file'
MODEL_FILE = 'trainingdata.svm.model'
CAT_LABEL = 'cat.txt'

def extractSift(input_file):
    print "extracting Sift features"
    all_features_dict = {}
    image = cv2.imread(input_file)
    gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    sift =cv2.SIFT()
    kp, descriptors = sift.detectAndCompute(gray,None)
    print "gathering sift features for", input_file,
    print descriptors.shape
    all_features_dict[input_file] = descriptors
    return all_features_dict
    
def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words
    
def writeHistogramsToFile(nwords, labels, fnames, all_word_histgrams, features_fname):
    data_rows = zeros(nwords + 1)  # +1 for the category label
    histogram = all_word_histgrams
    if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
        nwords = histogram.shape[0]
        data_rows = zeros(nwords + 1)
        print 'nclusters have been reduced to ' + str(nwords)
    data_row = hstack((labels[fnames], histogram))
    data_rows = vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(nwords):
        fmt = fmt + str(i) + ':%f '
    savetxt(features_fname, data_rows, fmt)

print "---------------------"
print "## extract Sift features"
all_files = []
all_files_labels = {}
all_features = {}

codebook_file = CODEBOOK_FILE
model_file = MODEL_FILE
#fnames = filename = raw_input('Enter input file name: ')
fname = 'scorpion.jpg'
all_features = extractSift(fname)
all_files_labels[fname] = 4  # label is unknown

print "---------------------"
print "## loading codebook from " + codebook_file
with open(codebook_file, 'rb') as f:
    codebook = load(f)

print "---------------------"
print "## computing visual word histograms"
word_histgram = computeHistograms(codebook, all_features[fname])

print "---------------------"
print "## write the histograms to file to pass it to the svm"
nclusters = codebook.shape[0]
writeHistogramsToFile(nclusters,
                      all_files_labels,
                      fname,
                      word_histgram,
                      HISTOGRAMS_FILE)

print "---------------------"
print "## test data with svm"

y,x=svmutil.svm_read_problem(HISTOGRAMS_FILE)
model=svmutil.svm_load_model(model_file)
result = svmutil.svm_predict(y,x,model)
cat_label = load(open("cat.txt", "rb" ))
print cat_label
print result[0][0]
