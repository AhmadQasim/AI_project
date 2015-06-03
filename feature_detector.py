from os.path import isdir, basename, join
import cv2
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like, random
import scipy.cluster.vq as vq
from cPickle import dump, HIGHEST_PROTOCOL
import svmutil


EXTENSIONS = [".jpg", ".bmp", ".png"]
DATASETPATH = 'D:\\AI_project-master\\dataset'
PRE_ALLOCATION_BUFFER = 1000  # for sift
HISTOGRAMS_FILE = 'trainingdata.svm'
K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
CODEBOOK_FILE = 'codebook.file'
NUM_SAMPLES=10
NUM_TRANING_SAMPLES=5
NUM_PYRAMIDS=16

def get_categories(datasetpath):
    cat_paths = [files
                 for files in glob(datasetpath + "/*")
                  if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats


def get_imgfiles(path, num_of_samples):
    all_files = []
    files = glob(path + "/*")
    for i in xrange(num_of_samples):
        all_files.extend([join(path, basename(files[random.randint(0,len(files),1)]))])
    return all_files


def extractSift(input_files):
    print "extracting Sift features"
    all_features_dict = {}
    for i,fname in enumerate(input_files):
        image = cv2.imread(fname)
        gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        sift =cv2.SIFT()
        kp, descriptors = sift.detectAndCompute(gray,None)
        print "gathering sift features for", fname,
        print descriptors.shape
        all_features_dict[fname] = descriptors
    return all_features_dict


def dict2numpy(dict):
    nkeys = len(dict)
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, 128))
    return array


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words


def writeHistogramsToFile(nwords, labels, fnames, all_word_histgrams, features_fname):
    data_rows = zeros(nwords + 1)  # +1 for the category label
    for fname in fnames:
        histogram = all_word_histgrams[fname]
        if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = zeros(nwords + 1)
            print 'nclusters have been reduced to ' + str(nwords)
        data_row = hstack((labels[fname], histogram))
        data_rows = vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(nwords):
        fmt = fmt + str(i) + ':%f '
    savetxt(features_fname, data_rows, fmt)

if __name__ == '__main__':
    print "---------------------"
    print "## loading the images and extracting the sift features"
    cats = get_categories(DATASETPATH)
    ncats = len(cats)
    print "searching for folders at " + DATASETPATH
    if ncats < 1:
        raise ValueError('Only ' + str(ncats) + ' categories found. Wrong path?')
    print "found following folders / categories:"
    print cats
    print "---------------------"    
    all_files = []
    all_files_labels = {}
    all_features = {}
    cat_label = {}
    training_features = {}
    training_files = []
    training_files_labels={}
    for cat, label in zip(cats, range(ncats)):
        cat_path = join(DATASETPATH, cat)
        cat_files = get_imgfiles(cat_path, NUM_SAMPLES)
        cat_features = extractSift(cat_files)
        all_files = all_files + cat_files
        all_features.update(cat_features)
        cat_label[cat] = label
        for i in xrange(NUM_TRANING_SAMPLES):
            training_features.update({cat_files[i]:cat_features[cat_files[i]]});
            training_files.append(cat_files[i])
            training_files_labels[cat_files[i]]=label
        for i in cat_files:
            all_files_labels[i] = label

    print "---------------------"
    print "## computing the visual words via k-means"
    all_features_array = dict2numpy(all_features)
    nfeatures = all_features_array.shape[0]
    nclusters = int(sqrt(nfeatures))
    codebook, distortion = vq.kmeans(all_features_array,
                                             nclusters,
                                             thresh=K_THRESH)

    with open(DATASETPATH + CODEBOOK_FILE, 'wb') as f:

        dump(codebook, f, protocol=HIGHEST_PROTOCOL)

    print "---------------------"
    print "## compute the visual words histograms for each image"
    all_word_histgrams = {}
    
    for imagefname in training_features:
        word_histgram = computeHistograms(codebook, training_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram

    print "---------------------"
    print "## write the histograms to file to pass it to the svm"
    writeHistogramsToFile(nclusters,training_files_labels,training_files,all_word_histgrams,DATASETPATH + HISTOGRAMS_FILE)

    print "---------------------"
    print "## train svm"
    y,x=svmutil.svm_read_problem(DATASETPATH + HISTOGRAMS_FILE)
    model_file=svmutil.svm_train(y,x)
    svmutil.svm_save_model('trainingdata.svm.model', model_file)

    print "--------------------"
    print "## outputting results"
    print "codebook file: " + DATASETPATH + CODEBOOK_FILE
    print "category      ==>  label"
    for cat in cat_label:
        print '{0:13} ==> {1:6d}'.format(cat, cat_label[cat])
