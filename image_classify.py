from os.path import join
import svmutil
from cPickle import load
from feature_detector import extractSift, computeHistograms,writeHistogramsToFile,get_categories,get_imgfiles

HISTOGRAMS_FILE = 'testdata.svm'
CODEBOOK_FILE = 'datasetcodebook.file'
DATASETPATH = 'D:\\AI_project-master\\dataset'
MODEL_FILE = 'trainingdata.svm.model'
CAT_LABEL = 'cat.txt'
TEST_SAMPLES= 2

print "---------------------"
print "## extract Sift features"
all_files = []
all_files_labels = {}
all_features = {}
cat_label = {}
cats = get_categories(DATASETPATH)
ncats = len(cats)

for cat, label in zip(cats, range(ncats)):
    cat_path = join(DATASETPATH, cat)
    cat_files = get_imgfiles(cat_path, TEST_SAMPLES)
    cat_features = extractSift(cat_files)
    all_files = all_files + cat_files
    all_features.update(cat_features)
    cat_label[cat] = label
    for i in cat_files:
        all_files_labels[i] = label

#fnames = filename = raw_input('Enter input file name: ')

print "---------------------"
print "## loading codebook from " + CODEBOOK_FILE
with open(CODEBOOK_FILE, 'rb') as f:
    codebook = load(f)

print "---------------------"
print "## computing visual word histograms"
all_word_histgrams={}
for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram

print "---------------------"
print "## write the histograms to file to pass it to the svm"
nclusters = codebook.shape[0]
writeHistogramsToFile(nclusters,
                      all_files_labels,
                      all_files,
                      all_word_histgrams,
                      HISTOGRAMS_FILE)

print "---------------------"
print "## test data with svm"

y,x=svmutil.svm_read_problem(HISTOGRAMS_FILE)
model=svmutil.svm_load_model(MODEL_FILE)
result = svmutil.svm_predict(y,x,model)
