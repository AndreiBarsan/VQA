# coding: utf-8

import sys

# Early fix to allow plotting to work with no X server.
import matplotlib
matplotlib.use('Agg')

# TODO(andrei): Use this toolkit for the evaluation, since it's the official
# one, and offers a simple breakdown of our errors based on question categories.
# TODO(andrei): Make these command-line arguments.
# data_dir = '../../VQA'
data_dir = '/data/vqa'
code_dir = '.'

# TODO(andrei): Better argument processing.
experiment_dir = sys.argv[1]
print("Using experiment dir: [{0}]".format(experiment_dir))

sys.path.insert(0, '%s/PythonHelperTools/vqaTools' % (code_dir))

from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import matplotlib.pyplot as plt
import skimage.io as io
import json
import random
import os

# set up file names and paths
taskType    ='OpenEnded'
dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
# dataSubType ='train2014'
dataSubType ='val2014'
annFile     ='%s/Annotations/%s_%s_annotations.json'%(data_dir, dataType, dataSubType)
quesFile    ='%s/Questions/%s_%s_%s_questions.json'%(data_dir, taskType, dataType, dataSubType)
imgDir      ='%s/Images/%s/%s/' %(data_dir, dataType, dataSubType)

resultType  ='baseline'
fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']

# An example result json file has been provided in './Results' folder.

[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = [
    '%s/%s_%s_%s_%s_%s.json'%(experiment_dir, taskType, dataType,
                                      dataSubType, resultType, fileType) for fileType in fileTypes]
figFile = '{0}/accuracy-bars-{1}'.format(experiment_dir, dataSubType)

# create vqa object and vqaRes object
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

# evaluate results
"""
If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
By default it uses all the question ids in annotation file
"""
vqaEval.evaluate()

# print accuracies
print "\n"
print "Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall'])
print "Per Question Type Accuracy is the following:"
for quesType in vqaEval.accuracy['perQuestionType']:
	print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
print "\n"
print "Per Answer Type Accuracy is the following:"
for ansType in vqaEval.accuracy['perAnswerType']:
	print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
print "\n"

# demo how to use evalQA to retrieve low score result
evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId]<35]   #35 is per question percentage accuracy
if len(evals) > 0:
	print 'ground truth answers'
	randomEval = random.choice(evals)
	randomAnn = vqa.loadQA(randomEval)
	vqa.showQA(randomAnn)

	print '\n'
	print 'generated answer (accuracy %.02f)'%(vqaEval.evalQA[randomEval])
	ann = vqaRes.loadQA(randomEval)[0]
	print "Answer:   %s\n" %(ann['answer'])

	imgId = randomAnn[0]['image_id']
	imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
	if os.path.isfile(imgDir + imgFilename):
		I = io.imread(imgDir + imgFilename)
		plt.imshow(I)
		plt.axis('off')
		plt.show()

# plot accuracy for various question types
fig = plt.figure()
# TODO(andrei): Rotate bar labels.
plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy[
    'perQuestionType'].values(), align='center')
plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation='0',fontsize=10)
plt.title('Per Question Type Accuracy', fontsize=10)
plt.xlabel('Question Types', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)

# Save the figure in both raster and vector format. The latter looks very
# pretty when included in LaTeX!
fig.savefig(figFile + '.png')
fig.savefig(figFile + '.eps')

# save evaluation results to ./Results folder
json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))

