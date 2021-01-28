# import multiprocessing
from multiprocessing import Process, Pool, Lock, Queue

import os
import sys
import time
import argparse
from tqdm import tqdm 

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import aggregation_config_pb2
from delf import datum_io
from delf import feature_aggregation_similarity
from delf.python.detect_to_retrieve import dataset
from delf.python.detect_to_retrieve import image_reranking
from delf import feature_io
from scipy.spatial import distance
from tqdm import tqdm
from numba import jit

from IPython.display import Image 
import random

import pickle as pkl

from absl import flags

# FLAGS = flags.FLAGS

# flags.DEFINE_string('query_descriptors_path', '/tmp/', 'a')
# flags.DEFINE_string('index_descriptors_path', '/tmp/', 'a')
# flags.DEFINE_string('dataset_file_path', '/tmp/', 'a')
# flags.DEFINE_string('result_save_path', '/tmp/', 'a')

#FLAGS.train_directory


cmd_args = None

_PR_RANKS = (1, 5, 10)

_STATUS_CHECK_LOAD_ITERATIONS = 50

_METRICS_FILENAME = 'metrics.txt'

@jit(nopython = True)
def euclidean_distance_numba(x, y):
    return np.sqrt(np.sum((x-y)**2))


def multi_retrieval(query_index):
    sample_query = query_descriptors[query_index]
    similarities = np.zeros([num_index_images])
    for index_idx, descriptors in enumerate(index_descriptors):
        similarities[index_idx] = euclidean_distance_numba(sample_query, descriptors)
    return np.argsort(similarities)[:1000]


def ComputeAveragePrecision(positive_ranks):
  """Computes average precision according to dataset convention.

  It assumes that `positive_ranks` contains the ranks for all expected positive
  index images to be retrieved. If `positive_ranks` is empty, returns
  `average_precision` = 0.

  Note that average precision computation here does NOT use the finite sum
  method (see
  https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision)
  which is common in information retrieval literature. Instead, the method
  implemented here integrates over the precision-recall curve by averaging two
  adjacent precision points, then multiplying by the recall step. This is the
  convention for the Revisited Oxford/Paris datasets.

  Args:
    positive_ranks: Sorted 1D NumPy integer array, zero-indexed.

  Returns:
    average_precision: Float.
  """
  average_precision = 0.0

  num_expected_positives = len(positive_ranks)
  if not num_expected_positives:
    return average_precision

  recall_step = 1.0 / num_expected_positives
  for i, rank in enumerate(positive_ranks):
    if not rank:
      left_precision = 1.0
    else:
      left_precision = i / rank

    right_precision = (i + 1) / (rank + 1)
    average_precision += (left_precision + right_precision) * recall_step / 2

  return average_precision


def ComputeMetrics(sorted_index_ids, ground_truth, query_list, desired_pr_ranks, check, start=0):
  """Computes metrics for retrieval results on the Revisited datasets.

  If there are no valid ground-truth index images for a given query, the metric
  results for the given query (`average_precisions`, `precisions` and `recalls`)
  are set to NaN, and they are not taken into account when computing the
  aggregated metrics (`mean_average_precision`, `mean_precisions` and
  `mean_recalls`) over all queries.

  Args:
    sorted_index_ids: Integer NumPy array of shape [#queries, #index_images].
      For each query, contains an array denoting the most relevant index images,
      sorted from most to least relevant.
    ground_truth: List containing ground-truth information for dataset. Each
      entry is a dict corresponding to the ground-truth information for a query.
      The dict has keys 'ok' and 'junk', mapping to a NumPy array of integers.
    desired_pr_ranks: List of integers containing the desired precision/recall
      ranks to be reported. Eg, if precision@1/recall@1 and
      precision@10/recall@10 are desired, this should be set to [1, 10]. The
      largest item should be <= #index_images.

  Returns:
    mean_average_precision: Mean average precision (float).
    mean_precisions: Mean precision @ `desired_pr_ranks` (NumPy array of
      floats, with shape [len(desired_pr_ranks)]).
    mean_recalls: Mean recall @ `desired_pr_ranks` (NumPy array of floats, with
      shape [len(desired_pr_ranks)]).
    average_precisions: Average precision for each query (NumPy array of floats,
      with shape [#queries]).
    precisions: Precision @ `desired_pr_ranks`, for each query (NumPy array of
      floats, with shape [#queries, len(desired_pr_ranks)]).
    recalls: Recall @ `desired_pr_ranks`, for each query (NumPy array of
      floats, with shape [#queries, len(desired_pr_ranks)]).

  Raises:
    ValueError: If largest desired PR rank in `desired_pr_ranks` >
      #index_images.
  """
  sorted_index_ids = sorted_index_ids[:,:check]
  num_queries, num_index_images = sorted_index_ids.shape
  num_desired_pr_ranks = len(desired_pr_ranks)
  ground_truth = ground_truth[start:]
  query_list =  query_list[start:]
  sorted_desired_pr_ranks = sorted(desired_pr_ranks)

  if sorted_desired_pr_ranks[-1] > num_index_images:
    raise ValueError(
        'Requested PR ranks up to %d, however there are only %d images' %
        (sorted_desired_pr_ranks[-1], num_index_images))

  # Instantiate all outputs, then loop over each query and gather metrics.
  mean_average_precision = 0.0
#   mean_precisions = np.zeros([num_desired_pr_ranks])
#   mean_recalls = np.zeros([num_desired_pr_ranks])
  average_precisions = np.zeros([num_queries])
#   precisions = np.zeros([num_queries, num_desired_pr_ranks])
#   recalls = np.zeros([num_queries, num_desired_pr_ranks])
  num_empty_gt_queries = 0
  for i, im in zip(range(num_queries), query_list):
#     ok_index_images = ground_truth[i]['ok']
#     junk_index_images = ground_truth[i]['junk']
    ok_index_images = ground_truth[i][im]
#   for i in range(num_queries):
#     ok_index_images = ground_truth[i]['ok']
#     junk_index_images = ground_truth[i]['junk']

    if not ok_index_images.size:
      average_precisions[i] = float('nan')
#       precisions[i, :] = float('nan')
#       recalls[i, :] = float('nan')
      num_empty_gt_queries += 1
      continue
    
    positive_ranks = np.arange(num_index_images)[np.in1d(
        sorted_index_ids[i], ok_index_images)]
#     junk_ranks = np.arange(num_index_images)[np.in1d(sorted_index_ids[i],
#                                                      junk_index_images)]
    adjusted_positive_ranks = AdjustPositiveRanks(positive_ranks)
#     adjusted_positive_ranks = AdjustPositiveRanks(positive_ranks, junk_ranks)


    average_precisions[i] = ComputeAveragePrecision(adjusted_positive_ranks)
#     precisions[i, :], recalls[i, :] = ComputePRAtRanks(adjusted_positive_ranks,
#                                                        desired_pr_ranks)
#     np.save('./precisions.npy',precisions)
#    print(i, average_precisions[i])
    mean_average_precision += average_precisions[i]
#     mean_precisions += precisions[i, :]
#     mean_recalls += recalls[i, :]

#   Normalize aggregated metrics by number of queries.
  num_valid_queries = num_queries - num_empty_gt_queries
  mean_average_precision /= num_valid_queries
#   mean_precisions /= num_valid_queries
#   mean_recalls /= num_valid_queries

  return mean_average_precision*100


def AdjustPositiveRanks(positive_ranks):
  """Adjusts positive ranks based on junk ranks.

  Args:
    positive_ranks: Sorted 1D NumPy integer array.
    junk_ranks: Sorted 1D NumPy integer array.

  Returns:
    adjusted_positive_ranks: Sorted 1D NumPy array.
  """
#   if not junk_ranks.size:
#     return positive_ranks

  adjusted_positive_ranks = positive_ranks
#   j = 0
#   for i, positive_index in enumerate(positive_ranks):
#     while (j < len(junk_ranks) and positive_index > junk_ranks[j]):
#       j += 1

#     adjusted_positive_ranks[i] -= j

  return adjusted_positive_ranks

def main(unused_argv):
    _PR_RANKS = (1, 5)
    result = np.load(cmd_args.result_save_path)
    
    # GT 위치
    query_list, index_list, ground_truth, _, _ = dataset.ReadDatasetFile(cmd_args.dataset_file_path)
    num_query_images = len(query_list)
    num_index_images = len(index_list)

    index_list_ = [i.replace("JPG",'npy') for i in index_list]
    query_list_ = [i.replace("JPG",'npy') for i in query_list]

    # 뽑혀진 query feature와 index feature를 pickle로 저장한 위치를 지정.
    query_descriptors_ = pkl.load(open(cmd_args.query_descriptors_path, 'rb'))
    index_descriptors = pkl.load(open(cmd_args.index_descriptors_path, 'rb'))

    # Query 개수에 변동이 생기는 경우, 여기서 handling 하면 됨.
    query_descriptors = query_descriptors_

    # mAP 계산, mAP_start는 시작하는 지점을 의미, 0으로 지정하면 됨.
    
    # 매번 실행시킬때마다 query에서 랜덤 이미지가 나옴   

    query_sample = []

    for _ in range(10):
        i = random.randint(0, len(query_list))  # query_list에서 random number 추출
        rand_image = query_list[i]
        rand_class = query_list[i][:-8]
        print('query sample:', rand_image)    
        pil_img = Image(filename='/home/smart02/Yonsei_Dataset/query/' + rand_class + '/' + rand_image)
        display(pil_img)

        print('Top 2:', result[i][:2])
        for j in result[i][:2]:
            im = Image(filename='/home/smart02/Yonsei_Dataset/index/' + index_list[j][:-8] + '/' + index_list[j])
            display(im)
            print(index_list[j])    

        query_sample.append(query_list[i])
    print(query_sample)

    mAP_start = 0
    #print('mAP', ComputeMetrics(result, ground_truth, query_list, _PR_RANKS, 5, start = mAP_start))
    print('mAP', ComputeMetrics(result, ground_truth, query_list, _PR_RANKS, 10, start = mAP_start))
    print('mAP', ComputeMetrics(result, ground_truth, query_list, _PR_RANKS, 15, start = mAP_start))
    print('mAP', ComputeMetrics(result, ground_truth, query_list, _PR_RANKS, 20, start = mAP_start))
    print('mAP', ComputeMetrics(result, ground_truth, query_list, _PR_RANKS, 50, start = mAP_start))
    print('mAP', ComputeMetrics(result, ground_truth, query_list, _PR_RANKS, 100, start = mAP_start))    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--query_descriptors_path',
      type=str,
      default='/home/smart02/LM/research/delf/delf/python/training/final_query_features_0126.txt')
  parser.add_argument(
      '--index_descriptors_path',
      type=str,
      default='/home/smart02/LM/research/delf/delf/python/training/final_index_features_0126_1.txt')
  parser.add_argument(
      '--dataset_file_path',
      type=str,
      default='/home/smart02/Yonsei_Dataset/Yonsei_Dataset_for_Feature/')
  parser.add_argument(
      '--result_save_path',
      type=str,
      default='/home/smart02/Yonsei_Dataset/rank_result.npy')
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)