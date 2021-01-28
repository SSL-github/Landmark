# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Extracts DELF features from a list of images, saving them to file.

The images must be in JPG format. The program checks if descrisptors already
exist, and skips computation for those.
"""

from delf.python.detect_to_retrieve import dataset
import os
from tqdm import tqdm
import numpy as np
import argparse
from tensorflow.python.platform import app
import sys
import time
import pickle
cmd_args = None

def main(unused_argv):
  # Read list of images.
 
  query_list, index_list, ground_truth, _, _ = dataset.ReadDatasetFile(cmd_args.pickle_path)
  index_list_ = [i.replace("jpg",'npy') for i in index_list]
  query_list_ = [i.replace("jpg",'npy') for i in query_list] 
  
  print('make path of images...')
  if cmd_args.name == 'index':
    paths_ = []
    for i in index_list_:
      image_paths=os.path.join(cmd_args.index_features_path, i)
      paths_.append(image_paths)
  
  if cmd_args.name == 'query':
  # Parse DelfConfig proto.
    paths_ = []
    for i in query_list_:
      image_paths=os.path.join(cmd_args.query_features_path, i)
      paths_.append(image_paths)
  print('finish making path of features!'+str(len(paths_)))
  
  print('start making pickle of images...')  
  descriptors=[] 
  for j in tqdm(range(len(paths_))):
    descriptor =np.load(paths_[j])
    descriptors.append(descriptor)
    
  with open(cmd_args.output_dir, 'wb') as f:
    pickle.dump(descriptors, f)  
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--pickle_path',
      type=str,
      default='/home/smart02/Last_Dataset/Last_additional_dataset_for_features/',
      help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
  parser.add_argument(
      '--name',
      type=str,
      default='query',
      help="""
      Path to list of images whose DELF features will be extracted.
      """)
  parser.add_argument(
      '--index_features_path',
      type=str,
      default='/home/smart02/LM/research/delf/delf/python/training/list_features_anes/index_features_100/',
      help="""
      Path to list of images whose DELF features will be extracted.
      """)  
  parser.add_argument(
      '--query_features_path',
      type=str,
      default='/home/smart02/LM/research/delf/delf/python/training/list_features_anes/query_features_100/',
      help="""
      Path to list of images whose DELF features will be extracted.
      """) 
  parser.add_argument(
      '--output_dir',
      type=str,
      default='test_features')

  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
