import os
import argparse
import pickle
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow.python.platform import app
# print('Hello World')
cmd_args = None


def landmark_directories(dataset_file_path, image_type):
    if image_type == 'query':
        dataset = sorted(os.listdir(dataset_file_path + image_type))
    else:
        dataset = sorted(os.listdir(dataset_file_path + image_type))
   
    try:       
        if '.ipynb_checkpoint' in dataset:
            dataset.remove('.ipynb_checkpoint')

    except:
        pass
    
    lm_dirs = []
    for landmarks in dataset:
        if landmarks == '.ipynb_checkpoint':
            landmarks = None
        else:
            lm_dirs.append(dataset_file_path +  image_type + "/" + landmarks)

    if None in lm_dirs:
        try:
            lm_dirs.remove(None)
        except:
            pass
    
    return dataset, lm_dirs

    
def grouped_images_and_classes(dataset, lm_dirs):
#     lm_dirs = lm_dirs + "/" + image_type
    grouped_imgs = []  # list of grouped images
    for imgs in lm_dirs:
    #     print(imgs)
        each_lm_imgs = []
        lm_ims = sorted(os.listdir(imgs))
        each_lm_imgs.append(lm_ims)
    #     each_lm_imgs.append(imgs)
        for im_sets in each_lm_imgs:
            grouped_imgs.append(im_sets)

    # grouped_imgs 

    extended_lms_ = [] # list of grouped landmarks
    for lms, folders in zip(dataset, grouped_imgs):
        extended_lms = []
        for _ in range(len(folders)):
            extended_lms.append(lms)
        extended_lms_.append(extended_lms)

    return grouped_imgs, extended_lms_


def obtain_all_landmarks_images(grouped_imgs, extended_lms):
    all_imgs, all_lms = [], []
    for im_list, lm_list in zip(grouped_imgs, extended_lms):
    #     for ims in im_list:
        all_imgs.extend(im_list)
        all_lms.extend(lm_list)
        
    return all_imgs, all_lms


def create_ground_truth(query_images, index_landmarks):
    gt_dict = dict()
    all_query_imgs = query_images
    for query_true in all_query_imgs:

        index_list = []

        for index, file in enumerate(index_landmarks):

            if file == query_true[:-8]:
                index_list.append(index)

        name = query_true
        gt_dict[name] = np.array(index_list)

    gt_list = []
    for file in all_query_imgs:   
        sample_dict = dict()  
        for key, value in gt_dict.items():

            if file == key:
                name = file
                sample_dict[name] = value

        gt_list.append(sample_dict)
            
    print(gt_list, len(gt_list))
    assert len(gt_list) == len(query_images)
    return gt_list



def main(argv):
    
    query_dataset, query_lm_dirs = landmark_directories(cmd_args.dataset_file_path, cmd_args.first_image_file_type)
    index_dataset, index_lm_dirs = landmark_directories(cmd_args.dataset_file_path, cmd_args.second_image_file_type)
#     print('lm_dirs size:', lm_dirs[:5])
    query_grouped_imgs, query_extended_lms_ = grouped_images_and_classes(query_dataset, query_lm_dirs)
    index_grouped_imgs, index_extended_lms_ = grouped_images_and_classes(index_dataset, index_lm_dirs)
    
    
    all_query_imgs, all_query_lms = obtain_all_landmarks_images(query_grouped_imgs, query_extended_lms_)
    all_index_imgs, all_index_lms = obtain_all_landmarks_images(index_grouped_imgs, index_extended_lms_)
#     print('dataset size:', len(dataset))
#     print('lm_dirs size:', len(lm_dirs))
#     print('query_grouped_imgs size:', len(query_grouped_imgs))
#     print('query_extended_lms_ size:', len(query_extended_lms_))
    
#     print('index_grouped_imgs size:', len(index_grouped_imgs))
#     print('index_extended_lms_ size:', len(index_extended_lms_))
    
#     print('all_query_imgs size:', len(all_query_imgs))
#     print('all_query_lms size:', len(all_query_lms))
    
#     print('all_index_imgs size:', len(all_index_imgs))
#     print('all_index_lms size:', len(all_index_lms))
    ground_truth = create_ground_truth(all_query_imgs, all_index_lms)
    
    with open(cmd_args.output_directory+"query_images.pkl", 'wb') as pickle_file:  # wb allows pickle to receive numeric data
        pickle.dump(all_query_imgs, pickle_file)
    
    with open(cmd_args.output_directory+"query_landmarks.pkl", 'wb') as pickle_file:  # wb allows pickle to receive numeric data
        pickle.dump(all_query_lms, pickle_file)
        
    with open(cmd_args.output_directory+"index_images.pkl", 'wb') as pickle_file:  # wb allows pickle to receive numeric data
        pickle.dump(all_index_imgs, pickle_file)
    
    with open(cmd_args.output_directory+"index_landmarks.pkl", 'wb') as pickle_file:  # wb allows pickle to receive numeric data
        pickle.dump(all_index_lms, pickle_file)
        
    with open(cmd_args.output_directory+"ground_truth.pkl", 'wb') as pickle_file:  # wb allows pickle to receive numeric data
        pickle.dump(ground_truth, pickle_file)

        
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--first_image_file_type',
      type=str,
      default='query',
      help="""
      'query' to get query images .
      """)
  parser.add_argument(
      '--second_image_file_type',
      type=str,
      default='index',
      help="""
      'index' to get index images.
      """)
  parser.add_argument(
      '--dataset_file_path',
      type=str,
      default='/tmp/Last_Dataset',
      help="""
      Dataset file for Revisited Oxford or Paris dataset, in .mat format.
      """)
  parser.add_argument(
      '--output_directory',
      type=str,
      default='/tmp/destination_file',
      help="""
      Dataset file for Revisited Oxford or Paris dataset, in .mat format.
      """)
      
#   parser.add_argument(
#       '--images_dir',
#       type=str,
#       default='/tmp/images',
#       help="""
#       Directory where dataset images are located, all in .jpg format.
#       """)
#   parser.add_argument(
#       '--output_features_dir',
#       type=str,
#       default='/tmp/features',
#       help="""
#       Directory where DELF features will be written to. Each image's features
#       will be written to a file with same name, and extension replaced by .delf.
#       """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
