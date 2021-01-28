"""

이미지가 위치한 폴더를 지정하면, 클래스에 각각 0, 1 .... n개의 label을 달고,
각 클래스에 해당하는 파일명을 가지고 있는 csv파일을 만들어줍니다.

이를 통해 만들어지는 csv파일이 있어야만 학습에 사용되는 Tfrecord를 제작할 수 있습니다.

"""

import os

from absl import app
from absl import flags

import pandas as pd

from tqdm import tqdm



FLAGS = flags.FLAGS

flags.DEFINE_string('train_directory', '/tmp/', 'data directory.')
flags.DEFINE_string('output_directory', '/tmp/',
                                        'train dataset에 대한 정보가 담긴 csv가 저장될 위치')

#FLAGS.train_directory


def main(unused_argv):
    
    print("Making csv file.....")
    
    # directory를 train directory로 변경
    os.chdir(FLAGS.train_directory)
    
    # 데이터셋에 대한 정보를 담을 data frame을 구성, column은 landmark_id와 image의 파일 이름으로 구성됨.
    raw_data = {'landmark_id' : [], 'images' : []}
    df = pd.DataFrame(raw_data)
    df['landmark_id'] = df['landmark_id'].astype('int')
    
    folder_location = FLAGS.train_directory
    folder_list = os.listdir(folder_location)
    folder_list.remove('.ipynb_checkpoints')
    folder_list = sorted(folder_list)
    
    print("Find ", len(folder_list), " classes!")
    
    # Using for loop to all folder in folder_location
    for index, folder in tqdm(enumerate(folder_list)):
        #print(index)
        image_list = sorted(os.listdir(folder)) 
        image_name_concat = ''
        for list_element in image_list:
            image_name_concat = image_name_concat + ' ' + list_element[:-4]
        df.loc[int(index)] = [int(index), image_name_concat]

    
    print("Finish!! Check output directory.")
    print("Output file: ", FLAGS.output_directory)
    df.to_csv(FLAGS.output_directory, index = False, encoding = 'euc-kr')

if __name__ == '__main__':
    app.run(main)
