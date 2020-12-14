import argparse
import boto3
import numpy as np
import os
import pandas as pd
from botocore.config import Config
from common import paths
from datetime import datetime
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_model", type=str, required=False, default="M00000", help="new model name")
    parser.add_argument("--target_size", type=str, required=False, default="224,224", help="target image size")
    parser.add_argument("--num_models", type=int, required=False, default=5, help="number of models for ensemble")
    # parser.add_argument("--color_mode", type=str, required=False, default="rgb", help="single=grayscale, ensemble=rgb")
    args = parser.parse_args()
    return args


def evaluate(args):
# set-up
    start_time = datetime.now()

    path = os.path.join(paths.data_dir, args.new_model)
    data_labels_path = os.path.join(path, args.new_model + '_test.csv')
    data_labels_path = data_labels_path.replace("\\", "/")
    data_labels = pd.read_csv(data_labels_path)

    target_size_str = args.target_size.split(',')
    target_size = [int(target_size_str[0]), int(target_size_str[1])]
    test_generator = ImageDataGenerator(rescale=1./255)
    test_data_generator = test_generator.flow_from_dataframe(data_labels, directory=path,
                                                             x_col='filename', y_col='label',
                                                             target_size=target_size,
                                                             color_mode='grayscale' if args.num_models==1 else 'rgb',
                                                             batch_size=16, shuffle=False)
# load model from list
    json_files = []
    h5_files = []
    if args.num_models == 1:
        json = os.path.join(path, args.new_model + ".json")
        json = json.replace("\\", "/")
        json_files.append(json)
        h5 = os.path.join(path, args.new_model + ".h5")
        h5 = h5.replace("\\", "/")
        h5_files.append(h5)
    else:
        for i in range(args.num_models):
            json = os.path.join(path, args.new_model + "_" + str(i) + ".json")
            json = json.replace("\\", "/")
            json_files.append(json)
            h5 = os.path.join(path, args.new_model + "_" + str(i) + ".h5")
            h5 = h5.replace("\\", "/")
            h5_files.append(h5)

# load and predict
    cnt = 1
    class_predictions = []
    for i, j in zip(json_files, h5_files):
        with open(i, "r") as json_file:
            loaded_model_json = json_file.read()
            json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(j)
        print(str(cnt) + " model loaded")
        cnt += 1
        model.compile(loss='categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['accuracy'])
        class_predictions.append(model.predict_generator(test_data_generator))

    class_prediction = np.array(class_predictions).mean(axis=0)
    y_pred = np.argmax(class_prediction, axis=1)

    target_names = ['Bad', 'Good']
    df = pd.DataFrame({'filename': test_data_generator.filenames,
                       'FCST_ITEM_ID': [target_names[i] for i in y_pred],
                       'FCST_YLD': class_prediction[:, target_names.index('Bad')]})

# process & save result
    result = pd.merge(data_labels, df, on='filename')
    del result['Unnamed: 0']
    result['difference'] = np.where(result['label'] == result['FCST_ITEM_ID'], '', '1')
    # difference = result['difference'].value_counts()
    # print(difference)
    # pd.concat(result, difference)

    result_csv = os.path.join(path, args.new_model + "_result.csv")
    result.to_csv(result_csv)

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))


if __name__ == '__main__':
    args = parse_arguments()
    evaluate(args)
