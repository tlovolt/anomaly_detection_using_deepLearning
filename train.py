import argparse
import gc
import glob
import logging
import numpy as np
import os
import pandas as pd
import random
import shutil
import sys
import tensorflow as tf
import time
# from algorithm.nets import ResNet_SF, focal_loss, lr_schedule, TL, focal_loss_w_ls
from common.nets import ResNet_SF, focal_loss, lr_schedule, TL, focal_loss_w_ls
from keras import applications
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import compute_class_weight


class train:
    def __init__(self):
        self.seednum = 2020
        self.set_random_seed()
        self.args = self.parse_arguments()

        args = self.args
        self.prev_model_dir = os.path.join(paths.data_dir, args.prev_model)
        self.new_model_dir = os.path.join(paths.data_dir, args.new_model)
        # self.prev_model_dir = self.prev_model_dir
        # self.prev_csv_path = "{}/data/{}/{}".format(paths.irm_env, args.prev_model, args.prev_model)
        self.prev_csv_path = "data/{}/{}.csv".format(args.prev_model, args.prev_model)
        self.new_csv_path = "data/{}/{}.csv".format(args.prev_model, args.new_model)

        # if args.split == 0:
        # self.get_s3_data()
        # self.release_model()

    def set_random_seed(self):
        np.random.seed(self.seednum)
        tf.random.set_seed(self.seednum)
        random.seed(self.seednum)

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--job_name", type=str, required=False, default="M00011_TRAIN_20200904_135250_1", help="job name")
        parser.add_argument("--epochs", type=int, required=False, default=10, help="number of epochs")
        parser.add_argument("--batch_size", type=int, required=False, default=16, help="batch size")
        parser.add_argument("--learning_rate", type=float, required=False, default=1e-4, help="learning rate")
        parser.add_argument("--target_size", type=str, required=False, default="512,512", help="target image size")
        # parser.add_argument("--model", type=str, required=True, help="model name")
        parser.add_argument("--model", type=str, required=False, default="resnet50", help="model name")
        parser.add_argument("--augmentation", action='store_true')
        parser.add_argument("--class_weight", action='store_true')
        parser.add_argument("--color_mode", type=str, required=False, default="rgb")
        parser.add_argument("--checkpoint", action='store_true')
        parser.add_argument("--lr_schedule", action='store_true')
        parser.add_argument("--lr_reduce", action='store_true')
        parser.add_argument("--earlystopping", action='store_true')
        parser.add_argument("--log_training", action='store_false')
        parser.add_argument("--patience", type=int, required=False, default=10, help="patience")
        parser.add_argument("--train", action='store_true')
        parser.add_argument("--test", action='store_true')
        parser.add_argument("--num_models", type=int, required=False, default=5, help="number of models for ensemble")
        parser.add_argument("--split", type=int, required=False, default=0, help="split")
        parser.add_argument("--validation_ratio", type=float, required=False, default=0.2, help="validation ratio")
        # parser.add_argument("--date", type=str, required=True, help="date of data")
        # parser.add_argument("--date", type=str, required=False, default= "20200716", help="date of data")
        parser.add_argument("--prev_model", type=str, required=False, default="M00011", help="previous model name")
        parser.add_argument("--new_model", type=str, required=False, default="M00011", help="next model name")
        args = parser.parse_args()
        return args

    def train_model(self):
        # try:
        args = self.args
        data_labels_path = os.path.join(paths.root_dir, self.prev_csv_path)
        data_labels = pd.read_csv(data_labels_path, index_col=0)

        labels = data_labels[['label']]

        if args.augmentation == True: print('Using data augmentation (horizontal_flip, vertical_flip')
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True if args.augmentation else False,
            vertical_flip=True if args.augmentation else False,
            rescale=1. / 255, interpolation_order=1)

        # data_dir = os.path.join(paths.data_dir, args.prev_model)

        target_size_str = args.target_size.split(',')
        target_size = [int(target_size_str[0]), int(target_size_str[1])]
        input_shape = [int(target_size_str[0]), int(target_size_str[1]), 3 if args.color_mode == 'rgb' else 1]

        sss = StratifiedShuffleSplit(n_splits=args.num_models, test_size=args.validation_ratio,
                                     random_state=self.seednum)
        for s, (train_index, valid_index) in enumerate(sss.split(np.zeros(len(labels)), labels)):
            if s != args.split:
                continue

            # save_dir = os.path.join(paths.root_dir, args.job_name)
            # if not os.path.isdir(self.prev_model_dir): os.makedirs(self.prev_model_dir)

            train_data = data_labels.iloc[train_index]
            valid_data = data_labels.iloc[valid_index]

            train_data_generator = datagen.flow_from_dataframe(train_data, directory=self.prev_model_dir,
                                                               x_col='filename', y_col='label',
                                                               target_size=target_size,
                                                               color_mode=args.color_mode,
                                                               batch_size=args.batch_size,
                                                               shuffle=True)

            valid_data_generator = datagen.flow_from_dataframe(valid_data, directory=self.prev_model_dir,
                                                               x_col='filename', y_col='label',
                                                               target_size=target_size,
                                                               color_mode=args.color_mode,
                                                               batch_size=args.batch_size,
                                                               shuffle=False)

            if args.model == 'resnet20':
                print("Learning Resnet20(SF)")
                model = ResNet_SF(version=1)(input_shape=input_shape)
            elif args.model == 'resnet50':
                print("Learning Resnet50(Transfer Learning)")
                model = TL('resnet50')(input_shape=input_shape)
                # model = TL('resnet50')(input_shape=input_shape, selected_top='activation_40')
                # plot_model(model, to_file='resnet50.png')
            elif args.model == 'densenet121':
                print("Learning DenseNet121(Transfer Learning)")
                model = TL('densenet121')(input_shape=input_shape)
            elif args.model == 'efficientnetb0':
                print("Learning EfficientNetB0(Transfer Learning)")
                model = TL('efficientnetb0')(input_shape=input_shape)

            model.summary()

            callbacks = []
            if args.checkpoint == True:
                print("Using checkpoint (save_best_only, save_weight_only)")
                h5_name = '{}.h5'.format(args.new_model) if args.num_models == 1 else '{}_{}.h5'.format(args.new_model,
                                                                                                        s)
                # save_file_path = os.path.join(self.prev_model_dir, '{}_{}.h5'.format(args.new_model, s))
                save_file_path = os.path.join(self.prev_model_dir, h5_name)
                checkpoint = ModelCheckpoint(filepath=save_file_path,
                                             monitor='val_loss',
                                             verbose=2,
                                             save_best_only=True,
                                             save_weights_only=True)

                callbacks.append(checkpoint)

            if args.lr_schedule == True:
                print("Using lr_scheduler")
                lr_scheduler = LearningRateScheduler(lr_schedule)
                callbacks.append(lr_scheduler)

            if args.lr_reduce == True:
                print("Using lr_reducer")
                lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)
                callbacks.append(lr_reducer)

            if args.earlystopping == True:
                print("Using earlystopping")
                earlystopping = EarlyStopping(monitor='val_loss', patience=args.patience, verbose=2)
                callbacks.append(earlystopping)

            if args.log_training == True:
                print("Using log_training")
                log_path = os.path.join(self.prev_model_dir, args.new_model+'_%02d.log' % s)
                csv_logger = CSVLogger(log_path)
                callbacks.append(csv_logger)

            if args.class_weight == True:
                class_weight = compute_class_weight(class_weight="balanced",
                                                    classes=np.unique(train_data_generator.classes),
                                                    y=train_data_generator.classes)
                print("Using class_weight")
            else:
                class_weight = None

            if args.model == 'resnet20':
                model.compile(loss=[focal_loss(alpha=.75, gamma=2)],
                              optimizer=Adam(lr=lr_schedule(0) if args.lr_schedule else args.learning_rate),
                              metrics=['accuracy'])
            elif args.model == 'resnet50' or args.model == 'densenet121' or args.model == 'efficientnetb0':
                # model.compile(loss="categorical_crossentropy",
                #               optimizer=Adam(args.learning_rate),
                #               metrics=["accuracy"])
                # model.compile(loss=[focal_loss(gamma=1.)],
                #               optimizer=Adam(args.learning_rate),
                #               metrics=["accuracy"])
                model.compile(loss=[focal_loss_w_ls(alpha=.75)],
                              optimizer=Adam(args.learning_rate),
                              metrics=["accuracy"])

            history = model.fit_generator(train_data_generator,
                                          validation_data=valid_data_generator,
                                          epochs=args.epochs,
                                          verbose=2,
                                          workers=4,
                                          class_weight=class_weight,
                                          callbacks=callbacks)

            # history_file = os.path.join(self.prev_model_dir, '{}_{}.pkl'.format(args.new_model, s))
            # history_df = pd.DataFrame(history.history)
            # history_df.to_pickle(history_file)

            # del train_data
            # del valid_data
            # del train_data_generator
            # del valid_data_generator
            model_json = model.to_json()
            json_name = '{}.json'.format(args.new_model) if args.num_models == 1 else '{}_{}.json'.format(
                args.new_model, s)
            with open(os.path.join(self.prev_model_dir, json_name), 'w') as json_file:
                json_file.write(model_json)

            # with open(os.path.join(self.new_model_dir, json_name), 'w') as json_file:

            # del history, model
            # K.clear_session()
            # gc.collect()

            # if s + 1 == args.num_models or args.num_models == 1:
            if s + 1 == args.num_models:

                client = boto3.client(
                    service_name='glue', region_name='DS',
                    aws_access_key_id=os.getenv('AccessKey'),
                    aws_secret_access_key=os.getenv('IAMKEY'),
                    endpoint_url='http://glue.api.datalake.sec.com:8088'
                )

                # args.job_name = "M00002_train_20200903_125741_{}".format(s)
                prefix_job_name = args.job_name.rsplit('_', 1)[0]

                if args.num_models > 1:
                    while True:
                        already_running = False
                        time.sleep(1)
                        for i in range(args.num_models):
                            if s == i and args.num_models > 1:
                                continue
                            print("{}_{}".format(prefix_job_name, i))
                            response = client.get_job_runs(JobName="{}_{}".format(prefix_job_name, i))
                            print(response)
                            if response['JobRuns'][0]['JobRunState'] in ['STARTING', 'RUNNING', 'STOPPING']:
                                already_running = True
                                break

                        if already_running == False:
                            break


                # data_labels.loc[data_labels['label'] == 'not', 'label'] = 'Bad'
                data_labels['label'] = origin_labels
                data_labels.to_csv(os.path.join(paths.root_dir, self.new_csv_path))

                prev_files = [f for f in os.listdir(self.prev_model_dir) if args.prev_model in f]
                for prev_file in prev_files:
                    os.remove(os.path.join(self.prev_model_dir, prev_file))

                if os.path.isdir(self.new_model_dir):
                    shutil.rmtree(self.new_model_dir)

                os.rename(self.prev_model_dir, self.new_model_dir)

                for file in os.listdir(self.new_model_dir):
                    if os.path.splitext(file)[1][1:].upper() not in ['CSV', 'H5', 'JSON']:
                        continue
                    file_path = os.path.join(self.new_model_dir, file)
                    # print("====================================filepath========================================================================================================")
                    # print(file_path)
                    object_name = f'data/{args.new_model}/{file}'

                    if self.s3.is_exists(object_name):
                        self.s3.delete(object_name)

                    self.s3.save_file(file_path, object_name)
                print(args.num_models)
                print(type(args.num_models))
                response = client.create_job(
                    Name=prefix_job_name.replace("TRAIN", "EVALUATION"),
                    Description='resnet50_release_200mm_00000716',
                    Role=os.getenv('AccessKey'),
                    Command={
                        "Name": "slurm.python.app",
                        "ScriptLocation": f'{paths.root_dir}/algorithm/evaluate.py'
                    },
                    DefaultArguments={
                                        "--py-kernel-name": "python36_keras_2.3.1",
                                        "--partition-name": "v100",
                                        "--new_model": args.new_model,
                                        "--num_models":str(args.num_models)
                                        # "--target_size": str(args.target_size)
                                      }
                )
                self.s3.delete('data/{}'.format(args.prev_model))

                response = client.start_job_run(
                    JobName=prefix_job_name.replace("TRAIN", "EVALUATION")
                )

            del train_data
            del valid_data
            del train_data_generator
            del valid_data_generator
            del history, model
            K.clear_session()
            gc.collect()

        #     if not os.path.isdir(self.prev_model_dir):
        #
        #
        #
        # finally:
        #     del train_data
        #     del valid_data
        #     del train_data_generator
        #     del valid_data_generator
        #     del history, model
        #     K.clear_session()
        #     gc.collect()
        #

if __name__ == '__main__':
    service = train()
    service.get_s3_data()
    service.train_model()



