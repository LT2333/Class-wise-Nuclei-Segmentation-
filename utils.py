import os
import sys
import numpy as np
from tqdm import tqdm
from skimage import io
from PIL import Image
import numpy as np
import torch
from skimage.morphology import label
import matplotlib.pylab as plt
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_recall_fscore_support

class Config():

    name = None

    img_width = 256
    img_height = 256

    img_channel = 3

    batch_size = 16

    learning_rate = 1e-3
    learning_momentum = 0.9
    weight_decay = 1e-4

    shuffle = False

    def __init__(self):
        self.IMAGE_SHAPE = np.array([
            self.img_width, self.img_height, self.img_channel
        ])

    def display(self):
        """Display Configuration values"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

# Configurations

class Option(Config):
    """Configuration for training on Kaggle Data Science Bowl 2018
    Derived from the base Config class and overrides specific values
    """

    # root dir of training and validation set
    # root_dir = 'data/full_data/Train_out'
    root_dir = "data/full_full_data/Train_out"

    # root dir of testing set
    test_dir = 'data/full_full_data/Test_out'

    # save segmenting results (prediction masks) to this folder
    results_dir = 'results'

    num_workers = 4     	# number of threads for data loading
    shuffle = True      	# shuffle the data set
    batch_size = 10
    epochs = 80		# number of epochs to train
    is_train = False	        # True for training, False for testing/inferrence
    save_model = True  	# True for saving the model, False for not saving the model

    n_gpu = 1				# number of GPUs

    learning_rate = 1e-3	# learning rage
    weight_decay = 1e-4		# weight decay

    pin_memory = True   	# use pinned (page-locked) memory. when using CUDA, set to True

    is_cuda = False  # True --> GPU

    # is_cuda = torch.cuda.is_available()  	# True --> GPU
    num_gpus = torch.cuda.device_count()  	# number of GPUs
    checkpoint_dir = "./checkpoint"  		# dir to save checkpoints
    dtype = torch.cuda.FloatTensor if is_cuda else torch.Tensor  # data type
    is_colored = True         # class-wise segmentation or pure segmentation

class Utils(object):
    def __init__(self, stage1_train_src, stage1_train_dest, stage1_test_src, stage1_test_dest):
        self.opt = Option
        self.stage1_train_src = stage1_train_src
        self.stage1_train_dest = stage1_train_dest
        self.stage1_test_src = stage1_test_src
        self.stage1_test_dest = stage1_test_dest

    # Combine all separated masks into one mask
    def assemble_masks(self, path):
        # mask = np.zeros((self.config.IMG_HEIGHT, self.config.IMG_WIDTH), dtype=np.uint8)
        mask = None
        for i, mask_file in enumerate(next(os.walk(os.path.join(path, 'masks')))[2]):
            mask_ = Image.open(os.path.join(path, 'masks', mask_file)).convert("RGB")
            # mask_ = mask_.resize((self.config.IMG_HEIGHT, self.config.IMG_WIDTH))
            mask_ = np.asarray(mask_)
            if i == 0:
                mask = mask_
                continue
            mask = mask | mask_
        # mask = np.expand_dims(mask, axis=-1)
        return mask

    # read all training data and save them to other folder
    def prepare_training_data(self):
        # get imageId
        train_ids = next(os.walk(self.stage1_train_src))[1]

        # read training data
        X_train = []
        Y_train = []
        print('reading training data starts...')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(train_ids)):
            path = os.path.join(self.stage1_trpathain_src, id_)
            dest = os.path.join(self.stage1_train_dest, id_)
            img = Image.open(os.path.join(path, 'images', id_ + '.png')).convert("RGB")
            mask = self.assemble_masks(path)
            img.save(os.path.join(dest, 'image.png'))
            Image.fromarray(mask).save(os.path.join(dest, 'mask.png'))

        print('reading training data done...')

    # read testing data and save them to other folder
    def prepare_testing_data(self):
        # get imageId
        test_ids = next(os.walk(self.stage1_test_src))[1]
        # read training data
        print('reading testing data starts...')
        sys.stdout.flush()
        for n, id_ in tqdm(enumerate(test_ids)):
            path = os.path.join(self.stage1_test_src, id_, 'images', id_+'.png')
            dest = os.path.join(self.stage1_test_dest, id_)
            if not os.path.exists(dest):
                os.mkdir(dest)
            img = Image.open(path).convert("RGB")
            img.save(os.path.join(dest, 'image.png'))

        print('reading testing data done...')


def compute_other_scores(predictions, img_ids):
    """
    compute precison, recall and F1
    """
    if Option.is_train:
        file_name = 'val_other_scores.txt'
    else:
        file_name = 'test_other_scores.txt'

    with open(file_name, 'a') as file:
        precs= []
        recalls = []
        F1s= []
        for i in range(0, len(img_ids)):
            pred = predictions[i]
            img_id = img_ids[i]
            if Option.is_train:
                path = Option.root_dir
            else:
                path = Option.test_dir

            if Option.is_colored:
                mask_path = os.path.join(path, img_id, 'label.npy')
                mask = np.load(mask_path)
                for j in range(256):
                    results = precision_recall_fscore_support(mask[j], pred[j], average='macro',zero_division=1)
                    precs.append(results[0])
                    recalls.append(results[1])
                    F1s.append(results[2])
            else:
                mask_path = os.path.join(path, img_id, 'mask.png')
                mask = np.asarray(Image.open(mask_path).convert('L'), dtype=np.bool)
                results = precision_recall_fscore_support(mask, pred, average='macro', zero_division=1)
                precs.append(results[0])
                recalls.append(results[1])
                F1s.append(results[2])

        if len(precs) > 0:
            avg = sum(precs)/len(precs)
            file.write( str(avg) + '\n')
            print('Avg Precison:',str(avg))
        if len(recalls) > 0:
            avg = sum(recalls) / len(recalls)
            file.write(str(avg) + '\n')
            print('Avg Recalls:', str(avg))
        if len(F1s) > 0:
            avg = sum(F1s)/len(F1s)
            file.write( str(avg) + '\n')
            print('Avg F1s:',str(avg))


def compute_accuracy(predictions, img_ids):

    if Option.is_train:
        file_name = 'val_accuracy.txt'
    else:
        file_name = 'test_accuracy.txt'
    with open(file_name, 'a') as file:
        accs = []
        for i in range(0, len(img_ids)):
            pred = predictions[i]
            img_id = img_ids[i]
            if Option.is_train:
                label_path = os.path.join(Option.root_dir, img_id, 'label.npy')
            else:
                label_path = os.path.join(Option.test_dir, img_id, 'label.npy')
            label = np.load(label_path)
            sum = np.sum(pred == label)
            accs.append(float(sum/(255*255)))

        if len(accs) > 0:
            avg = np.sum(accs)/len(accs)
            file.write( str(avg) + '\n')
            print('Avg Accuracy:',str(avg))

def compute_iou(predictions, img_ids):

    if Option.is_train:
        file_name = 'val_IOU.txt'
    else:
        file_name = 'test_IOU.txt'

    with open(file_name, 'a') as file:
        ious = []
        for i in range(0, len(img_ids)):
            pred = predictions[i]
            img_id = img_ids[i]
            if Option.is_train:
                path = Option.root_dir
            else:
                path = Option.test_dir

            if Option.is_colored:
                mask_path = os.path.join(path, img_id, 'label.npy')
                mask = np.load(mask_path)
                for j in range(256):
                    ious.append(jaccard_score(mask[j], pred[j], average='micro'))
            else:
                mask_path = os.path.join(path, img_id, 'mask.png')
                mask = np.asarray(Image.open(mask_path).convert('L'), dtype=np.bool)
                ious.append(jaccard_score(mask,pred,average='micro'))
        if len(ious) > 0:
            avg = sum(ious)/len(ious)
            file.write( str(avg) + '\n')
            print('Avg IOU:',str(avg))



def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def save_imgs(preds_test_upsampled, test_ids):
    """
    :param preds_test_upsampled: list, for each elements, numpy array (Width, Height)
    :param test_ids: list, for each elements, image id
    """
    # save as imgs
    for i in range(0, len(test_ids)):
        path = os.path.join(Option.results_dir, test_ids[i])
        if not os.path.exists(path):
            os.mkdir(path)

        if Option.is_colored:
            plt.imsave(os.path.join(path, 'prediction.png'), preds_test_upsampled[i], vmin=0, vmax=7, cmap='jet')
        else:
            plt.imsave(os.path.join(path, 'prediction.png'), preds_test_upsampled[i], cmap='gray')
    print('Finshed saving predictions')


