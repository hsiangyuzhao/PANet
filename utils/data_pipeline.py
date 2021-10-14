import os
import torch
import torchio as tio
import numpy as np
from monai.data import Dataset
from monai.transforms import LoadImaged, SpatialPadd, RandSpatialCropd, Compose, RandScaleIntensityd, \
    RandShiftIntensityd, Rand3DElasticd, RandAxisFlipd, ToTensord, MapLabelValued, AddChanneld, \
    NormalizeIntensityd, EnsureTyped
from sklearn.model_selection import train_test_split
from scipy.ndimage import measurements


class SubjectReader:
    """
    Data pipeline based on MONAI framework, used for training.
    """

    def __init__(self, image_root, training_size):
        """
        Args:
            image_root: root for BraTS image storage.
            training_size: the size of 3D patch during training.
                larger training size usually leads to better performance but also causes larger video memory
                occupation, usually size of 128*128*128 is good.
        """
        self.image_dir = image_root
        self.training_size = training_size
        self.subject_list = os.listdir(self.image_dir)

    def get_subjects(self, subject_list, is_training=True):
        print('Subject path: {}'.format(self.image_dir))
        subjects = []
        for index, subject_name in enumerate(subject_list):
            if is_training:  # for training images the label map shall be provided.
                subject = {'t1': os.path.join(self.image_dir, subject_name, subject_name + '_t1.nii.gz'),
                           't2': os.path.join(self.image_dir, subject_name, subject_name + '_t2.nii.gz'),
                           't1ce': os.path.join(self.image_dir, subject_name, subject_name + '_t1ce.nii.gz'),
                           'flair': os.path.join(self.image_dir, subject_name, subject_name + '_flair.nii.gz'),
                           'label': os.path.join(self.image_dir, subject_name, subject_name + '_seg.nii.gz'),
                           'name': subject_name}
            else:
                subject = {'t1': os.path.join(self.image_dir, subject_name, subject_name + '_t1.nii.gz'),
                           't2': os.path.join(self.image_dir, subject_name, subject_name + '_t2.nii.gz'),
                           't1ce': os.path.join(self.image_dir, subject_name, subject_name + '_t1ce.nii.gz'),
                           'flair': os.path.join(self.image_dir, subject_name, subject_name + '_flair.nii.gz'),
                           'name': subject_name}
            subjects.append(subject)
        print('Subjects prepared. Number of subjects: {}'.format(len(subjects)))
        return subjects

    def get_dataset(self, test_size, random_state):
        """
        Get train/validation data split for training.
        Args:
            test_size: the size of test set.
            random_state: random seed for data split, used for reproducibility.
        """
        print('Prepare train & val dataset. Test size: {}; Random state: {}'.format(test_size, random_state))
        train_transform = self.get_training_transform()
        val_transform = self.get_evaluation_transform(inference=False)
        train_subject_list, val_subject_list = train_test_split(self.subject_list,
                                                                test_size=test_size,
                                                                random_state=random_state)
        print('Train subjects: {}'.format(train_subject_list))
        print('Val subjects: {}'.format(val_subject_list))
        train_subjects = self.get_subjects(train_subject_list)
        val_subjects = self.get_subjects(val_subject_list)
        trainset = Dataset(data=train_subjects, transform=train_transform)
        valset = Dataset(data=val_subjects, transform=val_transform)
        print('Dataset prepared. Trainset length: {}; Valset length: {}'.format(len(trainset), len(valset)))
        return trainset, valset

    def get_testset(self):
        """
        deprecated.
        """
        inference_transform = self.get_evaluation_transform(inference=True)
        subjects = self.get_subjects(self.subject_list, is_training=False)
        testset = Dataset(data=subjects, transform=inference_transform)
        print('BraTS Validation dataset prepared. Length: {}'.format(len(testset)))
        return testset

    def get_trainset(self):
        """
        Get the entire BraTS training set.
        Used when the optimal configuration has been found to train on the whole dataset.
        """
        train_transform = self.get_training_transform()
        subjects = self.get_subjects(self.subject_list)
        trainset = Dataset(data=subjects, transform=train_transform)
        # trainset = PersistentDataset(data=subjects, transform=train_transform, cache_dir='./data_cache')
        print('BraTS Training dataset prepared. Length: {}'.format(len(trainset)))
        return trainset

    @staticmethod
    def get_evaluation_transform(inference=False):
        """
        Transform for validation data during training.
        """
        training_keys = ('t1', 't2', 't1ce', 'flair', 'label')  # BraTS training set contains label
        image_keys = ('t1', 't2', 't1ce', 'flair')
        if inference:
            load = LoadImaged(keys=image_keys)
            pad = SpatialPadd(keys=image_keys, spatial_size=(240, 240, 160))
        else:
            load = LoadImaged(keys=training_keys)
            pad = SpatialPadd(keys=training_keys, spatial_size=(240, 240, 160))
        if inference:
            preprocess = Compose([
                load,
                AddChanneld(keys=image_keys),
                pad,
                NormalizeIntensityd(keys=image_keys),
                ToTensord(keys=image_keys)
            ])
        else:
            preprocess = Compose([
                load,
                AddChanneld(keys=training_keys),
                pad,
                MapLabelValued(keys='label', orig_labels=(0, 1, 2, 4), target_labels=(0, 1, 2, 3)),
                NormalizeIntensityd(keys=image_keys),
                ToTensord(keys=training_keys)
            ])
        return preprocess

    def get_training_transform(self):
        """
        Data loading and augmentation during training.
        """
        training_keys = ('t1', 't2', 't1ce', 'flair', 'label')  # BraTS training set contains label
        image_keys = ('t1', 't2', 't1ce', 'flair')
        augment = Compose([
            LoadImaged(keys=training_keys),
            AddChanneld(keys=training_keys),
            MapLabelValued(keys='label', orig_labels=(0, 1, 2, 4), target_labels=(0, 1, 2, 3)),
            RandSpatialCropd(keys=training_keys, roi_size=self.training_size, random_size=False),
            NormalizeIntensityd(keys=image_keys),
            RandAxisFlipd(keys=training_keys, prob=0.5),
            RandScaleIntensityd(keys=image_keys, factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=image_keys, offsets=0.1, prob=0.5),
            Rand3DElasticd(keys=training_keys, prob=0.5, mode='nearest',
                           sigma_range=(1, 20), magnitude_range=(0.3, 2.3),
                           rotate_range=(np.pi / 6, np.pi / 6, np.pi / 6),
                           shear_range=(0.1, 0.1, 0.1),
                           scale_range=(0.2, 0.2, 0.2)),
            EnsureTyped(keys=training_keys)
        ])
        return augment


class InferenceReader:
    """
    Data pipeline during inference based on TorchIO framework. Used for generate predictions and submit to the portal.
    """

    def __init__(self, image_root, patch_test=True):
        """
        Args:
            image_root: root for BraTS image storage.
            patch_test: Boolean value. Whether to use patch-based pipeline during inference.
                If set to `False`, inference will be conducted on the whole volume, or inference will be based on
                image patches just as training.
        """
        self.image_dir = image_root
        self.subject_list = os.listdir(self.image_dir)
        self.patch_test = patch_test

    def get_subjects(self, subject_list, is_training=False):
        print('Subject path: {}'.format(self.image_dir))
        subjects = []
        for index, subject_name in enumerate(subject_list):
            if is_training:
                subject = tio.Subject(
                    t1=tio.ScalarImage(
                        os.path.join(self.image_dir, subject_name, subject_name + '_t1.nii.gz')),
                    t2=tio.ScalarImage(
                        os.path.join(self.image_dir, subject_name, subject_name + '_t2.nii.gz')),
                    t1ce=tio.ScalarImage(
                        os.path.join(self.image_dir, subject_name, subject_name + '_t1ce.nii.gz')),
                    flair=tio.ScalarImage(
                        os.path.join(self.image_dir, subject_name, subject_name + '_flair.nii.gz')),
                    label=tio.LabelMap(
                        os.path.join(self.image_dir, subject_name, subject_name + '_seg.nii.gz')),
                    name=subject_name
                )
            else:
                subject = tio.Subject(
                    t1=tio.ScalarImage(
                        os.path.join(self.image_dir, subject_name, subject_name + '_t1.nii.gz')),
                    t2=tio.ScalarImage(
                        os.path.join(self.image_dir, subject_name, subject_name + '_t2.nii.gz')),
                    t1ce=tio.ScalarImage(
                        os.path.join(self.image_dir, subject_name, subject_name + '_t1ce.nii.gz')),
                    flair=tio.ScalarImage(
                        os.path.join(self.image_dir, subject_name, subject_name + '_flair.nii.gz')),
                    name=subject_name
                )
            subjects.append(subject)
        print('Subjects prepared. Number of subjects: {}'.format(len(subjects)))
        return subjects

    def get_dataset(self, test_size, random_state):
        print('Prepare val dataset for cross-validation. Test size: {}; Random state: {}'.format(test_size,
                                                                                                 random_state))
        preprocess = self.get_preprocessing_transform()
        _, val_subject_list = train_test_split(self.subject_list, test_size=test_size, random_state=random_state)
        print('Val subjects: {}'.format(val_subject_list))
        transform = preprocess
        val_subjects = self.get_subjects(val_subject_list, is_training=True)
        valset = tio.SubjectsDataset(subjects=val_subjects, transform=transform)
        print('Dataset prepared. Valset length: {}'.format(len(valset)))
        return valset

    def get_testset(self):
        preprocess = self.get_preprocessing_transform()
        subjects = self.get_subjects(self.subject_list)
        transform = preprocess
        testset = tio.SubjectsDataset(subjects=subjects, transform=transform)
        print('BraTS Validation dataset prepared. Length: {}'.format(len(testset)))
        return testset

    def get_trainset(self):
        preprocess = self.get_preprocessing_transform()
        subjects = self.get_subjects(self.subject_list, is_training=True)
        transform = preprocess
        testset = tio.SubjectsDataset(subjects=subjects, transform=transform)
        print('BraTS Training dataset prepared. Length: {}'.format(len(testset)))
        return testset

    def get_preprocessing_transform(self):
        if self.patch_test:
            preprocess = tio.Compose([
                tio.RemapLabels({4: 3}),
                tio.CropOrPad((240, 240, 160)),
                tio.ZNormalization(),
                tio.OneHot(num_classes=4),
            ])
        else:
            preprocess = tio.Compose([
                tio.RemapLabels({4: 3}),
                tio.ZNormalization(),
                tio.OneHot(num_classes=4),
            ])
        return preprocess


def overlap_labels(target_tensor):
    """
    receives discrete targets and returns overlap ones
    :param target_tensor: shape (B, C, H, W, D), one-hot-encoded label maps with discrete BraTS labels (ET, NET/NCR, ED)
    :return: overlapped label maps (ET, TC, WT)
    """
    bg = target_tensor[:, 0:1, ...]
    # necrotic and non-enhancing tumor core
    net = target_tensor[:, 1:2, ...]
    # peritumoral edema
    ed = target_tensor[:, 2:3, ...]
    # GD-enhancing tumor
    et = target_tensor[:, 3:4, ...]

    tc = et + net
    wt = et + net + ed
    targets = torch.cat([bg, et, tc, wt], dim=1)
    return targets


def discretize_labels(target_tensor):
    """
    receives overlap targets and returns discrete ones
    :param target_tensor: shape (B, C, H, W, D), one-hot-encoded label maps with overlapped BraTS labels (ET, TC, WT)
    :return: discrete label maps (ET, NET/NCR, ED)
    """
    labelmap = torch.zeros(target_tensor.shape[1:], device=target_tensor.device)
    # enhancing tumor
    et = target_tensor[1, ...]
    # tumor core
    tc = target_tensor[2, ...]
    # whole tumor
    wt = target_tensor[3, ...]

    net = torch.logical_and(tc, torch.logical_not(et))
    ed = torch.logical_and(wt, torch.logical_not(tc))

    labelmap[et.to(torch.bool)] = 4  # float type cannot be passed as index
    labelmap[net] = 1
    labelmap[ed] = 2
    return labelmap


class RemoveMinorConnectedComponents:
    """
    Remove minor connected components in the volume, used as the post-processing of predicted enhancing tumors.
    """
    def __init__(self, thr):
        """
        Args:
            thr: threshold.
        """
        self.thr = thr

    def __call__(self, pred):
        # pred.shape: (1, num_classes, H, W, (D)...)
        predictions = pred.clone().cpu()
        num_classes = predictions.shape[1]
        pred_list = [predictions[:, 0: 1, ...]]  # background channel
        for c in range(1, num_classes):
            class_pred = predictions[:, c: c + 1, ...]  # get corresponding class predictions
            labeled_cc, num_cc = measurements.label(class_pred.numpy())  # acquire connect components

            for cc in range(1, num_cc + 1):
                single_cc = (labeled_cc == cc) * 1.0  # retrieve every connect component
                single_volume = np.sum(single_cc)
                if single_volume < self.thr:
                    index = ~(single_cc == 1)
                    class_pred *= index  # remove the component if the size is smaller than the threshold
            pred_list.append(class_pred)

        return torch.cat(pred_list, dim=1).to(pred.device)


class ETThresholdSuppression:
    """
    Replace predicted enhancing tumor with NET/NCR to avoid false positives.
    """

    def __init__(self, thr, is_overlap=True, global_replace=True):
        """
        Args:
            thr: threshold for ET suppression, typically 300 or 500.
            is_overlap: Boolean value, whether the label maps are overlapped or discrete.
            global_replace: Boolean value.
                If set to `True`, the suppression is based on the entire volume, or the suppression is based on
                connected components. Default `True`.
        """
        self.thr = thr
        self.is_overlap = is_overlap
        self.gr = global_replace

    def global_replace(self, pred):
        # replace enhancing tumor by necrosis according to the whole volume of all enhancing tumor
        # pred.shape: (1, num_classes, H, W, (D)...)
        predictions = pred.clone()
        if self.is_overlap:
            et_voxels = torch.sum(predictions[:, 1, ...])
            if et_voxels < self.thr:
                predictions[:, 2][predictions[:, 1] == 1.0] = 1.0
                predictions[:, 3][predictions[:, 1] == 1.0] = 1.0
                predictions[:, 1] = 0.0
        else:
            # When optimizing directly on vanilla labels, channel 1 denotes net/ncr
            # channel 2 denotes ed and channel 3 denotes et
            # Thus if predicted et volume is smaller than threshold, replace et with net/ncr
            et_voxels = torch.sum(predictions[:, 3, ...])
            if et_voxels < self.thr:
                predictions[:, 1][predictions[:, 3] == 1.0] = 1.0
                predictions[:, 3] = 0.0
        return predictions

    def local_replace(self, pred):
        # replace enhancing tumor by necrosis according to the size of regions of enhancing tumor regions
        # pred.shape: (1, num_classes, H, W, (D)...)
        predictions = pred.clone().cpu()
        if self.is_overlap:
            bg = predictions[:, 0:1, ...]
            et = predictions[:, 1:2, ...]
            tc = predictions[:, 2:3, ...]
            wt = predictions[:, 3:4, ...]
            labeled_cc, num_cc = measurements.label(et.cpu().numpy())
            for cc in range(1, num_cc + 1):
                single_cc = (labeled_cc == cc) * 1.0
                single_volume = np.sum(single_cc)
                if single_volume < self.thr:
                    index = ~(single_cc == 1)  # ~ means reverse operations
                    # ~index represents the regions that should be replaced by net/ncr
                    et *= index
                    tc += ~index
                    wt += ~index
                    tc[tc > 1] = 1.0
                    wt[wt > 1] = 1.0
            final = torch.cat([bg, et, tc, wt], dim=1)
        else:
            bg = predictions[:, 0:1, ...]
            net = predictions[:, 1:2, ...]
            ed = predictions[:, 2:3, ...]
            et = predictions[:, 3:4, ...]
            labeled_cc, num_cc = measurements.label(et.cpu().numpy())
            for cc in range(1, num_cc + 1):
                single_cc = (labeled_cc == cc) * 1.0
                single_volume = np.sum(single_cc)
                if single_volume < self.thr:
                    index = ~(single_cc == 1)
                    et *= index
                    net += ~index
                    net[net > 1] = 1.0
            final = torch.cat([bg, net, ed, et], dim=1)
        return final.to(pred.device)

    def __call__(self, pred):
        if self.gr:
            return self.global_replace(pred)
        else:
            return self.local_replace(pred)
