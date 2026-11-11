"""Computes pre-processed images and features once and stores them on disk, so later runs only load them.

Layout of the cache (one folder per pre-processing setting and data split)::

    cache/<prep key>/<split>/<subject>/
        images/T1w.nii.gz, T2w.nii.gz, BrainMask.nii.gz, GroundTruth.nii.gz   (registered to the atlas)
        voxels_seed<seed>.npz          brain voxel indices, their labels and the sampled training voxels
        features/<group>_v<version>_<params>.npz           features of every brain voxel
        features/<group>_v<version>_<params>_fit<ids>.npz  same, for a fold-dependent group fitted on <ids>

Rules that keep the cache correct:

- The key of each file contains everything that changes its content (pre-processing settings, seed, group
  version and parameters, training subjects), so a changed setting never reuses an old file.
- Only voxels inside the brain mask are stored, classified and evaluated; voxels outside are background.
- The training voxels are sampled once per subject with a fixed seed, so every feature group and every run uses
  the same voxels.
- The test subjects are stored under their own split folder and are only processed for a final test run.
"""
import concurrent.futures
import hashlib
import json
import os
import typing as t
import zlib

import numpy as np
import SimpleITK as sitk

import mialab.data.structure as structure
import mialab.features as features
import mialab.filtering.feature_extraction as fltr_feat
import mialab.utilities.config as cfg
import mialab.utilities.file_access_utilities as futil
import mialab.utilities.paths as paths
import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]

IMAGE_NAMES = {structure.BrainImageTypes.T1w: 'T1w',
               structure.BrainImageTypes.T2w: 'T2w',
               structure.BrainImageTypes.BrainMask: 'BrainMask',
               structure.BrainImageTypes.GroundTruth: 'GroundTruth'}

# share of the voxels of each label (0 = background, 1-5 = tissues) sampled for training, as in the course code
SAMPLING_LABELS = [0, 1, 2, 3, 4, 5]
SAMPLING_PERCENTAGES = [0.0003, 0.004, 0.003, 0.04, 0.04, 0.02]
SAMPLING_VERSION = 1  # increase when the sampling changes


def short_hash(value) -> str:
    """A short, stable hash of any JSON-serializable value."""
    return hashlib.sha1(json.dumps(value, sort_keys=True).encode()).hexdigest()[:8]


def _atomic_savez(path: str, **arrays):
    """Saves arrays to an .npz file so that an interrupted job never leaves a broken file behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _subject_seed(seed: int, subject_id: str) -> int:
    return (seed * 1_000_003 + zlib.crc32(subject_id.encode())) % 2 ** 32


def _init_worker(atlas_dir: str, threads: int):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    putil.load_atlas_images(atlas_dir)


class FeatureCache:
    """Cached pre-processed images, voxel samples and features of one data split ('train' or 'test')."""

    def __init__(self, config: dict, split: str = 'train', data_dir: str = paths.DATA_DIR,
                 cache_dir: str = paths.CACHE_DIR, workers: int = 1):
        if split not in ('train', 'test'):
            raise ValueError(f'Unknown split "{split}"')
        self.config = config
        self.split = split
        self.data_dir = data_dir
        self.workers = max(1, workers)
        self.prep_key = 'prep_' + short_hash(config['preprocessing'])
        self.root = os.path.join(cache_dir, self.prep_key, split)
        os.makedirs(self.root, exist_ok=True)
        info = os.path.join(cache_dir, self.prep_key, 'preprocessing.json')
        if not os.path.exists(info):
            with open(info, 'w') as f:
                json.dump(config['preprocessing'], f, indent=4)

        crawler = futil.FileSystemDataCrawler(os.path.join(data_dir, split), LOADING_KEYS,
                                              futil.BrainImageFilePathGenerator(), futil.DataDirectoryFilter())
        self._paths = dict(sorted(crawler.data.items()))
        self.subject_ids = list(self._paths)
        if not self.subject_ids:
            raise FileNotFoundError(f'No subjects found in {os.path.join(data_dir, split)}')

    # ----- pre-processed images -----

    def _image_path(self, subject_id: str, name: str) -> str:
        return os.path.join(self.root, subject_id, 'images', name + '.nii.gz')

    def ensure_preprocessed(self, subject_ids: t.List[str] = None):
        """Registers and pre-processes every subject that is not in the cache yet."""
        missing = [s for s in (subject_ids or self.subject_ids)
                   if not os.path.exists(self._image_path(s, 'GroundTruth'))]
        if not missing:
            return
        print('-' * 5, f'Pre-processing {len(missing)} {self.split} subject(s)...', flush=True)
        params = cfg.pre_process_params(self.config)
        tasks = [(s, dict(self._paths[s]), params) for s in missing]
        self._run(_preprocess_task, [(*task, self.root) for task in tasks])

    def load_images(self, subject_id: str, labels: bool = False) -> t.Dict[str, sitk.Image]:
        """Loads the pre-processed images of a subject. The labels are only included if asked for."""
        names = ['T1w', 'T2w', 'BrainMask'] + (['GroundTruth'] if labels else [])
        return {name: sitk.ReadImage(self._image_path(subject_id, name)) for name in names}

    def reference_image(self, subject_id: str) -> sitk.Image:
        """An image with the geometry of the subject in atlas space, used to save segmentations."""
        return sitk.ReadImage(self._image_path(subject_id, 'BrainMask'))

    # ----- brain voxels and training samples -----

    def _voxels_path(self, subject_id: str) -> str:
        return os.path.join(self.root, subject_id, f'voxels_seed{self.config["seed"]}_v{SAMPLING_VERSION}.npz')

    def voxels(self, subject_id: str) -> dict:
        """The brain voxels of a subject.

        Returns:
            dict: ``brain_idx`` (flat index of every brain voxel in the image array), ``labels`` (label of every
                brain voxel) and ``sample_pos`` (positions of the sampled training voxels within ``brain_idx``).
        """
        path = self._voxels_path(subject_id)
        if not os.path.exists(path):
            self.ensure_preprocessed([subject_id])
            mask_image = sitk.ReadImage(self._image_path(subject_id, 'BrainMask'))
            labels_image = sitk.ReadImage(self._image_path(subject_id, 'GroundTruth'))
            brain_idx = np.flatnonzero(sitk.GetArrayFromImage(mask_image).ravel() > 0).astype(np.int32)
            labels = sitk.GetArrayFromImage(labels_image).ravel()[brain_idx].astype(np.uint8)

            np.random.seed(_subject_seed(self.config['seed'], subject_id))
            sample_mask = fltr_feat.RandomizedTrainingMaskGenerator.get_mask(
                labels_image, SAMPLING_LABELS, SAMPLING_PERCENTAGES, background_mask=mask_image)
            sample_idx = np.flatnonzero(sitk.GetArrayFromImage(sample_mask).ravel() > 0)
            sample_pos = np.searchsorted(brain_idx, sample_idx).astype(np.int32)
            _atomic_savez(path, brain_idx=brain_idx, labels=labels, sample_pos=sample_pos)
        with np.load(path) as data:
            return dict(data)

    # ----- features -----

    def _feature_path(self, subject_id: str, group: features.FeatureGroup, fit_ids: t.List[str] = None) -> str:
        name = f'{group.name}_v{group.version}_{short_hash(group.params)}'
        if group.fold_dependent:
            name += '_fit' + short_hash(sorted(fit_ids))
        return os.path.join(self.root, subject_id, 'features', name + '.npz')

    def ensure_features(self, group: features.FeatureGroup, subject_ids: t.List[str],
                        fit_cache: 'FeatureCache' = None, fit_ids: t.List[str] = None):
        """Computes the features of a group for every subject that is not in the cache yet.

        Args:
            group: The feature group.
            subject_ids: The subjects to compute the features for.
            fit_cache: For a fold-dependent group: the cache of the training subjects it is fitted on.
            fit_ids: For a fold-dependent group: the training subjects of the fold. Must never contain a
                subject of ``subject_ids`` that is evaluated in this fold.
        """
        if group.fold_dependent and not fit_ids:
            raise ValueError(f'Feature group "{group.name}" is fold dependent and needs fit_ids')
        missing = [s for s in subject_ids if not os.path.exists(self._feature_path(s, group, fit_ids))]
        if not missing:
            return
        self.ensure_preprocessed(missing)
        for s in missing:
            self.voxels(s)
        if group.fold_dependent:
            fit_cache = fit_cache or self
            group.fit(list(fit_ids), features.FitContext(lambda s: fit_cache.load_images(s, labels=True)))
        print('-' * 5, f'Computing feature group "{group.name}" for {len(missing)} subject(s)...', flush=True)
        tasks = [(self, s, group, self._feature_path(s, group, fit_ids)) for s in missing]
        self._run(_feature_task, tasks)

    def load_features(self, subject_id: str, group: features.FeatureGroup,
                      fit_ids: t.List[str] = None) -> t.Tuple[np.ndarray, t.List[str]]:
        """Returns the features of every brain voxel of a subject (rows) and the feature names (columns)."""
        with np.load(self._feature_path(subject_id, group, fit_ids)) as data:
            return data['values'], [f'{group.name}:{n}' for n in data['names']]

    def matrix(self, subject_ids: t.List[str], groups: t.List[features.FeatureGroup], rows: str,
               fit_ids: t.List[str] = None) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray, t.List[str]]:
        """Builds the feature matrix of several subjects.

        Args:
            subject_ids: The subjects.
            groups: The feature groups (their features must be computed already).
            rows: 'sample' for the sampled training voxels, 'brain' for all brain voxels.
            fit_ids: The training subjects that fold-dependent groups were fitted on.

        Returns:
            tuple: features (voxels x features), labels, subject index of each row, feature names.
        """
        blocks, labels, owner, names = [], [], [], []
        for i, s in enumerate(subject_ids):
            voxels = self.voxels(s)
            take = voxels['sample_pos'] if rows == 'sample' else slice(None)
            columns = []
            names = []
            for group in groups:
                values, group_names = self.load_features(s, group, fit_ids)
                columns.append(values[take])
                names += group_names
            blocks.append(np.concatenate(columns, axis=1))
            labels.append(voxels['labels'][take])
            owner.append(np.full(len(labels[-1]), i, dtype=np.int32))
        return np.concatenate(blocks), np.concatenate(labels), np.concatenate(owner), names

    # ----- parallel execution -----

    def _run(self, function, tasks: list):
        if self.workers == 1 or len(tasks) == 1:
            putil.load_atlas_images(paths.atlas_dir(self.data_dir))
            for task in tasks:
                function(*task)
            return
        threads = max(1, (os.cpu_count() or 1) // self.workers)
        with concurrent.futures.ProcessPoolExecutor(self.workers, initializer=_init_worker,
                                                    initargs=(paths.atlas_dir(self.data_dir), threads)) as pool:
            for future in [pool.submit(function, *task) for task in tasks]:
                future.result()  # re-raises errors of the workers


def _preprocess_task(subject_id: str, subject_paths: dict, params: dict, root: str):
    img = putil.pre_process_images(subject_id, subject_paths, **params)
    folder = os.path.join(root, subject_id, 'images')
    os.makedirs(folder, exist_ok=True)
    for key, name in IMAGE_NAMES.items():
        tmp = os.path.join(folder, f'{name}.tmp{os.getpid()}.nii.gz')
        sitk.WriteImage(img.images[key], tmp, True)
        os.replace(tmp, os.path.join(folder, name + '.nii.gz'))


def _feature_task(cache: FeatureCache, subject_id: str, group: features.FeatureGroup, path: str):
    images = cache.load_images(subject_id, labels=False)
    brain_idx = cache.voxels(subject_id)['brain_idx']
    shape = sitk.GetArrayFromImage(images['T1w']).shape
    result = group.compute(images)
    names, columns = [], []
    for name, value in result.items():
        array = sitk.GetArrayFromImage(value) if isinstance(value, sitk.Image) else np.asarray(value)
        if array.shape != shape:
            raise ValueError(f'Feature "{group.name}:{name}" has shape {array.shape}, expected {shape}')
        names.append(name)
        columns.append(array.ravel()[brain_idx].astype(np.float32))
    _atomic_savez(path, values=np.stack(columns, axis=1), names=np.array(names))
