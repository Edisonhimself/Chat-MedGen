import os
import logging
import warnings

from chatmedgen.common.registry import registry
from chatmedgen.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from chatmedgen.datasets.datasets.text_caps import TextCapDataset

from chatmedgen.datasets.datasets.unnatural_instruction import UnnaturalDataset
from chatmedgen.datasets.datasets.multitask_conversation import MultiTaskConversationDataset



@registry.register_builder("multitask_conversation")
class MultitaskConversationBuilder(BaseDatasetBuilder):
    train_dataset_cls = MultiTaskConversationDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/multitask_conversation/default.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets


@registry.register_builder("unnatural_instruction")
class UnnaturalInstructionBuilder(BaseDatasetBuilder):
    train_dataset_cls = UnnaturalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/nlp/unnatural_instruction.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()
        build_info = self.config.build_info
        datasets = dict()

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_path=build_info.ann_path,
        )

        return datasets









class AllRefCOCOBuilder(BaseDatasetBuilder):

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        image_path = build_info.image_path
        ann_path = build_info.ann_path

        datasets = dict()

        if not os.path.exists(image_path):
            warnings.warn("image path {} does not exist.".format(image_path))
        if not os.path.exists(ann_path):
            warnings.warn("ann path {} does not exist.".format(ann_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=ann_path,
            vis_root=image_path,
            dataset=build_info.dataset,
            splitBy=build_info.splitBy
        )

        return datasets
    

        return datasets


@registry.register_builder("textcaps_caption")
class TextcapCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = TextCapDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/textcaps/caption.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            ann_path=build_info.ann_path,
            vis_root=build_info.image_path,
        )

        return datasets
    
