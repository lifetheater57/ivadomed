import os
import json
import shutil
import sys
import joblib

import torch
import torch.backends.cudnn as cudnn

from ivadomed import training as imed_training
from ivadomed import evaluation as imed_evaluation
from ivadomed import testing as imed_testing
from ivadomed import metrics as imed_metrics
from ivadomed import transforms as imed_transforms
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader, film as imed_film

cudnn.benchmark = True

# List of not-default available models i.e. different from Unet
MODEL_LIST = ['UNet3D', 'HeMISUnet', 'FiLMedUnet']

def define_device(gpu_id):
    """Define the device used for the process of interest.

    Args:
        gpu_id (int): ID of the GPU
    Returns:
        Bool, device: True if cuda is available
    """
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        print("Cuda is not available.")
        print("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        gpu_number = int(gpu_id)
        torch.cuda.set_device(gpu_number)
        print("Using GPU number {}".format(gpu_number))
    return cuda_available, device


def get_new_subject_split(path_folder, center_test, split_method, random_seed,
                          train_frac, test_frac, log_directory):
    """Randomly split dataset between training / validation / testing.

    Randomly split dataset between training / validation / testing
        and save it in log_directory + "/split_datasets.joblib"
    Args:
        path_folder (string): Dataset folder
        center_test (list): list of centers to include in the testing set
        split_method (string): see imed_loader_utils.split_dataset
        random_seed (int):
        train_frac (float): between 0 and 1
        test_frac (float): between 0 and 1
        log_directory (string): output folder
    Returns:
        list, list list: Training, validation and testing subjects lists
    """
    train_lst, valid_lst, test_lst = imed_loader_utils.split_dataset(path_folder=path_folder,
                                                                     center_test_lst=center_test,
                                                                     split_method=split_method,
                                                                     random_seed=random_seed,
                                                                     train_frac=train_frac,
                                                                     test_frac=test_frac)

    # save the subject distribution
    split_dct = {'train': train_lst, 'valid': valid_lst, 'test': test_lst}
    joblib.dump(split_dct, "./" + log_directory + "/split_datasets.joblib")

    return train_lst, valid_lst, test_lst


def get_subdatasets_subjects_list(split_params, bids_path, log_directory):
    """Get lists of subjects for each sub-dataset between training / validation / testing.

    Args:
        split_params (dict):
        bids_path (string): Path to the BIDS dataset
        log_directory (string): output folder
    Returns:
        list, list list: Training, validation and testing subjects lists
    """
    if split_params["fname_split"]:
        # Load subjects lists
        old_split = joblib.load(split_params["fname_split"])
        train_lst, valid_lst, test_lst = old_split['train'], old_split['valid'], old_split['test']
    else:
        train_lst, valid_lst, test_lst = get_new_subject_split(path_folder=bids_path,
                                                               center_test=split_params['center_test'],
                                                               split_method=split_params['method'],
                                                               random_seed=split_params['random_seed'],
                                                               train_frac=split_params['train_fraction'],
                                                               test_frac=split_params['test_fraction'],
                                                               log_directory=log_directory)
    return train_lst, valid_lst, test_lst


def get_film_metadata_models(ds_train, metadata_type, debugging=False):
    if metadata_type == "mri_params":
        metadata_vector = ["RepetitionTime", "EchoTime", "FlipAngle"]
        metadata_clustering_models = imed_film.clustering_fit(ds_train.metadata, metadata_vector)
    else:
        metadata_clustering_models = None

    ds_train, train_onehotencoder = imed_film.normalize_metadata(ds_train,
                                                                 metadata_clustering_models,
                                                                 debugging,
                                                                 metadata_type,
                                                                 True)

    return ds_train, train_onehotencoder, metadata_clustering_models


def display_selected_model_spec(params):
    """Display in terminal the selected model and its parameters.

    Args:
        params (dict): keys are param names and values are param values
    Returns:
        None
    """
    print('\nSelected architecture: {}, with the following parameters:'.format(params["name"]))
    for k in list(params.keys()):
        if k != "name":
            print('\t{}: {}'.format(k, params[k]))


def display_selected_transfoms(params, dataset_type):
    """Display in terminal the selected transforms for a given dataset.

    Args:
        params (dict):
        dataset_list (list): e.g. ['testing'] or ['training', 'validation']
    Returns:
        None
    """
    print('\nSelected transformations for the {} dataset:'.format(dataset_type))
    for k in list(params.keys()):
        print('\t{}: {}'.format(k, params[k]))


def get_subdatasets_transforms(transform_params):
    """Get transformation parameters for each subdataset: training, validation and testing.

    Args:
        transform_params (dict):
    Returns:
        dict, dict, dict
    """
    train, valid, test = {}, {}, {}
    subdataset_default = ["training", "validation", "testing"]
    # Loop across transformations
    for transform_name in transform_params:
        subdataset_list = ["training", "validation", "testing"]
        # Only consider subdatasets listed in dataset_type
        if "dataset_type" in transform_params[transform_name]:
            subdataset_list = transform_params[transform_name]["dataset_type"]
        # Add current transformation to the relevant subdataset transformation dictionaries
        for subds_name, subds_dict in zip(subdataset_default, [train, valid, test]):
            if subds_name in subdataset_list:
                subds_dict[transform_name] = transform_params[transform_name]
                if "dataset_type" in subds_dict[transform_name]:
                    del subds_dict[transform_name]["dataset_type"]
    return train, valid, test


def run_main():
    if len(sys.argv) <= 1:
        print("\nivadomed [config.json]\n")
        return

    with open(sys.argv[1], "r") as fhandle:
        context = json.load(fhandle)

    command = context["command"]
    log_directory = context["log_directory"]
    if not os.path.isdir(log_directory):
        print('Creating log directory: {}'.format(log_directory))
        os.makedirs(log_directory)
    else:
        print('Log directory already exists: {}'.format(log_directory))

    # Define device
    cuda_available, device = define_device(context['gpu'])

    # Get subject lists
    train_lst, valid_lst, test_lst = get_subdatasets_subjects_list(context["split_dataset"],
                                                                   context['bids_path'],
                                                                   log_directory)

    # Get transforms for each subdataset
    transform_train_params, transform_valid_params, transform_test_params = \
        get_subdatasets_transforms(context["transformation"])
    if command == "train":
        display_selected_transfoms(transform_train_params, dataset_type="training")
        display_selected_transfoms(transform_valid_params, dataset_type="validation")
    elif command == "test":
        display_selected_transfoms(transform_test_params, dataset_type="testing")

    # Loader params
    loader_params = context["loader_parameters"]
    extra_params = {"bids_path": context['bids_path'],
                    "target_suffix": context["target_suffix"],
                    "metadata_type": False
                    }
    if "FiLMedUnet" in context and context["FiLMedUnet"]["applied"]:
        extra_params.update({"metadata_type": context["FiLMedUnet"]["metadata"]})
    loader_params.update(extra_params)

    # METRICS
    metric_fns = [imed_metrics.dice_score,
                  imed_metrics.multi_class_dice_score,
                  imed_metrics.hausdorff_3D_score,
                  imed_metrics.precision_score,
                  imed_metrics.recall_score,
                  imed_metrics.specificity_score,
                  imed_metrics.intersection_over_union,
                  imed_metrics.accuracy_score]

    # MODEL PARAMETERS
    model_params = context["default_model"]
    model_context_list = [model_name for model_name in MODEL_LIST
                          if model_name in context and context[model_name]["applied"]]
    if len(model_context_list) == 1:
        model_params["name"] = model_context_list[0]
        model_params.update(context[model_context_list[0]])
    elif len(model_context_list) > 1:
        print('ERROR: Several models are selected in the configuration file: {}.'
              'Please select only one (i.e. only one where: "applied": true).'.format(model_context_list))
        exit()
    else:
        # Default model i.e. Unet
        pass
    # If multi-class output, then add background class
    if model_params["out_channel"] > 1:
        model_params.update({"out_channel": model_params["out_channel"] + 1})
    # Display for spec' check
    display_selected_model_spec(params=model_params)
    # Update loader params
    loader_params.update({"model_params": model_params})

    if command == 'train':
        # LOAD DATASET
        # Get Training dataset
        ds_train = imed_loader.load_dataset(**{**loader_params,
                                               **{'data_list': train_lst[:3], 'transforms_params': transform_train_params,
                                                  'dataset_type': 'training'}})
        # Get Validation dataset
        ds_valid = imed_loader.load_dataset(**{**loader_params,
                                               **{'data_list': valid_lst[:3], 'transforms_params': transform_valid_params,
                                                  'dataset_type': 'validation'}})
        # If FiLM, normalize data
        if model_params["name"] == "FiLMedUnet":
            # Normalize metadata before sending to the FiLM network
            results = get_film_metadata_models(ds_train=ds_train, metadata_type=model_params['metadata'],
                                               debugging=context["debugging"])
            ds_train, train_onehotencoder, metadata_clustering_models = results
            ds_valid = imed_film.normalize_metadata(ds_valid, metadata_clustering_models, context["debugging"],
                                                    model_params['metadata'])
            model_params.update({"film_onehotencoder": train_onehotencoder,
                                 "n_metadata": len([ll for l in train_onehotencoder.categories_ for ll in l])})
            joblib.dump(metadata_clustering_models, "./" + log_directory + "/clustering_models.joblib")
            joblib.dump(train_onehotencoder, "./" + log_directory + "/one_hot_encoder.joblib")

        # RUN TRAINING
        imed_training.train(model_params=model_params,
                            dataset_train=ds_train,
                            dataset_val=ds_valid,
                            training_params=context["training_parameters"],
                            log_directory=log_directory,
                            cuda_available=cuda_available,
                            metric_fns=metric_fns,
                            debugging=context["debugging"])

        # Save config file within log_directory
        shutil.copyfile(sys.argv[1], "./" + log_directory + "/config_file.json")

    elif command == 'test':
        # LOAD DATASET
        # Aleatoric uncertainty
        if context['uncertainty']['aleatoric'] and context['uncertainty']['n_it'] > 0:
            transformation_dict = transform_test_params
        else:
            transformation_dict = transform_valid_params
        # Get Testing dataset
        ds_test = imed_loader.load_dataset(**{**loader_params, **{'data_list': test_lst,
                                                                  'transforms_params': transformation_dict,
                                                                  'dataset_type': 'testing',
                                                                  'requires_undo': True}})

        # UNDO TRANSFORMS
        undo_transforms = imed_transforms.UndoCompose(transformation_dict)

        if model_params["name"] == "FiLMedUnet":
            metadata_clustering_models = joblib.load("./" + log_directory + "/clustering_models.joblib")
            one_hot_encoder = joblib.load("./" + log_directory + "/one_hot_encoder.joblib")
            ds_test = imed_film.normalize_metadata(ds_test, metadata_clustering_models, context["debugging"],
                                                   model_params['metadata'])
            model_params.update({"film_onehotencoder": one_hot_encoder,
                                 "n_metadata": len([ll for l in one_hot_encoder.categories_ for ll in l])})

        # RUN INFERENCE
        testing_params = context["testing_parameters"]
        testing_params.update(context["training_parameters"])
        testing_params.update({context["target_suffix"]})
        testing_params.update({'undo_transforms': undo_transforms, 'slice_axis': loader_params['slice_axis']})
        imed_testing.test(model_params=model_params,
                          dataset_test=ds_test,
                          testing_params=testing_params,
                          log_directory=log_directory,
                          device=device,
                          cuda_available=cuda_available,
                          metric_fns=metric_fns,
                          debugging=context["debugging"])

    elif command == 'eval':
        # PREDICTION FOLDER
        path_preds = os.path.join(log_directory, 'pred_masks')
        # If the prediction folder does not exist, run Inference first
        if not os.path.isdir(path_preds):
            print('\nRun Inference\n')
            context["command"] = "test"
            run_main(context)

        # RUN EVALUATION
        imed_evaluation.evaluate(bids_path=context['bids_path'],
                                 log_directory=log_directory,
                                 path_preds=path_preds,
                                 target_suffix=context["target_suffix"],
                                 eval_params=context["evaluation_parameters"])

if __name__ == "__main__":
    run_main()
