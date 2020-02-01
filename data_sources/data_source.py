import os
import pickle
import numpy as np


class DataSource(object):
    """
    A base class for creating a data source and manipulating that data source.
    """
    def __init__(self, params):
        self.p = params
        self.data_tags = None
        self.training_dataset = None
        self.validation_dataset = None
        
        self.num_training_samples = self.p.trainer.batch_size * \
                                    int((self.p.trainer.training_set_size * self.p.trainer.num_samples) //
                                     self.p.trainer.batch_size)
        self.num_validation_samples = self.p.trainer.num_samples - self.num_training_samples

    def generate_data(self):
        """
        Generate the data.
        """
        raise NotImplementedError('Should be implemented by the child class')
    
    def load_dataset(self):
        """
        Load a saved dataset.
        """
        # Get all the files in the directory
        file_list = self.get_file_list()

        # Concatenate the data corresponding to a list of files
        data = self.concatenate_file_data(file_list)

        # Shuffle the data and create the training and the validation datasets
        data = self.shuffle_data_dictionary(data)
        self.training_dataset, self.validation_dataset = self.split_data_into_training_and_validation(data)

    def split_data_into_training_and_validation(self, data):
        """
        Split data intro training and validation sets.
        """
        training_dataset = self.get_data_from_indices(data, np.arange(self.num_training_samples))
        validation_dataset = self.get_data_from_indices(data, np.arange(self.num_training_samples,
                                                                        self.p.trainer.num_samples))
        return training_dataset, validation_dataset

    def generate_training_batch(self, start_index):
        """
        Generate a training batch from the dataset.
        """
        assert self.training_dataset is not None
        assert self.data_tags is not None
        return self.get_data_from_indices(self.training_dataset,
                                          np.arange(start_index, start_index + self.p.trainer.batch_size))
    
    def generate_validation_batch(self):
        """
        Generate a validation batch from the dataset.
        """
        assert self.validation_dataset is not None
        assert self.data_tags is not None
        
        # Sample indices and get data
        index_array = np.random.choice(self.num_validation_samples, self.p.trainer.batch_size)
        return self.get_data_from_indices(self.validation_dataset, index_array)
        
    def shuffle_datasets(self):
        """
        Shuffle the training and the validation datasets. This could be helpful in between the epochs for randomization.
        """
        assert self.data_tags is not None
        assert self.training_dataset is not None
        assert self.validation_dataset is not None
        self.training_dataset = self.shuffle_data_dictionary(self.training_dataset)
        self.validation_dataset = self.shuffle_data_dictionary(self.validation_dataset)
    
    def shuffle_data_dictionary(self, data_dictionary):
        """
        Shuffle a dictionary of the data.
        """
        num_samples = np.shape(data_dictionary[self.data_tags[0]])[0]
        shuffle_order = np.random.permutation(num_samples)
        for data_tag in self.data_tags:
            data_dictionary[data_tag] = data_dictionary[data_tag][shuffle_order]
        return data_dictionary

    def get_file_list(self, file_type='.pkl'):
        """
        Get a sorted list of all the files in the data directory.
        """
        # Note (Somil): Since we moved from a string to a list convention for data directories, we are adding
        # additional code here to make sure it is backwards compatible.
        if isinstance(self.p.data_creation.data_dir, str):
            self.p.data_creation.data_dir = [self.p.data_creation.data_dir]
        
        file_list = []
        for i in range(len(self.p.data_creation.data_dir)):
            file_list.extend([os.path.join(self.p.data_creation.data_dir[i], f)
                              for f in os.listdir(self.p.data_creation.data_dir[i]) if f.endswith(file_type)])
        return file_list

    def concatenate_file_data(self, file_list):
        """
        Concatenate the data from different files in the file list. This function assumes that each file in file_list
        stores a dictionary and each element of that dictionary is an array with the zeroth dimension being the batch
        dimension.
        """
        # Get all the tags in the data
        if self.data_tags is None:
            self.get_data_tags(file_list[0])
            
        # Create all the keys
        data = {}
        for tag in self.data_tags:
            data[tag] = []
        
        # Load the data
        for filename in file_list:
            data_current = self._get_current_data(filename)
            for tag in self.data_tags:
                data[tag].append(data_current[tag])
        
        # Concatenate all the data
        for tag in self.data_tags:
            data[tag] = np.vstack(data[tag])
        
        return data

    def _get_current_data(self, filename):
        """
        Load and return the data stored in filename.
        This can be overriden in subclasses
        (see image_data_source.py).
        """
        with open(filename, 'rb') as handle:
            data_current = pickle.load(handle)
        return data_current

    def get_data_tags(self, example_file, file_type='.pkl'):
        """
        Get the keys of the dictionary saved in the example file.
        """
        if file_type == '.pkl':
            with open(example_file, 'rb') as handle:
                data = pickle.load(handle)
            self.data_tags = list(data.keys())
        else:
            raise NotImplementedError

    def get_data_from_indices(self, data_dictionary, indices):
        """
        Get the data corresponding to a given indices.
        """
        data = {}
        for tag in self.data_tags:
            try:
                data[tag] = data_dictionary[tag][indices]
            except KeyError:
                print("no this key in the current data file!")
        return data
