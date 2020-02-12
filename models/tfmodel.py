import os
import tensorflow as tf
import numpy as np
from collections import OrderedDict


class TFModel(object):
    """
    Abstract class to represent framework components. Provides common functionality to keep
    performance statistics, help with model loading/saving/migration, access and count parameters, 
    hyper-parameters, etc. For most use-cases, see specific sub-classes: e.g, NIPModel for camera 
    ISPs, or DCN for learned compression.

    # Working with hyper-parameters
    The framework provides the 'ParamSpec' class to help with hyper-parameter definitions, validation
    and storage. See documentation of that class for details, and existing TFModel sub-classes for 
    more examples.

    # Accessing model parameters
    - parameters - list of all trainable parameters in the model (useful for loading/saving/counting parameters)
    - variables  - list of all variables in the model (useful for initialization)

    # Usage of model strings in the framework:
    - summary                 - a human-readable summary of the model (name + rudimentary layer specs + parameter count)
    - model_code              - represents a concise, coded summary of the models hyper parameters
    - class_name              - convenience method to access class name
    - scoped_name             - class name (lower case) [+ postfix label] (e.g., unet / unet_a / fan)
                                used as a directory name for storing models
    """

    def __init__(self, label, **kwargs):  
        self._label = '_'+label if label is not None else ''
        self._model = None
        self.reset_performance_stats()        

    def reset_performance_stats(self):
        self.performance = {
            'loss': {'training': [], 'validation': []},
        }

    @property
    def parameters(self):
        return self._model.trainable_weights
    
    @property
    def variables(self):
       return self._model.variables
        
    def count_parameters(self):
        return np.sum([np.prod(tv.shape.as_list()) for tv in self.parameters])
    
    def count_parameters_breakdown(self):
        return OrderedDict([(tv.name, np.prod(tv.shape.as_list())) for tv in self.parameters])

    def save_model(self, dirname, epoch=0):
        if not dirname.endswith(self.scoped_name):
            dirname = os.path.join(dirname, self.scoped_name)

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        self._model.save_weights(os.path.join(dirname, self.class_name.lower()))

    def load_model(self, dirname):
        if not dirname.endswith(self.scoped_name):
            dirname = os.path.join(dirname, self.scoped_name)
        print('<', os.path.join(dirname, self.class_name.lower()))
        self._model.load_weights(os.path.join(dirname, self.class_name.lower()))
        self.reset_performance_stats()

    def migrate_model(self, dirname, mapping=None, verbose=False):
        """
        Migrate a pre-trained model from a TF checkpoint. Popular reasons include
        changed TF version or changed variable names. The function loads specific variables
        from the checkpoint and uses their values for new weights. The mapping is defined
        in the 'mapping' dictionary. The new model can later be saved using 'save_model'.

        Hint: It may be useful to use tf.keras.backend.clear_session() to make sure variable 
        names are not changing during the migration.

        :param dirname: directory with a saved TF checkpoint
        :param mapping: dict {'new name' : 'old name'}
        :param verbose: self explanatory
        """
        if not dirname.endswith(self.scoped_name):
            dirname = os.path.join(dirname, self.scoped_name)

        if verbose:
            print('# All variables found in the checkpoint')
            for i, (var_name, _) in enumerate(tf.train.list_variables(dirname)):
                var = tf.train.load_variable(dirname, var_name)
                print('{0:3d}.  {1:30s} -> {2.shape}'.format(i, var_name, var))

        if mapping is not None:
            for var in self._model.trainable_variables:
                var_name = var.name.replace(':0', '')
                if var_name not in mapping:
                    print('warning: mapping for {} = {} not found'.format(var.name, var_name))
                    continue
                var_value = tf.train.load_variable(dirname, mapping[var_name])
                print('{} = {} {} <- {} {}'.format(var.name, var_name, var.shape, mapping[var_name], var_value.shape))
                var.assign(var_value)
        
        self.reset_performance_stats()

    @property
    def class_name(self):
        return type(self).__name__

    def summary(self):
        return '{} model [{:,} parameters]'.format(self.class_name, self.count_parameters())

    @property
    def model_code(self):
        raise NotImplementedError()

    @property
    def scoped_name(self):
        return '{}{}'.format(type(self).__name__.lower(), self._label)

    def get_hyperparameters(self):
        raise NotImplementedError()

    def __repr__(self):
        extra_params = ','.join('{}={}'.format(k, '"{}"'.format(v) if isinstance(v, str) else v) for k, v in self._h.changed_params().items())
        return '{}({})'.format(self.class_name, extra_params)

    def _has_attributes(self, attrs, message='Expected attributes not found: {}'):
        setup_status = {key: hasattr(self, key) for key in attrs}
        if not all(setup_status.values()):
            raise NotImplementedError(message.format([key for key, value in setup_status.items() if not value]))
