# SnowfinchWire.BroodAnalyzer

Estimation of size and age of Eurasian Snowfinch broods based on audio recordings

## Classification

In order to run predictions, use `classify.py` script placed in the project root directory. The script accepts a
few parameters, which can be supplied by specifying a path to a configuration file via `-c` command line option.

If the configuration file path is given by `-c` option, there is no need for other command line arguments, as long
as the supplied configuration provides all required parameters. The schema of configuration file is described in the
following subsection.

If `-c` option is ommitted, the following command line options are required:

* `-i <input_path>` - a path to input data; it can be an audio file, a directory with audio files or a directory with
  feeding statistics (in case of a simple brood size classifier),
* `-m <model_path>` - a path to a serialized trained model or a directory containing such model files; if directory
  with multiple model files is specified, the program will perform classification using each of the models and store
  results in corresponding subdirectories.

There are also several optional parameters:

* `-l <label_path>` - a path to a directory with label files, structured in the same way as the recordings in
  `input_path`; this option is not required, if not specified, the program will assume that label files are
  placed next to the corresponding recordings,
* `-o <output_directory>` - a path to a directory where prediction results are to be stored, `_out` by default,
* `-w <n_workers>` - number of parallel processes to use for classification, 10 by default,
* `-p <period>` - a period (in hours) of results aggregation, 48 by default,
* `--overlap-hours <overlap>` - an overlap for aggregation period given in hours, 0 by default,
* `--mt-threshold <threshold>` - a threshold used in multi target classification to determine whether a prediction score
  indicates belonging to a class or not; used only when a multi target capable model is supplied.

### Configuration file

The program configuration can be specified in a form of a YAML file with the following schema:

```yaml
data:
  recordings: <path>
  labels: <path>
model:
  path: <path>
  n_workers: <integer>
output_dir: <path>
aggregation:
  period_hours: <integer>
  overlap_hours: <integer>
feeding_detector:
  type: <type>
  path: <path>
  args:
    model_path: <path>
    extension: <extension>
  ...
```

The following attributes are required:

* `data.recordings` - same as `-i` command line option,
* `model.path` - same as `-m` command line option.

All the other attributes are optional and each of them corresponds to one of the command line options described above.

### Simple size classfier

The input directory for a simple brood size classifier should contain the following files:
* _feeding-stats.csv_ - a file with feeding statistics produced by _SnowfinchWire.BeggingCallsAnalyzer_,
* _snowfinch-broods.csv_ - a file with true brood information; the following columns are required:
  * _brood_id_ - brood identifier, string
  * _datetime_ - date and time in ISO format
  * _age_min_ - age of the youngest nestling, floating point number
  * _age_max_ - age of the eldest nestling, floating point number
**TODO: implement!**

### Embedded feeding detection

The program is capable of generating feeding labels for recordings by calling _SnowfinchWire.BeggingCallsAnalyzer_
interface. If the user wishes to do this, they can use the following options:

* `--feeding-detector-model <path>` - a path to a serialized trained feeding detector model,
* `--feeding-detector-path <path>` - a path to standalone _SnowfinchWire.BeggingCallsAnalyzer_ installation, if not
  specified embedded installation will be used,
* `--feeding-detector-type <type>` - a type of feeding detector to use; available options are `fe` (_feeding
  extraction_) and `oss` (_OpenSoundscape_), `fe` by default.

Alternatively, feeding detection can be configured via the configuration file described above, using the
`feeding_detector` object. The `type` attribute is equivalent to `--feeding-detector-type` command line option and
the `path` attribute is equivalent to `--feeding-detector-path` option. All attributes defined under `args` key are
transparently passed to the _SnowfinchWire.BeggingCallsAnalyzer_ launcher.

## Model training

In order to train a model, use `train_model.py` script located under the project root directory. The script requires
several command line arguments and options, which partly depend on the model type. The common argument is
`data_path` - a path to a directory containing the dataset definitions. There are also a few common options:

* `-c <path>` - a path to dataset split configuration file; all such files used for experiments conducted so far are
  located under `config` directory; this is a required option,
* `-a <architecture>` - a model architecture to use, can be one of the following:
    * _simple-ensemble_ to train simple brood size classifier,
    * One of CNN architectures supported by OpenSoundscape: 
      * _resnet18_, 
      * _resnet50_, 
      * _resnet101_, 
      * _resnet152_,
      * _vgg11_bn_, 
      * _densenet121_, 
      * _inception_v3_, 
      * _matchboxnet_.

  Set to _resnet18_ by default,
* `--out <output_directory>` - a path to a directory where prediction results are to be stored, `_out` by default.

### Simple size classifier

If the user chooses to train the simple size classifier, they can specify the following additional options:
* `--n-simple-models <count>` - a count of simple models of each type used in a target ensemble classifier, 20 by 
  default,
* `--ensemble-voting <type>` - a type of voting used by an ensemble model to determine final predictions; can be set 
  to _soft_ (default) or _hard_.

The dataset definition directory has to contain the following files:
* _feeding-stats.csv_ - a file with feeding statistics produced by _SnowfinchWire.BeggingCallsAnalyzer_,
* _snowfinch-broods.csv_ - a file with true brood information; the following columns are required:
  * _brood_id_ - brood identifier, string
  * _datetime_ - date and time in ISO format
  * _age_min_ - age of the youngest nestling, floating point number
  * _age_max_ - age of the eldest nestling, floating point number
  * _size_ - true brood size, integer

### OpenSoundscape

Training a MatchboxNet model or one of OpenSoundscape CNNs requires additional command line argument, `audio_path`. 
It has to specify a path to a directory containg audio recordings. Moreover, there are a few more options:
* `-d <duration>` - an audio sample duration in seconds for the model to work on, 2 by default,
* `-n <n_epochs>` - a number of training epochs, 10 by default,
* `-w <n_workers>` - a number of parallel processes used for training,
* `-b <batch_size>` - a batch size, 100 by default,
* `-l <learning_rate>` - a learning rate, 0.001 by default,
* `-t <target>` - classification target: _size_, _age_ or _all_ (default); if _all_ is chosen, two models are generated,
* `--age-mode <mode>` - classification mode for brood age: _single_ (default) or _multi_; use this option to train 
  multi target model,
* `--samples-per-class <n>` - specifies how many audio samples should be used per each class; by default the 
  quantity of samples representing the least numerous class is used.

The dataset definition directory has to contain the following files:
* _brood-size.csv_
* _brood-age.csv_
