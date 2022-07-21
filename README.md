# 

This repository is the collection of code for the ${\tau\tau}$ inariant mass inference to be used in the HH ${\to}$ bb ${\tau\tau}$ CMS analysis.  
The code is a work in progress.

# Installation

To install the correct environment use [conda](https://docs.conda.io/en/latest/)
```bash
conda env create -f mTT_env.yaml
conda activate mTT_env.yaml
conda env config vars set \
    LD_LIBRARY_PATH=$(realpath $(dirname $CONDA_EXE)/../lib):$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
    MTTFIT_HOME=$PWD
conda deactivate
conda activate mtt-fit
```
Alternatively, the [mTT_env-snapshot.yaml]() dictionary contains all the required packages and version explicitly mentioned.

# Steps

## Create .h5 files

.h5 files are the standard input to the NN taining script. The [create_h5.py]() script creates .h5 files containing the needed columns from the given input. The script assigns the [is_train](), [is_test]() and [is_valid]() labels to be used during training. The precision of the data is reduced to standard of half precision to avoid memory issues.    
To create the .h5 files, run

```bash
python create_h5.py --input input_patter --output output_file --target target_var --tree tree_name --train-size T --valid-size V --min m --max M --threads t --features /path/to/cfg.py
```

where 
```--input``` is the input file path (or glob pattern), 
```--output``` is the output .h5 file name, 
```--target``` is the target variables name, 
```--tree``` is the input ROOT tree name, 
```--train-size``` is the train sample size, 
```--valid-size``` is the validation sample size, 
```--min``` is the lower threshold for the target, 
```--max``` is the higher threshold for the target, 
```--threads``` is the number of threads to use, 
```--features```is the cfg file containing the FEATURES dictionary, 
The test sample size is deduced from the train and test samples size.

# Notes

## Forced dependencies

- ROOT v6.26.02 or greater is needed
- Numpy 1.21.4 or **lower** is needed by Numba

## Changes in the conda environment

When updating the conda enviroment file update also the snapshot dictionaty and be sure not to put any user specific path inside the dictionary. Run something like

```bash
conda activate mtt-fit
conda env export | sed "s_prefix:.*__g" | sed "s_variables:.*__g" | sed "s_LD\_LIBRARY\_PATH.*__g" | sed "s_MTTFIT\_HOME.*__g" > mTT_env-snapshot.yaml
```
to remove lines containing user paths.  
