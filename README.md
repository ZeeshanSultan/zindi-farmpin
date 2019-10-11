# Zindi Farm Pin

Solution to the crop classification challenge on Zindi

# Installation guide

## Set up conda environment

Using conda:

```
conda env create -f environment.yml
activate zindi_farm_pin
```

The packages necessary to run the project are now installed inside the conda environment.

**Note: The following sections assume you are located in your conda environment.**

## Set up project's module

To move beyond notebook prototyping, all reusable code should go into the `src/` folder package. To use that package inside your project, install the project's module in editable mode, so you can edit files in the `src/` folder and use the modules inside your notebooks :

```
pip install --editable .
```


# Usage

## Download Competition Data

Go to the [competition page](https://zindi.africa/competitions/farm-pin-crop-detection-challenge) and download all the data _except the .zip files_ (see the invoke command for that). 

## Invoke Commands

**Download satellite data**
```
invoke download-satellite data
```
Downloads all the satellite data and stores in `data/raw/*`

**Re-order dataset**
```
invoke reorder-dataset
```
The downloaded data is in .SAFE format, which is confusing. This command takes all the image data over multiple dumps and places them in one folder. For example `data/interim/images/2017-01-01`

**Create masks dataset**
```
invoke create-masks-dataset
```

**Create stacked masks dataset**
```
invoke create-stacked-masks-dataset
```

**Extract baseline features**
```
invoke create-baseline-dataset
```

Runs a script to create the basic baseline dataset containing descriptive statistics of all farms

**TODO**

* Command to extract farm masks for all bands
* Add new zip files to download script
