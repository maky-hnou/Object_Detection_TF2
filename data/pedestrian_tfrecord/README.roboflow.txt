
pedestrian dataset
==============================

This dataset was exported on August 22, 2020 at 2:12 PM GMT

It includes 476 images.
Pedestrian are annotated in Tensorflow TFRecord (raccoon) format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random brigthness adjustment of between -25 and +25 percent


