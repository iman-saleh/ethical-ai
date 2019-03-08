This repository contains artifacts that support work on Ethical AI

License Plate Blurring Script
=============================

The blur-license-plate-tutorial.ipynb demonstrates capabilities to detect and blur license plates in images.

The scripts comes with sample images for testing. The images are loaded from 'license-plate' directory and the processed (blurred) images are saved under 'license-plate-processed' directory.

The script is tested on Ubuntu 16.04 and Python 2.7

Contact
--------

Iman Saleh
iman.saleh@intel.com

License
---------

Affero GPLv3 http://www.gnu.org/licenses/agpl-3.0.html

Face Blurring Script
=============================

The blur-face-tutorial.ipynb demonstrates capabilities to detect and blur faces in images.

The scripts comes with sample images for testing. The images are loaded from 'pedestrian; directory and the processed (blurred) images are saved under 'pedestrian-processed' directory.

The script is tested on Ubuntu 16.04 and Python 2.7 and requires Tensorflow, Numpy, OpenCV and Pillow.

Contact
--------

Cory Ilo
cory.i.ilo@intel.com

License
---------

Affero GPLv3 http://www.gnu.org/licenses/agpl-3.0.html

Bias Utility Script
=============================

The Bias_Detection.ipynb demonstrates capabilities to measure bias in the model.

The script downloads images and models hosted in [AWS S3] (https://s3-us-west-1.amazonaws.com/strata-bias-data2/inputdata.tar.gz). The script runs detection on images and models in inputdata directory.  The script then calculates mean difference and unexplainable mean difference between pedestrian detection accuracy at day time vs night time images. The detected images are stored in outputdata directory.

The script is tested on Centos 7.6 and Python 3.6 and requires Tensorflow, Numpy, OpenCV and Pillow.

Contact
--------

Cindy Tseng
cindy.s.tseng@intel.com

License
---------

Affero GPLv3 http://www.gnu.org/licenses/agpl-3.0.html
