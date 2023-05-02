# Short overview of this project
## Requirements
All requirements are listed in requirements.txt, after instalation of these requirements some libraries
might not work because of numpy version. New numpy version uses bool instead of np.bool, you must fix this
manually, since I do not recall which libraries were affected by this.

## How to run
You can run this project from file experiment.py. Functions for setup of both detection and filtering experiments 
are present. In combine.py you can try applying trained CGP filters for noised images.

## hal-cgp library disclaimer
folder cgp contains library hal-cgp from github https://github.com/Happy-Algorithms-League/hal-cgp.
This library didn't allow numpy vectorization which I had to implement myself. I also had to implement function
to calculate number of active nodes per function type in CGP individual. Aside from that I also modified bunch of stuff
that I used in different projects, which I might have used in this one as well -> All modifications are marked
with comment # xkaras38


## Files and their purpose
- image_noising.py: This file contains functions for adding hurl noise to images. 
- image_setup_utils.py: This file contatins functions for loading of the image and using those to set
up datasets for both detection and filtering. 
- cgp_utils.py: Contains set of utils for cgp e.g. load, save cgp, pipelines to use cgp for filtering, detection, restoring whole images
etc.
- functions.py: Contains all CGP functions used in this project.
- detection.py: Contains class wrapping all functionality used for training of CGP for detection.
- regression.py: Contains class wrapping all functionality used for training of CGP for filtering.
- experiments.py: In this file is class which is used to initiate and log multiple runs with the same parameters. Furthermore
function to generate group of these classes for whole set of experiments is also present. (e.g. group of experiments
testing both rows and columns count in the CGP)
- parse_experiments.py: This file contains function to parse results of experiments and generate graphs from them.
- combine.py: Functionality to generate images before noising, after noising, and after denoising is in this file.
- readme.md: Describes whole project, also contains sneeky recursion for someone who gets this far.