# Setup for Annomator


### Quick Install - Terminal
Download annomator from github https://github.com/annomator
Use the setup folder and enter the following in a terminal
pip install -r annotator_requirements.txt
or for GPU 
pip install -r annotator_requirements_gpu.txt


### Install
It is recommended to use Python 3.6.6
Works with Python 2.7 and 3.3-3.6 (Not 3.7)

It is recommended to use virtualenv
Works as system/native, conda, virtualenv or pipenv

It is recommended to use Tensorflow 1.5.0 and TF Object Detection 1.0 BEFORE September 2018
Works with 1.5.0 - 1.10 (Not 1.11)

Download Python 3.6 and update to pip 18
Create/activate the virtual environment
pip install jupyter==6.4 # latest (7.0.1 bug is still active) 
pip install pillow==5.0.0 # some OS issues still exist with later versions 
pip install matplotlib # latest and only - I will lock down a version if needed 
pip install tensorflow==1.5.0 # or pip install tensorflow-gpu==1.5.0

I have later added this to ensure both 1.5.0 and 1.10.0 will work for most
pip install protobuf==3.6.1 (simply to ensure future version control)

annomator_requirements.txt and annomator_requirements_gpu.txt are the result

Windows - You may get an error msvcp140.dll
Download the Visual C++ 2015 Redistributable Update 3

Download annomator from github
https://github.com/annomator

### Install complete for most end-users

Annotating is setup for MSCOCO - see Demo
I recommend using the following open source programs
Gimp to annotate masks xnview for batching images VLC to turn video into images

### For Annomator Pro

If you want to train boxes or train masks you may need to protoc and PYTHONPATH
This is also needed for pbtxt, and therefore other models

I have included a coco category_index for the annotator
You can do the same for other models or your own data

Protoc
If Windows and protobuf 3.6.1 you are already done installing as pre-proprocessed - skip to PYTHONPATH

The linux and mac install is much simpler as since 3.5+ else forced to script or do individually

Protoc Download
https://github.com/protocolbuffers/protobuf/releases/tag/v3.6.1

PYTHONPATH
This can be setup permanently using bash file or windows environment. You can just run the following in each terminal session:

Linux/Mac - cd to the 'annomator' folder export PYTHONPATH=$PYTHONPATH:pwd:pwd/tf_slim_obj_det Windows - no need to cd. Just change absolute path to your tf_slim_obj_det folder set PYTHONPATH=%PYTHONPATH%;C:\path_to\tf_slim_obj_det