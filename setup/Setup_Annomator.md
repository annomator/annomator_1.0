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

It is recommended to use Tensorflow 1.5.0.  Works with 1.5.0 - 1.10 (Not 1.11)  
Works with TF Object Detection 1.0 BEFORE September 2018.  You simply need to move the train.py out of the legacy folder with later releases.

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

Download/Clone annomator from github  
https://github.com/annomator  


### Install complete for most end-users  

Annotating is setup for MSCOCO - see Demo  

I recommend using the following open source programs:  
Gimp to annotate masks  
xnview for batch processing images  
VLC to turn video into images  
Open Office for data analysis  

