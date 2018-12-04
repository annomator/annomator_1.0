Annomator

annomator_package_beta_0.1

First open source version of Annomator with trainer for a select few to review
Some of this package is covered under the tensorflow apache 2

The rest I am putting under the apache 2 licence until further notice

Copyright 2018 Annomator
Written by Arend Smits

Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software# distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


I am open sourcing a box and semantic encoder with codecs and trainers under the apache 2 licence.  The script and release are a little unconventional but there are several good reasons to do so.  I am releasing Annomator 0.1 Beta as a working prototype.  I will be moving towards a 1.0 release when it is more functional.  I have written the code, codecs and packaged it up to run on many different computers easily using open source tools on the most amount of computers.  
The real focus has been on shifting from annotating and gathering data to the end user.  The art of wrangling and training will no doubt be helped or handled by someone with dedicated skills but anyone who can use a paint program and a spreadsheet can be up and running quickly.  

The requirements are just Tensorflow, PIL and matplotlib so you can skip the setup if you already have them installed.  Jupyter is only used for instructions but have also included a little fix if you install jupyter without conda.  

All the credit should be given to Google for open sourcing Tensorflow, Slim and the Object Detection API.  I have included a slice to make the prototype run easily but this is not the recommended installation method.  Please see the official instrucions for setup and install.   The requirements are just Tensorflow, PIL and matplotlib so you can skip the setup if you know what you are doing.  

If you move the train.py file out of the legacy folder it will still work with the current version.

I have not put full attributions but the package, the image detection and the semantic enocoder and the codecs are novel and under the apache 2 licence but I would like to apply full attribution.  This version also right of response to give full atribution.
