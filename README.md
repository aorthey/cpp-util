cpp-utils
============

opencv/
------------

  * load_images.cpp: Load and extract info from image
  * face_recognition.cpp: simple opencv hair features to detect a set of faces
  * face_recognition_kalmanian_style.cpp: kalman filter on top of opencv hair feature detector
  * hough2.cpp: houghman transformation
 
cuda/
------------

  * cuda_interface.cu: testbed for cuda_functions.h
  * cuda_functions.cu: methods to conduct matrix multiplication on device (GPU) or host (CPU) and compare them 

core/
------------

  * iostream.cpp: having an inherited std::stream to display more infos like linenumber/file etc
  * vector-inheritance.cpp: inheritance of std::vector to add some functionality

control/
------------

  * car_planning.cpp: some pid control for car like system
  * linear_system.cpp: helper drawing function of linear system
