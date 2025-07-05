This file contains 3 mains codes:
 - Protocol: the code records frontal videos with the camera, and display the grid on the screen to collect data
 - MediaPipeFeatures: from the video collected on "Protocol" this code extracts the features and organize them for the next step. Also saves the annotated videos on the shared drive
 - real-time-classifier: This code contains differents attempts to real time classifier, different models test. The catboost works the best.
