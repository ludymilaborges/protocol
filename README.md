This file contains 3 principal codes:
 - Protocol: The code records frontal videos with the camera and displays the grid on the screen to collect data
 - MediaPipeFeatures: extracted from the video collected using the "Protocol". This code extracts the features and organises them for the next step. Also saves the annotated videos on the shared drive
 - real-time-classifier: This code contains various attempts at a real-time classifier and model tests. The catboost works the best.
And the data "label_all_data": where participants 1 to 10 are adults, and 11 to 15 are the children's data. Some kids had more sessions.
