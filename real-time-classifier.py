# #%%
# # Run 1st: Import the necessary modules.
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2
# import numpy as np
# import cv2
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd

# #%%
# # Run 2nd: Function to draw the face landmarks on the image.
# def draw_landmarks_on_image(rgb_image, detection_result):
#     face_landmarks_list = detection_result.face_landmarks
#     annotated_image = np.copy(rgb_image)

#     for idx in range(len(face_landmarks_list)):
#         face_landmarks = face_landmarks_list[idx]
#         face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#         face_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
#         ])

#         solutions.drawing_utils.draw_landmarks(
#             image=annotated_image,
#             landmark_list=face_landmarks_proto,
#             connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
#         solutions.drawing_utils.draw_landmarks(
#             image=annotated_image,
#             landmark_list=face_landmarks_proto,
#             connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
#         solutions.drawing_utils.draw_landmarks(
#             image=annotated_image,
#             landmark_list=face_landmarks_proto,
#             connections=mp.solutions.face_mesh.FACEMESH_IRISES,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

#     return annotated_image

# #%%
# # Run 3rd: Load the pre-trained model.
# data = pd.read_excel('label_cut_data.xlsx')
# X = data.iloc[:, 0:14]  # Features
# y = data.iloc[:, 15]    # Labels
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X, y)

# # Create a FaceLandmarker object.
# base_options = python.BaseOptions(model_asset_path='face_landmarker2.task')
# options = vision.FaceLandmarkerOptions(base_options=base_options,
#                                        output_face_blendshapes=True,
#                                        output_facial_transformation_matrixes=True,
#                                        num_faces=1)
# detector = vision.FaceLandmarker.create_from_options(options)

# #%%
# # Run 4th: Real-time camera capture and prediction.
# cap = cv2.VideoCapture(0)  # 0 for default camera
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the frame to RGB format
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_rgb_uint8 = np.array(frame_rgb, dtype=np.uint8)
#     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb_uint8)

#     # Perform face detection with the converted image
#     detection_result = detector.detect(image)

#     if detection_result.face_landmarks:
#         # Extract features for the model
#         features = [detection_result.face_blendshapes[0][i].score for i in range(9, 23)]

#         # Ensure features are in the correct shape for prediction
#         features = np.array(features).reshape(1, -1)

#         # Make a prediction
#         prediction = model.predict(features)
#         #print("Prediction:", prediction)

#         # Annotate the frame with landmarks
#         annotated_image = draw_landmarks_on_image(frame_rgb, detection_result)
#         annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

#         # Display the annotated frame with prediction text
#         cv2.putText(annotated_image_bgr, f"Prediction: {prediction[0]}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#         cv2.imshow('Annotated Image', annotated_image_bgr)
#     else:
#         cv2.imshow('Camera Feed', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# # %%


#%%
# Run 1st: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from collections import deque
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#%%
# Run 2nd: Function to draw the face landmarks on the image.
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image

#%%
# Run 3rd: Load the pre-trained model.
data = pd.read_excel('label_cut_data.xlsx')
X = data.iloc[:, 0:14]  # Features
y = data.iloc[:, 15]    # Labels
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create a FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker2.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

#%%
# Run 4th: Real-time camera capture with sliding window prediction.
cap = cv2.VideoCapture(0)  # 0 for default camera

# Initialize a deque (double-ended queue) to store the features from the last 30 frames.
window_size = 30
feature_window = deque(maxlen=window_size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb_uint8 = np.array(frame_rgb, dtype=np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb_uint8)

    # Perform face detection with the converted image
    detection_result = detector.detect(image)

    if detection_result.face_landmarks:
        # Extract features for the model
        features = [detection_result.face_blendshapes[0][i].score for i in range(9, 23)]
        feature_window.append(features)

        # If we have enough frames in the window, make a prediction
        if len(feature_window) == window_size:
            # Average the features over the window
            averaged_features = np.mean(feature_window, axis=0).reshape(1, -1)

            # Make a prediction
            prediction = model.predict(averaged_features)
            print("Prediction:", prediction)

            # Annotate the frame with landmarks
            annotated_image = draw_landmarks_on_image(frame_rgb, detection_result)
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            # Display the annotated frame with prediction text
            cv2.putText(annotated_image_bgr, f"Prediction: {prediction[0]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Annotated Image', annotated_image_bgr)
        else:
            cv2.imshow('Camera Feed', frame)
    else:
        cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
