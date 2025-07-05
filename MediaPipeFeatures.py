## this code extracts features from the video recorded from "protocol" code and save in a excel file. Also saves the videos with landmarks in the shared drive  and excludes 3s in the transitions

# #%% 
# Run 1st: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io

#%%
# Run 2nd: Function to Draw the face landmarks on the image
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image
def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  #plt.savefig('foo.png',dpi=400)
  plt.show()




  #%%
# Run 3rd: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker2.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)





#%% run to save only video features

# Define the file name
file_name = "face_blendshape_scores.txt"

# Open the file in append mode, using a buffer for writing
with io.open(file_name, 'a', buffering=1) as file:
    # Load the video
    cap = cv2.VideoCapture('video10.mp4')  # Replace 'your_video_path.mp4' with the path to your video file

    frame_count = 0  # Initialize the frame count
    landmark_frame_count = 0  # Initialize the landmark frame count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # Increment the frame count

        import numpy as np

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the RGB frame to a numpy array of type uint8
        frame_rgb_uint8 = np.array(frame_rgb, dtype=np.uint8)

        # Create an image object with the correct format
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb_uint8)

        # Perform face detection with the converted image
        detection_result = detector.detect(image)

        
        # Organize and write the blend shape scores to the file if landmarks are detected
        if detection_result.face_landmarks:
            landmark_frame_count += 1  # Increment the landmark frame count

            for i in range(9, 23):  # Include values from index 9 to 23 (24 is exclusive)
                file.write(str(detection_result.face_blendshapes[0][i].score) + "\t")

            # Write a new line for each frame
            file.write("\n")

    cap.release()

print("Face blendshape scores extraction completed.")
print("Total frames:", frame_count)
print("Frames with detected landmarks:", landmark_frame_count)



#%%
# Run to save video with original and annotated frames on the shared drive

# Define the input and output video file names
input_video_path = 'video10.mp4'
output_video_path = 'output_with_landmarks.mp4'
file_name = "face_blendshape_scores.txt"

# Open the file in append mode, using a buffer for writing
with io.open(file_name, 'a', buffering=1) as file:
    # Load the video
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0  # Initialize the frame count
    landmark_frame_count = 0  # Initialize the landmark frame count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1  # Increment the frame count

        # Convert the frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the RGB frame to a numpy array of type uint8
        frame_rgb_uint8 = np.array(frame_rgb, dtype=np.uint8)

        # Create an image object with the correct format
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb_uint8)

        # Perform face detection with the converted image
        detection_result = detector.detect(image)

        # Organize and write the blend shape scores to the file if landmarks are detected
        if detection_result.face_landmarks:
            landmark_frame_count += 1  # Increment the landmark frame count

            for i in range(9, 23):  # Include values from index 9 to 23 (24 is exclusive)
                file.write(str(detection_result.face_blendshapes[0][i].score) + "\t")

            # Write a new line for each frame
            file.write("\n")

            # Draw landmarks on the frame
            annotated_frame = draw_landmarks_on_image(frame_rgb, detection_result)
            # Convert the RGB annotated frame back to BGR format
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

            # Write the frame with landmarks to the output video
            out.write(annotated_frame_bgr)
        else:
            # Write the original frame if no landmarks are detected
            out.write(frame)

    cap.release()
    out.release()

print("Face blendshape scores extraction and video saving completed.")
print("Total frames:", frame_count)
print("Frames with detected landmarks:", landmark_frame_count)

#%% saving data from txt to csv
# Read the data
import csv

numbers = []
with open('face_blendshape_scores.txt', 'r') as file:
    for row in file:
        row = row.strip()
        if not row:
            continue
        numbers.append([float(num) for num in row.split('\t')])

# Run 7th: Save the data in a CSV file
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(numbers)

print("Data saved to output.csv")

#%% labeling 
import pandas as pd

# Function to add a numbering column
def add_numbering_column(df):
    # number of rows
    total_rows = len(df)
    
    # Configurations for numbering
    rows_per_value = 900
    max_value = 17
    
    # Create the numbering column
    numbering = []
    
    # Define numeration
    for i in range(total_rows):
        # Determine the current block
        block_number = i // rows_per_value
        # Determine the value to be assigned
        value = (block_number % max_value) + 1
        numbering.append(value)
    
    df['Numbering'] = numbering
    return df

# Paths for input CSV file and output CSV file
input_file = 'output.csv'  # Replace with the path of your input CSV file
output_file = 'data.csv'  # Replace with the path of your output CSV file

# Read the CSV file
df = pd.read_csv(input_file)

# Add the numbering column
df = add_numbering_column(df)

# Save the modified DataFrame to a new CSV file
df.to_csv(output_file, index=False)

print("Process concluded. The modified file was saved as:", output_file)


#%% cutting 10% of data on position transition
import pandas as pd

# Function to process the DataFrame
def process_dataframe(df):
    total_rows = len(df)
    rows_to_delete = set()

    # Remove the first 60 rows
    rows_to_delete.update(range(60))

    # Process each interval of 900 rows
    for start in range(900, total_rows, 900):
        end = start + 900
        if end > total_rows:
            end = total_rows

        # Remove 30 rows above and 60 rows below
        rows_to_delete.update(range(max(0, start - 30), min(total_rows, start + 60)))

    # Remove the identified rows
    df_cleaned = df.drop(index=sorted(rows_to_delete))

    return df_cleaned

# Paths for input CSV file and output CSV file
input_file = 'data.csv'  # Replace with the path of your input CSV file
output_file = 'cut_data.xlsx'  # Replace with the path of your output CSV file

# Read the CSV file
df = pd.read_csv(input_file)

# Process the DataFrame
df_cleaned = process_dataframe(df)

# Save the modified DataFrame to a new Excel file
df_cleaned.to_excel(output_file, index=False)

print("Process concluded. File modified saved as:", output_file)


#%% adding another label for in / out screen
import pandas as pd

# Function to add the new column based on the last column
def add_binary_column(df):
    last_column = df.columns[-1]  # Identifies the last column
    df['Binary'] = df[last_column].apply(lambda x: 2 if x > 9 else 1)  # Adds a new column with a value of 2 if the last column value is greater than 9, and 1 otherwise
    return df

# Paths for the input Excel file and the output Excel file
input_file = 'cut_data.xlsx'  # Replace with the path of your input Excel file
output_file = 'label_cut_data.xlsx'  # Replace with the path of your output Excel file

# Read the Excel file
df = pd.read_excel(input_file)

# Add the new column based on the last column
df = add_binary_column(df)

# Save the modified DataFrame to a new Excel file
df.to_excel(output_file, index=False)

print("Process concluded. File modified saved as:", output_file)
