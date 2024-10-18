#%%
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
  ax.set_title("Face BlendshapesTESTE")
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
file_name = "face_blendshape_scores1.txt"

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
file_name = "face_blendshape_scores_test.txt"

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

#%%
# Read the data
import csv

numbers = []
with open('features.txt', 'r') as file:
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

#%% labeling again

import pandas as pd

# Função para adicionar a coluna com a numeração
def add_numbering_column(df):
    # Número total de linhas
    total_rows = len(df)
    
    # Configurações para numeração
    rows_per_value = 900
    max_value = 17
    
    # Cria a coluna de numeração
    numbering = []
    
    # Define a numeração
    for i in range(total_rows):
        # Determine o bloco atual
        block_number = i // rows_per_value
        # Determine o valor a ser atribuído
        value = (block_number % max_value) + 1
        numbering.append(value)
    
    df['Numbering'] = numbering
    return df

# Caminho para o arquivo CSV de entrada e saída
input_file = 'output.csv'  # Substitua pelo caminho do seu arquivo CSV de entrada
output_file = 'data.csv'  # Substitua pelo caminho do seu arquivo CSV de saída

# Ler o arquivo CSV
df = pd.read_csv(input_file)

# Adicionar a coluna de numeração
df = add_numbering_column(df)

# Salvar o DataFrame modificado em um novo arquivo CSV
df.to_csv(output_file, index=False)

print("Processamento concluído. Arquivo modificado salvo como:", output_file)


#%% cutting 10% of data on position transition
import pandas as pd

# Função para processar o DataFrame
def process_dataframe(df):
    total_rows = len(df)
    rows_to_delete = set()

    # Remover as primeiras 60 linhas
    rows_to_delete.update(range(60))

    # Processar cada intervalo de 900 linhas
    for start in range(900, total_rows, 900):
        end = start + 900
        if end > total_rows:
            end = total_rows

        # Remover 30 linhas acima e 60 linhas abaixo
        rows_to_delete.update(range(max(0, start - 30), min(total_rows, start + 60)))

    # Remover as linhas identificadas
    df_cleaned = df.drop(index=sorted(rows_to_delete))

    return df_cleaned

# Caminho para o arquivo de entrada e saída
input_file = 'data.csv'  # Substitua pelo caminho do seu arquivo de entrada
output_file = 'cut_data2.xlsx'  # Substitua pelo caminho do seu arquivo de saída

# Ler o arquivo Excel
df = pd.read_csv(input_file)

# Processar o DataFrame
df_cleaned = process_dataframe(df)

# Salvar o DataFrame modificado em um novo arquivo Excel
df_cleaned.to_excel(output_file, index=False)

print("Processamento concluído. Arquivo modificado salvo como:", output_file)


#%% adicionando outro label para in / out screen
import pandas as pd

# Função para adicionar a nova coluna baseada na última coluna
def add_binary_column(df):
    last_column = df.columns[-1]  # Identifica a última coluna
    df['Binary'] = df[last_column].apply(lambda x: 2 if x > 9 else 1)  # Adiciona a nova coluna com base na última coluna
    return df

# Caminho para o arquivo Excel de entrada e saída
input_file = 'cut_data2.xlsx'  # Substitua pelo caminho do seu arquivo Excel de entrada
output_file = 'label_cut_data.xlsx'  # Substitua pelo caminho do seu arquivo Excel de saída

# Ler o arquivo Excel
df = pd.read_excel(input_file)

# Adicionar a nova coluna baseada na última coluna
df = add_binary_column(df)

# Salvar o DataFrame modificado em um novo arquivo Excel
df.to_excel(output_file, index=False)

print("Processamento concluído. Arquivo modificado salvo como:", output_file)

