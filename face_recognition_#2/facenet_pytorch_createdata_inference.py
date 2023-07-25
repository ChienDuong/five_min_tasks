import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define MTCNN module
mtcnn = MTCNN(keep_all=True, device=device)

# Define Inception Resnet V1 module
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def capture_images(video_source=4, num_images=20):
    video_capture = cv2.VideoCapture(video_source)
    images = []
    embeddings = []
    captured = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture image")
            break

        # Detect faces
        boxes, probs = mtcnn.detect(frame)

        # Draw bounding box
        if boxes is not None:
            for box in boxes:
                cv2.rectangle(frame, 
                              (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), 
                              (0, 255, 0), 
                              2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('\r'):  # Enter key is pressed
            print('Capture image {}'.format(captured + 1))

            # Extract face embeddings
            faces = mtcnn(frame)
            if faces is not None:
                for face in faces:
                    face_embedding = model(face.unsqueeze(0).to(device))

                    images.append(frame)
                    embeddings.append(face_embedding.cpu().detach().numpy())
                    captured += 1

            # If 20 images have been captured, break from the loop
            if captured >= num_images:
                break

    # Save embeddings for later use
    known_face_encoding = np.array(embeddings).mean(axis=0)
    np.save('embeddings.npy', known_face_encoding)

    video_capture.release()
    cv2.destroyAllWindows()

    return known_face_encoding


def recognize_person(known_face_encoding, video_source=4):
    if known_face_encoding is None:
        known_face_encoding = np.load('embeddings.npy')

    video_capture = cv2.VideoCapture(video_source)

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Detect faces
        boxes, _ = mtcnn.detect(frame)

        # Draw bounding box
        if boxes is not None:
            for box in boxes:
                cv2.rectangle(frame,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              (0, 255, 0),
                              2)

            # Extract face embeddings
            faces = mtcnn(frame)
            if faces is not None:
                for face in faces:
                    current_embedding = model(face.unsqueeze(0).to(device)).cpu().detach().numpy().flatten()
                    print(current_embedding)
                    np.save('current_embeddings.npy', np.array(current_embedding))
                    # Compare with saved embeddings
                    distances = cosine(current_embedding, known_face_encoding.flatten())

                    # Check if any of the distances is below a certain threshold
                    if distances < 0.6:
                        cv2.putText(frame, "Person recognized!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "No match found!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


known_face_encoding = None
known_face_encoding = capture_images(video_source=4, num_images=20)
print('Phase 1 complete. Now beginning phase 2...')
recognize_person(known_face_encoding, video_source=4)
