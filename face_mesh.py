import cv2
import time
import mediapipe as mp
import numpy as np

# capture video
cap = cv2.VideoCapture(0)

# --- Drawing and Create Face Mesh on Face ---
# drawing on faces
mpDraw = mp.solutions.drawing_utils

# create face mesh
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()

# convert nomalized to pixel coordinates
denormalize_coordinates = mpDraw._normalized_to_pixel_coordinates

# drawing specification
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# --- Landmark of eye ---
# landmark points to left eye
all_left_eye_idxs = list(mpFaceMesh.FACEMESH_LEFT_EYE)
# flatten and remove duplicates
all_left_eye_idxs = set(np.ravel(all_left_eye_idxs))

# landmark points to right eye
all_right_eye_idxs = list(mpFaceMesh.FACEMESH_RIGHT_EYE)
all_right_eye_idxs = set(np.ravel(all_right_eye_idxs))

# Combined for plotting Landmark points for both eye
all_idxs = all_left_eye_idxs.union(all_right_eye_idxs)

# The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs


# --- Formula Eye Aspect Ratio (EAR) ---
# calculate l2-norm between two points
def distance(point_1, point_2):
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


# calculate EAR for one eye
def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # eye landmark (x, y) coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # compute EAR
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
    coords_points = None

    return ear, coords_points


# calculate EAR
def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(
        landmarks, right_eye_idxs, image_w, image_h
    )
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


# --- info before start ---
# image resize
width = 800
height = 450

# color code
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# threshold for detection
ear_thresh = 0.13
time_thresh = 3
ear_below_thersh_time = 0
start_time = 0

while cap.isOpened():
    # read video
    success, img = cap.read()
    if not success:
        print("Not success")
        # If loading a video, use 'break' instead of 'continue'
        continue

    # resize the video
    img = cv2.resize(img, (width, height))

    # Convert the BGR to RGB image
    img.flags.writeable = False
    img_h, img_w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(img)

    img.flags.writeable = True
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        EAR, coordinates = calculate_avg_ear(
            landmarks, chosen_left_eye_idxs, chosen_right_eye_idxs, img_w, img_h
        )

        print(EAR)
        for faceLms in results.multi_face_landmarks:
            for idx in all_chosen_idxs:
                landmark = faceLms.landmark[idx]
                x, y, z = landmark.x, landmark.y, landmark.z

                if EAR > ear_thresh:
                    ear_below_thresh_time = 0
                    cv2.circle(
                        img,
                        (int(x * img.shape[1]), int(y * img.shape[0])),
                        2,
                        GREEN,
                        -1,
                    )

                    cv2.putText(
                        img,
                        text="Steady",
                        org=(15, 35),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

                else:
                    if ear_below_thresh_time == 0:
                        start_time = time.perf_counter()
                    ear_below_thresh_time = time.perf_counter() - start_time
                    cv2.circle(
                        img, (int(x * img.shape[1]), int(y * img.shape[0])), 2, RED, -1
                    )

                    cv2.putText(
                        img,
                        text="Drowsy",
                        org=(15, 35),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )

                    if ear_below_thresh_time >= time_thresh:
                        cv2.putText(
                            img,
                            text="Driver is sleeping!",
                            org=(15, 100),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1,
                            color=(0, 0, 255),
                            thickness=2,
                            lineType=cv2.LINE_AA,
                        )
                        print("Driver is sleeping!")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(
        img, f"FPS:{int(fps)}", (15, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3
    )
    """
    (20, 70)    => Font place or location
    3           => Scale
    (0, 255, 0) => Color
    3           => Thickness
    """

    cv2.imshow("image", img)
    cv2.waitKey(1)
