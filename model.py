import base64
import datetime
import io
import time
import cv2
import mediapipe as mp
import numpy as np
from torch.hub import load


class Model:
    def __init__(self, model_path):
        # --- PyTorch model for object detection ---
        # load the model
        self.model = load(
            "ultralytics/yolov5", "custom", path=model_path, force_reload=True
        )

        # object labels to detect
        self.object_labels = [
            "bottle",
            "cigarette",
            "phone",
            "smoke",
            "vape",
        ]

        # --- Drawing and Create Face Mesh on Face ---
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.denormalize_coordinates = self.mp_draw._normalized_to_pixel_coordinates

        # --- Landmark of eye ---
        # landmark points to left eye
        self.all_left_eye_idxs = list(self.mp_face_mesh.FACEMESH_LEFT_EYE)
        # flatten and remove duplicates
        self.all_left_eye_idxs = set(np.ravel(self.all_left_eye_idxs))

        # landmark points to right eye
        self.all_right_eye_idxs = list(self.mp_face_mesh.FACEMESH_RIGHT_EYE)
        self.all_right_eye_idxs = set(np.ravel(self.all_right_eye_idxs))

        # The chosen 12 points:   P1,  P2,  P3,  P4,  P5,  P6
        self.chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
        self.chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
        self.all_chosen_idxs = self.chosen_left_eye_idxs + self.chosen_right_eye_idxs

        # lips chosen 8 points mouth: P1 - P8
        self.lips_idxs = [61, 39, 0, 269, 291, 405, 17, 181]

        # nose chosen 6 points mouth: P1 - P6
        self.nose_idxs = [33, 263, 1, 61, 291, 199]

        # --- info before start ---
        # image resize
        self.width = 800
        self.height = 450

        # color code
        self.RED = (0, 0, 255)
        self.GREEN = (0, 255, 0)

        # threshold for detection
        self.ear_thresh = 0.13
        self.mar_thresh = 1.0
        self.ear_time_thresh = 1
        self.mar_time_thresh = 1
        self.focus_time_thresh = 1
        self.ear_below_thresh_time = 0
        self.mar_below_thresh_time = 0
        self.focus_below_thresh_time = 0
        self.start_time = 0

    # --- Formula Eye Aspect Ratio (EAR) ---

    # calculate EAR
    def calculate_avg_ear(
        self, landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h
    ):
        left_ear, left_lm_coordinates = self.get_ear(
            landmarks, left_eye_idxs, image_w, image_h
        )
        right_ear, right_lm_coordinates = self.get_ear(
            landmarks, right_eye_idxs, image_w, image_h
        )
        Avg_EAR = (left_ear + right_ear) / 2.0
        return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)

    # calculate EAR for one eye
    def get_ear(self, landmarks, refer_idxs, img_width, img_height):
        try:
            coords_points = []
            for i in refer_idxs:
                lm = landmarks[i]
                coord = self.denormalize_coordinates(lm.x, lm.y, img_width, img_height)
                coords_points.append(coord)

            # eye landmark (x, y) coordinates
            P2_P6 = self.distance(coords_points[1], coords_points[5])
            P3_P5 = self.distance(coords_points[2], coords_points[4])
            P1_P4 = self.distance(coords_points[0], coords_points[3])

            # compute EAR
            ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

        except:
            ear = 0.0
        coords_points = None

        return ear, coords_points

    # Calculate MAR
    def calculate_mar(self, landmarks, refer_idxs, image_w, image_h):
        try:
            coords_points = []
            for i in refer_idxs:
                lm = landmarks[i]
                coord = self.denormalize_coordinates(lm.x, lm.y, image_w, image_h)
                coords_points.append(coord)

            P2_P8 = self.distance(coords_points[1], coords_points[7])
            P3_P7 = self.distance(coords_points[2], coords_points[6])
            P4_P6 = self.distance(coords_points[3], coords_points[5])
            P1_P5 = self.distance(coords_points[0], coords_points[4])

            mar = (P2_P8 + P3_P7 + P4_P6) / (2.0 * P1_P5)

        except:
            mar = 0.0
        coords_points = None

        return mar, coords_points

    # calculate l2-norm between two points
    def distance(self, point_1, point_2):
        dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
        return dist

    # Sleeping detection
    def detect_sleeping(self, EAR, ear_thresh, ear_time_thresh, img):
        sleep_duration = 0
        sleep_condition = False

        if EAR < ear_thresh:
            if self.ear_below_thresh_time == 0:
                self.start_time = time.perf_counter()
            self.ear_below_thresh_time = time.perf_counter() - self.start_time

            cv2.putText(
                img,
                text="Close Eyes",
                org=(15, 35),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=self.RED,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            if self.ear_below_thresh_time >= ear_time_thresh:
                sleep_duration = self.ear_below_thresh_time
                sleep_condition = True
                cv2.putText(
                    img,
                    text="Driver is sleeping!",
                    org=(15, 120),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
        else:
            self.ear_below_thresh_time = 0

        return sleep_duration, sleep_condition

    def detect_yawning(self, MAR, mar_thresh, mar_time_thresh, img):
        yawn_duration = 0
        yawn_condition = False

        if MAR > mar_thresh:
            if self.mar_below_thresh_time == 0:
                self.start_time = time.perf_counter()
            self.mar_below_thresh_time = time.perf_counter() - self.start_time

            cv2.putText(
                img,
                text="Open mouth",
                org=(15, 35),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1,
                color=self.RED,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            if self.mar_below_thresh_time >= mar_time_thresh:
                yawn_duration = self.mar_below_thresh_time
                yawn_condition = True
                cv2.putText(
                    img,
                    text="Driver is yawning!",
                    org=(15, 120),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
        else:
            self.mar_below_thresh_time = 0

        return yawn_duration, yawn_condition

    def detect_head_focus(self, landmarks, img_w, img_h, focus_time_thresh, img):
        notfocus_duration = 0
        notfocus_condition = False

        try:
            face_2d = []
            face_3d = []

            for idx in self.nose_idxs:
                lm = landmarks[idx]
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                x, y = int(lm.x * img_w), int(lm.y * img_h)

                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])

            # Get 2d coord
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            # Cam Matrix
            cam_matrix = np.array(
                [
                    [focal_length, 0, img_w / 2],
                    [0, focal_length, img_h / 2],
                    [0, 0, 1],
                ]
            )

            # Distortion Matrix
            distortion_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            _, rotation_vec, _ = cv2.solvePnP(
                face_3d, face_2d, cam_matrix, distortion_matrix
            )

            # rmat
            rmat, _ = cv2.Rodrigues(rotation_vec)

            # Getting Angles
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360

            # Calculated Axis rot angle
            if y < -10:
                text = "Looking Right"
            elif y > 10:
                text = "Looking Left"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"

            if text != "Forward":
                if self.focus_below_thresh_time == 0:
                    self.start_time = time.perf_counter()
                self.focus_below_thresh_time = time.perf_counter() - self.start_time

                if self.focus_below_thresh_time >= focus_time_thresh:
                    notfocus_duration = self.focus_below_thresh_time
                    notfocus_condition = True
                    cv2.putText(
                        img,
                        text="Driver not Focus!",
                        org=(15, 120),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
            else:
                self.focus_below_thresh_time = 0

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            return text, p1, p2, x, y, notfocus_duration, notfocus_condition

        except Exception as e:
            print(f"Error: {e}")
            return (
                "Forward",
                (0, 0),
                (0, 0),
                0,
                0,
                notfocus_duration,
                notfocus_condition,
            )

    # Mediapipe Face Mesh
    def detect_face_mesh(self, img):
        face_detection_results = None

        # resize the video
        img = cv2.resize(img, (self.width, self.height))

        # Convert the BGR to RGB image
        img.flags.writeable = False
        img_h, img_w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            for _ in results.multi_face_landmarks:
                # EAR and MAR Calculate
                EAR, _ = self.calculate_avg_ear(
                    landmarks,
                    self.chosen_left_eye_idxs,
                    self.chosen_right_eye_idxs,
                    img_w,
                    img_h,
                )
                MAR, _ = self.calculate_mar(landmarks, self.lips_idxs, img_w, img_h)

                sleep_duration, sleep_condition = self.detect_sleeping(
                    EAR, self.ear_thresh, self.ear_time_thresh, img
                )
                yawn_duration, yawn_condition = self.detect_yawning(
                    MAR, self.mar_thresh, self.mar_time_thresh, img
                )

                head_pose_text, p1, p2, x, y, notfocus_duration, notfocus_condition = (
                    self.detect_head_focus(
                        landmarks, img_w, img_h, self.focus_time_thresh, img
                    )
                )

                cv2.line(img, p1, p2, (255, 0, 0), 3)
                cv2.putText(
                    img,
                    head_pose_text,
                    (15, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    f"x: {np.round(x, 2)}",
                    (600, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    img,
                    f"y: {np.round(y, 2)}",
                    (600, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                if (
                    sleep_condition == False
                    and yawn_condition == False
                    and notfocus_condition == False
                ):
                    cv2.circle(img, (15, 15), 2, self.GREEN, -1)
                    cv2.putText(
                        img,
                        "Steady",
                        (15, 35),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        self.GREEN,
                        2,
                    )

                else:
                    data = {
                        "time": datetime.datetime.now().isoformat(),
                        "ear": EAR,
                        "mar": MAR,
                        "sleep_duration": sleep_duration,
                        "yawning_duration": yawn_duration,
                        "focus_duration": notfocus_duration,
                    }

                    # add data to detection_results
                    face_detection_results = data

        fps = int(1 / (time.perf_counter() - self.start_time))
        cv2.putText(
            img, f"FPS: {fps}", (15, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2
        )

        return face_detection_results

    # Detect objects using YOLOv5
    def detect_objects(self, img):
        # Perform object detection on the img
        results = self.model(img)

        # Flags to indicate if any object is detected in the current img
        object_detected = {label: False for label in self.object_labels}

        # Draw bounding boxes on the img
        for detection in results.xyxy[0]:
            x1, y1, x2, y2, confidence, class_idx = detection
            label = self.model.names[int(class_idx)]

            # Check if the label is in the object_labels list
            if label in self.object_labels:
                # Convert coordinates to integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label and confidence score
                text = f"{label}: {confidence:.2f}"
                cv2.putText(
                    img,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # Track detected objects
                object_detected[label] = True

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # encode
        _, buffer = cv2.imencode(".jpg", img)
        io_buf = io.BytesIO(buffer)

        # decode
        decoded_img = cv2.imdecode(np.frombuffer(io_buf.getbuffer(), np.uint8), -1)

        return decoded_img, object_detected

    def convert_base64_to_img(self, payload):
        nparr = np.frombuffer(base64.b64decode(payload), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def analyze(self, payload):
        # Convert base64 to img
        img = self.convert_base64_to_img(payload)

        # Detect drowsiness, yawning, and head focus
        face_detection_results = self.detect_face_mesh(img)

        # Detect objects in the img
        decoded_img, object_detected = self.detect_objects(img)

        return decoded_img, object_detected, face_detection_results
