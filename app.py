import datetime
import os
import uuid
from cloud_storage import CloudStorage
from db import Mysql
from dotenv import load_dotenv
from flask import Flask, render_template
from flask_mqtt import Mqtt
from model import Model
from structlog import get_logger
from video import Video

load_dotenv()


storage = CloudStorage(
    project_id=os.getenv("GOOGLE_PROJECT_ID"),
    credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    bucket_name=os.getenv("GOOGLE_STORAGE_BUCKET_NAME"),
)

db = Mysql(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE"),
)

video = Video()

ml_model = Model(os.getenv("MODEL_PATH"))

app = Flask(__name__)
app.logger = get_logger()

app.config["HTTP_URL"] = os.getenv("HTTP_URL")
app.config["HTTP_PORT"] = os.getenv("HTTP_PORT")
app.config["MQTT_BROKER_URL"] = os.getenv("MQTT_BROKER_URL")
app.config["MQTT_BROKER_PORT"] = int(os.getenv("MQTT_BROKER_PORT"))

mqtt_client = Mqtt(app, connect_async=True)


@mqtt_client.on_connect()
def handle_connect(client, userdata, flags, rc):
    if rc != 0:
        print(f"Connection failed with result code {rc}")
        return

    app.logger.info(
        f"Connected to MQTT broker at {app.config['MQTT_BROKER_URL']}:{
            app.config['MQTT_BROKER_PORT']}"
    )

    topics = [
        "stream/#",
        "open_stream",
        "close_stream",
    ]

    for topic in topics:
        mqtt_client.subscribe(topic)
        app.logger.info(f"{topic} subscribed")


@mqtt_client.on_message()
def handle_message(client, userdata, message):
    try:
        topic = message.topic
        payload = message.payload

        if topic.startswith("open_stream"):
            open_stream(payload.decode("utf-8"))

        elif topic.startswith("stream"):
            vehicle_uuid = None
            topic_parts = topic.split("/")

            if len(topic_parts) > 1:
                vehicle_uuid = topic_parts[1]

            process_img(vehicle_uuid, payload)

        elif topic.startswith("close_stream"):
            close_stream(payload.decode("utf-8"))

    except Exception as e:
        app.logger.error(e)
        return


def open_stream(vehicle_uuid):
    if vehicle_uuid in video.frames:
        app.logger.info(f"[open_stream] {vehicle_uuid}: stream already open")
        return

    app.logger.info(f"[open_stream] {vehicle_uuid}: opening stream")

    video.frames[vehicle_uuid] = list()
    ml_model.detections[vehicle_uuid] = list()

    db.cursor.execute(
        "SELECT EXISTS(SELECT 1 FROM vehicles WHERE uuid = %s)", (vehicle_uuid,))
    vehicle = db.cursor.fetchone()[0]

    if not vehicle:
        app.logger.info(
            f"[open_stream] {vehicle_uuid}: adding vehicle to database"
        )
        db.cursor.execute(
            "INSERT INTO vehicles (uuid) VALUES (%s)", (vehicle_uuid,)
        )
        db.connection.commit()
        return
    db.connection.commit()

    app.logger.info(
        f"[open_stream] {vehicle_uuid}: vehicle already exists in database"
    )


def process_img(vehicle_uuid, payload):
    if vehicle_uuid is None:
        return

    if vehicle_uuid not in video.frames:
        app.logger.info(f"[stream] {vehicle_uuid}: stream not open")
        return

    decoded_img, object_detected, face_detection_results = ml_model.analyze(
        payload
    )

    if any(value for value in object_detected.values()) or face_detection_results["ear"] != 0:
        for key, value in object_detected.items():
            if value:
                mqtt_client.publish(f"alert/{vehicle_uuid}", f"{key}")

        if face_detection_results:
            mqtt_client.publish(f"alert/{vehicle_uuid}", "not focus")

        ml_model.detections[vehicle_uuid].append({
            "object_detected": object_detected,
            "face_detection_results": face_detection_results
        })
        video.write_to_buffer(vehicle_uuid, decoded_img)


def close_stream(vehicle_uuid):
    if len(video.frames[vehicle_uuid]) == 0:
        app.logger.info(f"[close_stream] {vehicle_uuid}: no frames to compile")
        return

    app.logger.info(f"[close_stream] {vehicle_uuid}: closing stream")

    filename = video.save_to_file(vehicle_uuid)

    app.logger.info(
        f"[close_stream] {vehicle_uuid}: video saved to {filename}"
    )

    video_url = storage.upload(
        filename,
        f"{datetime.date.today()}/{vehicle_uuid}-{str(uuid.uuid4())}.mp4"
    )

    app.logger.info(
        f"[close_stream] {vehicle_uuid}: video uploaded to {video_url}"
    )

    db.cursor.execute(
        "SELECT id FROM vehicles WHERE uuid = %s", (vehicle_uuid,)
    )
    vehicle_id = db.cursor.fetchone()[0]

    if not vehicle_id:
        app.logger.error(
            f"[close_stream] {vehicle_uuid}: vehicle id not found"
        )
        return

    db.cursor.execute(
        "INSERT INTO videos (vehicle_id, uuid, url) VALUES (%s, %s, %s)",
        (vehicle_id, str(uuid.uuid4()), video_url),
    )
    video_id = db.cursor.lastrowid

    for detection in ml_model.detections[vehicle_uuid]:
        if not any(value for value in detection["object_detected"].values()) and not detection["face_detection_results"]["ear"]:
            continue

        db.cursor.execute(
            "INSERT INTO alerts (video_id, uuid, ear, mar, sleep_duration, yawning_duration, focus_duration, time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                video_id,
                str(uuid.uuid4()),
                detection["face_detection_results"]["ear"],
                detection["face_detection_results"]["mar"],
                detection["face_detection_results"]["sleep_duration"],
                detection["face_detection_results"]["yawning_duration"],
                detection["face_detection_results"]["focus_duration"],
                detection["face_detection_results"]["time"],
            ),
        )
        alerts_id = db.cursor.lastrowid

        if detection["object_detected"]:
            detected_str = ""
            for key, value in detection["object_detected"].items():
                if value:
                    detected_str += f"{key}, "

            match detected_str:
                case "":
                    detected_str = None
                case _:
                    detected_str = detected_str[:-2]

            db.cursor.execute(
                "UPDATE alerts SET object_detected = %s WHERE id = %s",
                (detected_str, alerts_id),
            )

    db.connection.commit()

    if vehicle_uuid in video.frames:
        del video.frames[vehicle_uuid]

    if vehicle_uuid in ml_model.detections:
        del ml_model.detections[vehicle_uuid]

    if os.path.exists(filename):
        os.remove(filename)

    app.logger.info(f"[close_stream] {vehicle_uuid}: stream closed")


@ app.route("/")
def index():
    return render_template("index.html")


@ app.route("/stream")
def stream():
    return render_template("streamer.html")


def main():
    app.run(
        host=app.config["HTTP_URL"],
        port=app.config["HTTP_PORT"],
        # debug=True,
    )


if __name__ == "__main__":
    main()
