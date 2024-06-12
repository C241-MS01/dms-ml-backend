from cloud_storage import CloudStorage
from db import Mysql
from dotenv import load_dotenv
from flask import Flask, render_template
from flask_mqtt import Mqtt
from model import Model
from os import getenv
from structlog import get_logger
from video import Video

load_dotenv()


storage = CloudStorage(
    project_id=getenv("GOOGLE_PROJECT_ID"),
    credentials_path=getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    bucket_name=getenv("GOOGLE_STORAGE_BUCKET_NAME"),
)

db = Mysql(
    host=getenv("MYSQL_HOST"),
    user=getenv("MYSQL_USER"),
    password=getenv("MYSQL_PASSWORD"),
    database=getenv("MYSQL_DATABASE"),
)

video = Video()

ml_model = Model(getenv("MODEL_PATH"))

app = Flask(__name__)
app.logger = get_logger()

app.config["HTTP_URL"] = getenv("HTTP_URL")
app.config["HTTP_PORT"] = getenv("HTTP_PORT")
app.config["MQTT_BROKER_URL"] = getenv("MQTT_BROKER_URL")
app.config["MQTT_BROKER_PORT"] = int(getenv("MQTT_BROKER_PORT"))

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
        "open_stream/#",
        "close_stream/#",
    ]

    for topic in topics:
        mqtt_client.subscribe(topic)
        app.logger.info(f"Subscribed to {topic}")


@mqtt_client.on_message()
def handle_message(client, userdata, message):
    try:
        topic = message.topic
        payload = message.payload

        if topic.startswith("open_stream"):
            open_stream(payload)
        elif topic.startswith("stream"):
            process_img(payload)
        elif topic.startswith("close_stream"):
            close_stream()

    except Exception as e:
        app.logger.error(e)
        return


def open_stream(vehicle_id):
    try:
        db.cursor.execute(
            "SELECT id FROM vehicles WHERE id = %s", (vehicle_id,))
        vehicle = db.cursor.fetchone()

        if not vehicle:
            db.cursor.execute(
                "INSERT INTO vehicles (id) VALUES (%s)", (vehicle_id,))
            db.connection.commit()
            app.logger.info(f"Vehicle {vehicle_id} added to database")
            return

        db.connection.commit()
        app.logger.info(f"Vehicle {vehicle_id} already exists in database")

    except Exception as e:
        app.logger.error(e)
        return


def process_img(payload):
    try:
        decoded_img, object_detected, face_detection_results = ml_model.analyze(payload)

        if any(value for value in object_detected.values()) or face_detection_results:
            app.logger.info(f"Object detected: {object_detected}")
            app.logger.info(f"Face detection results: {
                            face_detection_results}")

            for key, value in object_detected.items():
                if value:
                    mqtt_client.publish("alert", f"{key} detected")

            if face_detection_results:
                mqtt_client.publish("alert", "drowsiness detected")

            video.write_to_buffer(decoded_img)

    except Exception as e:
        app.logger.error(e)
        return


def close_stream():
    # temporary
    try:
        if len(video.frames) == 0:
            app.logger.info("no frames to compile")
            return

        app.logger.info(f"compiling {len(video.frames)} frames into video")

        filename = video.save_to_file()

        app.logger.info(f"video saved to {filename}")

        # store to db, and upload to cloud storage

        # storage.upload(filename, "videos")
        # db.cursor.execute(
        #     "INSERT INTO detection_history (video_url) VALUES (%s)", (filename)
        # )

    except Exception as e:
        app.logger.error(e)
        return


# Video stream test
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
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
