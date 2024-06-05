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

ml_model = Model()

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
        f"Connected to MQTT broker at {app.config['MQTT_BROKER_URL']}:{app.config['MQTT_BROKER_PORT']}"
    )

    topics = [
        "stream/#",
        "close_stream/#",
    ]

    for topic in topics:
        mqtt_client.subscribe(topic)
        app.logger.info(f"Subscribed to {topic}")


@mqtt_client.on_message()
def handle_message(client, userdata, message):
    topic = message.topic
    payload = message.payload

    app.logger.info(f"Received message on topic {topic}")
    app.logger.info(f"frames: {len(video.frames)}")

    match topic:
        case "stream":
            process_frame(payload)
        case "close_stream":
            close_stream()


def process_frame(payload):
    try:
        # global last_detection_time

        payload.decode("utf-8")
        img = video.convert_base64_to_img(payload)

        # _, _, _, ... = ml_model.analyze(img)

        # if detected:
        #     video.write_to_buffer(img)
        #     mqtt_client.publish("alert", "drowsiness detected")

        video.write_to_buffer(img)  # temporary

    except Exception as e:
        app.logger.error(e)
        return


def close_stream():
    # temporary
    try:
        if len(video.frames) == 0:
            return

        filename = video.save_to_file()
        video.release_video_writer()

        app.logger.info("video saved to file")

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
