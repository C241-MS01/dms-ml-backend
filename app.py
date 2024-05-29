from flask import Flask, render_template
from flask_mqtt import Mqtt
from dotenv import load_dotenv
from structlog import get_logger
import os

load_dotenv()

app = Flask(__name__)
app.config["HTTP_URL"] = os.getenv("HTTP_URL")
app.config["HTTP_PORT"] = os.getenv("HTTP_PORT")
app.config["MQTT_BROKER_URL"] = os.getenv("MQTT_BROKER_URL")
app.config["MQTT_BROKER_PORT"] = int(os.getenv("MQTT_BROKER_PORT"))
app.logger = get_logger()
mqtt_client = Mqtt(app)


@mqtt_client.on_connect()
def handle_connect(client, userdata, flags, rc):
    if rc != 0:
        print(f"Connection failed with result code {rc}")
        return

    app.logger.info(
        f"Connected to MQTT broker at {app.config['MQTT_BROKER_URL']}:{app.config['MQTT_BROKER_PORT']}"
    )

    mqtt_client.subscribe("video")
    app.logger.info("Subscribed to topic 'video'")


@mqtt_client.on_message()
def handle_message(client, userdata, message):
    app.logger.info(f"Received message: {message.payload.decode()}")


# Video stream test
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stream")
def stream():
    return render_template("streamer.html")


def main():
    app.run(host=app.config["HTTP_URL"], port=app.config["HTTP_PORT"])


if __name__ == "__main__":
    main()
