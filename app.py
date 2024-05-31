from flask import Flask, render_template
from flask_mqtt import Mqtt
from model import Model
from dotenv import load_dotenv
from structlog import get_logger
from os import getenv

load_dotenv()

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

    topic = "video/#"

    mqtt_client.subscribe(topic)
    app.logger.info(f"Subscribed to topic: {topic}")


@mqtt_client.on_message()
def handle_message(client, userdata, message):
    try:
        message.payload.decode("utf-8")
        ml_model.process_frame(message.payload)

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
