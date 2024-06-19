# DMS ML Backend

## Description

This Flask application serves as the backend for processing [MQTT](https://mqtt.org/) messages related to vehicle streams and feeding them to a machine learning (ML) model. It then stores the footages to the [Google Cloud Storage](https://cloud.google.com/storage/) bucket and the metric data to the [MySQL](https://www.mysql.com/) database.

### MQTT Topics

The application subscribes to the following MQTT topics:

- `open_stream`: Triggers stream initiation. Expects the payload to be the vehicle's UUID.
- `stream/#`: Receives image frame buffers from the IoT device. The UUID is extracted from the subtopic.
- `close_stream`: Triggers stream termination. Expects the payload to be the vehicle's UUID.

### MQTT Message Flow

- `open_stream`:
  - Verifies if the received UUID exists in the database.
  - If found, creates in-memory lists to store footage and metrics.
  - If not found, inserts the vehicle into the database and then creates the lists.

- `stream/#`:
  - Identifies the vehicle UUID from the subtopic.
  - Converts the buffer images to base64 strings for live streaming (published to `base64/{UUID}` topic).
  - Feeds the images to the ML model for processing.
  - If distractions or objects are detected, efficiently appends the frames and metrics to the corresponding in-memory lists.

- `close_stream`:
  - Checks if the lists are empty. If so, simply deletes them.
  - If not empty:
    - Saves footages to Google Cloud Storage.
    - Saves metrics and the corresponding footage URLs to the MySQL database.
    - Deletes the in-memory lists.

## Getting Started

1. If you do not use devcontainer, ensure you have [Python](https://www.python.org/downloads/)  3.12 installed:

    ```bash
    python --version
    ```

2. Create virtual environment and activate it:

    ```bash
    test -d venv || (python -m venv .venv && . .venv/bin/activate)
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements-pytorch.txt && pip install -r requirements.txt
    ```

4. Create a copy of the `.env.example` file and rename it to `.env`:

    ```bash
    cp .env.example .env
    ```

    Update the configuration values as needed.

5. Run the application:

    ```bash
    python app.py
    ```

## License

This project is licensed under the [MIT License](LICENSE), providing an open and permissive licensing approach for further development and distribution.
