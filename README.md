# DMS ML Backend

## Description

Flask application that serves as the backend for relaying [MQTT](https://mqtt.org/) messages to the ML model. It then saves the footages to the [Google Cloud Storage](https://cloud.google.com/storage) bucket and the metric data to the [MySQL](https://www.mysql.com/) database.

## Getting Started

1. If you do not use devcontainer, ensure you have [Python](https://www.python.org/downloads/)  3.12 installed:

    ```bash
    python --version
    ```

2. Create a copy of the `.env.example` file and rename it to `.env`:

    ```bash
    cp .env.example .env
    ```

    Update the configuration values as needed.

3. Create virtual environment and activate it:

    ```bash
    test -d venv || (python -m venv .venv && . .venv/bin/activate)
    ```

4. Install the required Python packages:

    ```bash
    pip install -r requirements-pytorch.txt && pip install -r requirements.txt
    ```

5. Run the application:

    ```bash
    python app.py
    ```

## License

This project is licensed under the [MIT License](LICENSE), providing an open and permissive licensing approach for further development and distribution.
