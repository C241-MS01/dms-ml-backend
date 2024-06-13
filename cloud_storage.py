from google.cloud import storage
from google.oauth2 import service_account


class CloudStorage:
    def __init__(self, project_id, credentials_path, bucket_name):
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        self.client = storage.Client(
            project=project_id,
            credentials=credentials,
        )
        self.bucket = self.client.bucket(bucket_name)

    def upload(self, content, destination) -> str:
        blob = self.bucket.blob(destination)
        blob.upload_from_filename(content)
        blob.make_public()
        return blob.public_url
