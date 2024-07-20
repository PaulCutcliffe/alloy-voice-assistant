import requests
from requests.auth import HTTPBasicAuth
import base64
from datetime import datetime

class WordPressPublisher:
    def __init__(self, site_url, username, app_password):
        self.site_url = site_url
        self.api_url = f"{site_url}/wp-json/wp/v2"
        self.auth = HTTPBasicAuth(username, app_password)

    def create_post(self, title, content, image_path, capture_time):
        # First, upload the image
        media_id = self.upload_media(image_path)

        # Ensure title and content are properly encoded
        title = title.encode('utf-8').decode('utf-8')
        content = content.encode('utf-8').decode('utf-8')

        post_data = {
            'title': title,
            'content': content,
            'status': 'publish',
            'featured_media': media_id,
            'date': capture_time.isoformat()
        }

        response = requests.post(f"{self.api_url}/posts", json=post_data, auth=self.auth)
        
        if response.status_code == 201:
            print(f"Post created successfully. Post ID: {response.json()['id']}")
            return response.json()['id']
        else:
            print(f"Failed to create post. Status code: {response.status_code}")
            print(response.text)
            return None

    def upload_media(self, file_path):
        with open(file_path, 'rb') as file:
            media_data = file.read()
        
        file_name = file_path.split('/')[-1]
        content_type = 'image/gif' if file_name.endswith('.gif') else 'image/jpeg'
        
        headers = {
            'Content-Type': content_type,
            'Content-Disposition': f'attachment; filename={file_name}'
        }

        response = requests.post(
            f"{self.api_url}/media",
            data=media_data,
            headers=headers,
            auth=self.auth
        )

        if response.status_code == 201:
            print(f"Media uploaded successfully. Media ID: {response.json()['id']}")
            return response.json()['id']
        else:
            print(f"Failed to upload media. Status code: {response.status_code}")
            print(response.text)
            return None