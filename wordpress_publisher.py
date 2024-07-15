import requests
from requests.auth import HTTPBasicAuth
import base64
from datetime import datetime

class WordPressPublisher:
    def __init__(self, site_url, username, app_password):
        self.site_url = site_url
        self.api_url = f"{site_url}/wp-json/wp/v2"
        self.auth = HTTPBasicAuth(username, app_password)

    def create_post(self, title, content, image_path, capture_time, prompt):
        # First, upload the image
        media_id = self.upload_image(image_path)

        # Ensure title and content are properly encoded
        title = title.encode('utf-8').decode('utf-8')
        content = content.encode('utf-8').decode('utf-8')

        # Format the content to include the prompt
        full_content = f"{content}\n\n<h3>AI Prompt Used:</h3>\n<p>{prompt}</p>"

        post_data = {
            'title': title,
            'content': full_content,
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

    def upload_image(self, image_path):
        with open(image_path, 'rb') as img:
            image_data = img.read()
        
        headers = {
            'Content-Type': 'image/jpeg',
            'Content-Disposition': f'attachment; filename={image_path.split("/")[-1]}'
        }

        response = requests.post(
            f"{self.api_url}/media",
            data=image_data,
            headers=headers,
            auth=self.auth
        )

        if response.status_code == 201:
            print(f"Image uploaded successfully. Media ID: {response.json()['id']}")
            return response.json()['id']
        else:
            print(f"Failed to upload image. Status code: {response.status_code}")
            print(response.text)
            return None
