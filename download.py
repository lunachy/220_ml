# coding:utf-8
import requests
import uuid
from PIL import Image
import os

width, height = 70, 30
url = "https://www.we.com/image_https.jsp?1490856191016"
for i in range(250):
    resp = requests.get(url)
    filename = "./captchas/" + str(uuid.uuid4()) + ".jpg"
    with open(filename, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()
        f.close()
    im = Image.open(filename)
    if im.size != (width, height):
        os.remove(filename)
    else:
        print filename
