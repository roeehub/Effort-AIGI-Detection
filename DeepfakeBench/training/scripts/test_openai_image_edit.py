from openai import OpenAI
import base64

client = OpenAI(api_key="******************************")

response = client.images.edit(
    model="gpt-image-1",
    image=open("/Users/roeedar/Downloads/real_ffhq_00015.png", "rb"),
    prompt="Give this guy sunglasses and a hat",
    size="1024x1024",
)

img_base64 = response.data[0].b64_json

with open("out.png", "wb") as f:
    f.write(base64.b64decode(img_base64))

print("Saved out.png")
