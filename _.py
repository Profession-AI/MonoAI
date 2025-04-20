from coicoi.models.image_model import ImageModel

image_model = ImageModel(
    provider="openai",
    model="dall-e-3"
)

print(image_model.generate("a white siamese cat"))

