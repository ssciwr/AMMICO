from google.cloud import vision
import io


def detect_text(subdict):
    """Detects text in the file."""

    path = subdict["filename"]
    client = vision.ImageAnnotatorClient()

    with io.open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    subdict = {"text": []}
    for text in texts:
        subdict["text"].append(text.description)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return subdict
