# Text detector

*Tutorial coming soon!*

Text on the images can be extracted using the `TextDetector` class (`text` module). The text is initally extracted using the Google Cloud Vision API and then translated into English with googletrans. The translated text is cleaned of whitespace, linebreaks, and numbers using Python syntax and spaCy. 

Please note that for the Google Cloud Vision API (the TextDetector class) you need to set a key in order to process the images. This key is ideally set as an environment variable using for example
```
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "<path_to_your_service_account_key>.json"
```
where you place the key on your Google Drive if running on colab, or place it in a local folder on your machine.

## Read your image data into `ammico`

`ammico` reads in files from a directory. You can iterate through directories in a recursive manner and filter by extensions. Note that the order of the files may vary on different OS. Reading in these files creates a dictionary `image_dict`, with one entry per image file, containing the file path and filename. This dictionary is the main data structure that ammico operates on and is extended successively with each detector run as explained below.

For reading in the files, the ammico function `find_files` is used, with optional keywords:

| input key | input type | possible input values |
| --------- | ---------- | --------------------- |
`path` | `str` | the directory containing the image files (defaults to the location set by environment variable `AMMICO_DATA_HOME`) |
| `pattern` | `str\|list` | the file extensions to consider (defaults to "png", "jpg", "jpeg", "gif", "webp", "avif", "tiff") |
| `recursive` | `bool` | include subdirectories recursively (defaults to `True`) |
| `limit` | `int` | maximum number of files to read (defaults to `20`, for all images set to `None` or `-1`) |
| `random_seed` | `int` | the random seed for shuffling the images; applies when only a few images are read and the selection should be preserved (defaults to `None`) |

## Example usage

The text detection is carried out using the following method call:
```
for key in image_dict.keys():
    image_dict[key] = ammico.TextDetector(
        image_dict[key],  
    ).analyse_image()

```

A detailed description of the output keys and data types is given in the following table.

| output key | output type | output value |
| ---------- | ----------- | ------------ |
| `text` | `str` | the extracted text in the original language |
| `text_language` | `str` | the detected dominant language of the extracted text |
| `text_english` | `str` | the text translated into English |
| `text_clean` | `str` | the text after cleaning from numbers and unrecognizable words |