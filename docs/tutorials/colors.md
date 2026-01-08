# Color composition analysis

*Tutorial coming soon!*

The color composition of the images can be extracted using the `ColorDetector` class (`colors` module). 

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

The color detection is carried out using the following method call:
```
for key in image_dict.keys():
    image_dict[key] = ammico.ColorDetector(
        image_dict[key],  
    ).analyse_image()

```
This returns a dictionary with color names as keys and their percentage presence in the image as values (rounded to 2 decimal places).