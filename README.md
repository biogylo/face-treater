# face-treater
 Checks all the pictures in a folder and copies to another folder only the pictures where the face is recognizable enough and applies transformations to keep the eyes and nose centered.


## How to set pictures up for detection and treatment

To load the pictures, the pictures must be inside the folder
```
raw_pictures/
```
example:
```
raw_pictures/peter-whatevs.jpg
raw_pictures/potato-picture.png
raw_pictures/banana.gif
```
and their filenames must be declared inside a csv file with two or more columns, including at least the column "picture_filename", which by the way, should include the picture filename. It will preserve any other columns you wish to add in the new dataframe.

**Filename:** raw_picture_info.csv

|index|name|picture_filename|random_property|
|-----|----|----------------|---------------|
|0|Peter's Picture|peter-whatevs.jpg|Orange|
|1|BIGGEST POTATO|potato-picture.png|Green|
|2|SMALLEST BANANA|banana.gif|Purple|

The program will return only valid pictures, with the same filenames, but treated, inside a folder called treated_pictures:
```
treated_pictures/
```
example:
```
treated_pictures/peter-whatevs.jpg
treated_pictures/banana.gif
```

and the new csv file will be a copy of the last one, with the difference that it will preserve only the treated pictures rows

**Filename:** 'treated_picture_info.csv'

|index|name|picture_filename|random_property|
|-----|----|----------------|---------------|
|0|Peter's Picture|peter-whatevs.jpg|Orange|
|2|SMALLEST BANANA|banana.gif|Purple|

## Untreatable pictures
Blurry pictures, pictures where the subject has glasses, pictures with faces looking to another direction, or unrecognizable pictures will not be added to the treated picture folder and database, instead, an entry will be added to another csv, showing the same columns, but it will declare the reason it was rejected in a new column, called rejected. The csv with rejected picture will be caled rejected_picture_info.csv.

**Filename:** 'rejected_picture_info.csv'

|index|name|picture_filename|random_property|rejected|
|-----|----|----------------|---------------|--------|
|1|BIGGEST POTATO|potato-picture.png|Green|BlurryPicture|
