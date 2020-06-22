# face-treater
 Checks all the pictures in a folder and copies to another folder only the pictures where the face is recognizable enough and applies transformations to keep the eyes and nose centered.


## How to set pictures up for detection and treatment

To load the pictures, the pictures must be inside the folder
                            raw_pictures/
        example:
                            raw_pictures/peter-whatevs.jpg
                            raw_pictures/potato-picture.png
                            raw_pictures/banana.gif
and their filenames must be declared inside a csv file with two
or more columns, including at least "name" and "picture_filename". It will
preserve any other columns you wish to add in the new dataframe.

raw_picture.csv

            name                picture_filename        random_property
        0   Peter's Picture     peter-whatevs.jpg       Orange
        1   BIGGEST POTATO      potato-picture.png      Green
        2   SMALLEST BANANA     banana.gif              Purple

The program will return only valid pictures, with the same filenames,
but treated, inside a folder:
                            treated_pictures/
        example:
                            treated_pictures/peter-whatevs.jpg
                            treated_pictures/banana.gif

and the new csv file will be a copy of the last one, with the difference
that it will preserve only the treated pictures rows

treated_pictures.csv

            name                picture_filename        random_property
        0   Peter's Picture     peter-whatevs.jpg       Orange
        2   SMALLEST BANANA     banana.gif              Purple
