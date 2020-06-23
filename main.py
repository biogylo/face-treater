# Import picture list
import pandas as pd
import numpy as np

import cv2

from tqdm import tqdm # Progress bar library
import face_treater_config as cfg # Config file with the necessary constants
import face_detector

treated_info = None
raw_info = None
rejected_info = None

print("face-treater/main.py")

print('\nLoading "{0}"'.format(cfg.RAW_INFO_FILENAME))

try:
    raw_info = pd.read_csv(cfg.RAW_INFO_FILENAME)
    print('Loaded "{0}" successfully.'.format(cfg.RAW_INFO_FILENAME))
    print('\tRaw info with {0} entries, and columns = {1}\n'.format(len(raw_info.index),list(raw_info.columns)))
    if cfg.FILENAME_COLUMN not in raw_info.columns:
        print('The column "{0}" is not available in the DataFrame.'.format(cfg.FILENAME_COLUMN) )
        print('Change the column name of the CSV or change the constant "FILENAME_COLUMN"')
        print('in the file "config.py" to the column name with the picture filenames.')
        print('Exiting...')
        exit()
except FileNotFoundError:
    print('The file "{0}" was not found.'.format(cfg.RAW_INFO_FILENAME))
    print('Please create such a file with name equal to: "{0}", or change the constant RAW_INFO_FILENAME in the file config.py.'.format(cfg.RAW_INFO_FILENAME))
    print('Please read the "README.md" file to understand its requirements.')
    print('\nThe program cannot continue, as it is missing that file.')
    print('Exiting...')
    exit()
# Loads just in case there was an interrupted session at some point,
print('\nLoading "{0}"'.format(cfg.REJECTED_INFO_FILENAME))

try:
    rejected_info = pd.read_csv(cfg.REJECTED_INFO_FILENAME)
    print('Loaded "{0}" successfully.'.format(cfg.REJECTED_INFO_FILENAME))
except FileNotFoundError:
    print('\tThe file "{0}" was not found.'.format(cfg.REJECTED_INFO_FILENAME))
    print('\t\tCreating a new rejected info file, "{0}".'.format(cfg.REJECTED_INFO_FILENAME))
    rejected_info = pd.DataFrame(columns = raw_info.columns)
    rejected_info.to_csv(cfg.REJECTED_INFO_FILENAME,index=False)
    print('\t\tCreated "{0}" successfully.'.format(cfg.REJECTED_INFO_FILENAME))

print('\nLoading "{0}"'.format(cfg.TREATED_INFO_FILENAME))

try:
    treated_info = pd.read_csv(cfg.TREATED_INFO_FILENAME)
    print('Loaded "{0}" successfully.'.format(cfg.TREATED_INFO_FILENAME))
except FileNotFoundError:
    print('\tThe file "{0}" was not found.'.format(cfg.TREATED_INFO_FILENAME))
    print('\t\tCreating a new treated info file, "{0}".'.format(cfg.TREATED_INFO_FILENAME))
    treated_info = pd.DataFrame(columns = raw_info.columns)
    treated_info.to_csv(cfg.TREATED_INFO_FILENAME,index=False)
    print('\t\tCreated "{0}" successfully.'.format(cfg.TREATED_INFO_FILENAME))


# Do all filenames in the raw_info not in the treated_info
#pending_indexes = [raw_row in treated_info[cfg.FILENAME_COLUMN for raw_row in raw_info[cfg.FILENAME_COLUMN].iterrows()]
pending_indexes = np.logical_not(np.array(raw_info[cfg.FILENAME_COLUMN].isin(treated_info[cfg.FILENAME_COLUMN].values)))
pending_pictures = raw_info[pending_indexes]

for picture_row in tqdm(pending_pictures.to_dict('records')):
    picture_copy = picture_row.copy()
    picture_path = cfg.PICTURE_FOLDER + picture_copy[cfg.FILENAME_COLUMN]
    try:
        current_picture = cv2.imread(picture_path)
    except FileNotFoundError:
        print('FileNotFoundError: The file in {0} does not exist, skipping.'.format(picture_path))
        continue

    blurryness = face_detector.get_blurryness(current_picture)

    if blurryness <= cfg.BLUR_TRESHOLD:
        picture_copy['rejected'] = "Too blurry. Blurryness = {0}".format(blurryness)
        rejected_info = rejected_info.append(picture_copy,ignore_index=True)
        rejected_info.to_csv(cfg.REJECTED_INFO_FILENAME, index=False)
        continue

    landmarks = face_detector.get_landmarks()

    cv2.imshow(picture_path,current_picture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    treated_info = treated_info.append(picture_copy,ignore_index=True)
    treated_info.to_csv(cfg.TREATED_INFO_FILENAME,index=False)

print("\nDone.")
