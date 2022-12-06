import imagehash as ih
import numpy as np
import os
import time
import pandas as pd
from PIL import Image
import av
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def str_to_hex(x : str):
    """
    Converts a string in hex format to an integer. 
    Labels with issues are not strings, so they are appened to an error indicator.
    """
    if isinstance(x, str):
        x = int(x, 16)
    else:
        x = "error" + str(x)
    return x

def get_folder_hashes(folder : str, metadata : bool = True) -> pd.DataFrame:
    """
    Scans a folder, getting all files in them (+ subdirectories) and adding them to a dataframe
    Parameters
    ----------
    folder : str
        The filepath to the folder to be scanned
    metadata : boolean
        Boolean indicating if other file properties, like image dimensions, should be added to the output.

    Returns:
        a Pandas dataframe of files in the folder, their metadata, and hashes
    """
    time0 = time.time()
    folders = []
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for file in filenames:
            folders.append(str(os.path.join(dirpath, file)))

    # initialize the list that will be the dataframe
    hash_df = []
    # number to track bad files
    error_count = 1
    for fp in tqdm(folders):
        path = fp.replace(folder, "")
        row = [fp, path]

        try:
            hash = get_hash(fp, metadata = metadata)
            hash_df.append(row + [*hash])

        except (OSError, NotImplementedError) as e:
            print(e)
            row = row + [error_count]
            if metadata: # add empty metadata columns
                row += [np.nan, np.nan]
            hash_df.append(row)
            error_count += 1

    columns = ["full_path", "image", "hash", "extension"]
    if metadata:
        columns += ["height", "width"]
        sort_cols = ["hash", "height"]
    else:
        sort_cols = ["hash"]
    
    hash_df = pd.DataFrame(hash_df, columns = columns)
    hash_df["hash"] = hash_df["hash"].apply(lambda x: str_to_hex(x))

    # sort descending since keep = first is used for duplicates, ie the largest are kept
    hash_df.sort_values(sort_cols, ascending = False, inplace = True)

    hash_df.to_csv("hashes.csv", index = False)
    print(time.time() - time0, " seconds taken")
    return hash_df

def get_hash(filepath : str, hash_function = ih.phash, metadata : bool = True) -> tuple:
    """
    Gets the hash of an image or video at a filepath, as well as other important metadata.

    Parameters
    ----------
    filepath : str
        The filepath to the given image or video.
    hash_function : function
        The function which produces the hash of an image. Should be compatible with PIL images.
    metadata : boolean
        Boolean indicating if other file properties, like image dimensions, should be returned.

    Returns:
        a tuple containing the hash, extension and optionally height and width
    """
    extension = filepath.split(".")[-1]

    if extension.lower() in ["mp4", "avi", "webm", "mov", "m4v"]:
        stream = av.open(filepath)
        # somewhat inefficient but the only way I could find for getting the first frame only in PyAV
        for frame in stream.decode(video = 0):
            frame = frame.to_ndarray(format = "rgb24")
            break
        stream.close()
        # imagehash requires a PIL image, so convert the np array
        im = Image.fromarray(frame)

    elif extension.lower() in ["jpg", "jpeg", "png", "gif", "bmp", "jfif", "webp"]:
        im = Image.open(filepath)

        if extension.lower() == "webp":
            # this accursed file format shall no longer continue its reign of terror
            new_fp = filepath.replace(".webp", ".png") # assuming .webp never occurs elsewhere ;)
            im.save(new_fp)
        
    else:
        raise NotImplementedError(f"Files with extension {extension} can't be loaded...")
    
    hash = str(hash_function(im))

    if metadata:
        h = im.height
        w = im.width

        out = (hash, extension, h, w)
    else:
        out = (hash, extension)
    return out


def move_duplicates(df_hash : pd.DataFrame, out_folder : str, exts_to_delete : list = ["webp"]) -> None:
    """
    Moves duplicates using hash dictionary, saving in a new output folder and retaining the structure of the target folders.
    Parameters
    ----------
    df_hash : pd.DataFrame
        The dataframe returned by get_folder_hashes().
    out_folder : str
        An output folder path string
    exts_to_delete : list
        List of strings of extensions that will get moved to the duplicates folder i.e. removed from the original folder.

    Returns:
        None
    """
    bad_ext = df_hash["extension"].isin(exts_to_delete)
    duplicated_ims = df_hash.loc[(df_hash.duplicated("hash")) | bad_ext, :]

    for in_path, ims in duplicated_ims[["full_path", "image"]].itertuples(index=False, name = None):
        try:
            out_path = out_folder + ims # os join fails here?!?
            out_path = out_path.replace("\\", "/")
            out_folder_i = "/".join(out_path.split("/")[:-1])
            os.makedirs(out_folder_i, exist_ok = True)
            os.rename(in_path, out_path)
        except (FileNotFoundError, PermissionError):
            print(f"{in_path} does not exist")
        

if __name__ == "__main__":
    # example and test for a folder i had
    hash_df = get_folder_hashes(r"C:\Projects\Ghibli Papes")
    move_duplicates(hash_df, out_folder = "C:\Projects\duplicates")