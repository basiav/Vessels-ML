import zlib
import base64
import numpy as np

COLOR_LABEL_DICT = {
    1: {"number": "1", "name": "RCA proximal", "color": [221, 85, 85, 255]},
    2: {"number": "2", "name": "RCA mid", "color": [78, 228, 122, 255]},
    3: {"number": "3", "name": "RCA distal", "color": [167, 70, 236, 255]},
    4: {"number": "4", "name": "Posterior descending artery", "color": [190, 181, 116, 255]},
    5: {"number": "5", "name": "Left main", "color": [109, 182, 197, 255]},
    6: {"number": "6", "name": "LAD proximal", "color": [204, 102, 157, 255]},
    7: {"number": "7", "name": "LAD mid", "color": [124, 211, 95, 255]},
    8: {"number": "8", "name": "LAD apical", "color": [93, 88, 218, 255]},
    9: {"number": "9", "name": "First diagonal", "color": [225, 129, 81, 255]},
    10: {"number": "9a", "name": "First diagonal", "color": [73, 233, 173, 255]},
    11: {"number": "10", "name": "Second diagonal", "color": [181, 119, 187, 255]},
    12: {"number": "10a", "name": "Second diagonal", "color": [177, 194, 112, 255]},
    13: {"number": "11", "name": "Proximal circumflex artery", "color": [105, 153, 201, 255]},
    14: {"number": "12", "name": "Intermediate/anterolateral artery", "color": [208, 98, 121, 255]},
    15: {"number": "12a", "name": "Obtuse marginal", "color": [91, 215, 101, 255]},
    16: {"number": "12b", "name": "Obtuse marginal", "color": [136, 84, 222, 255]},
    17: {"number": "13", "name": "Distal circumflex artery", "color": [230, 179, 77, 255]},
    18: {"number": "14", "name": "Left posterolateral", "color": [122, 184, 181, 255]},
    19: {"number": "14a", "name": "Left posterolateral", "color": [191, 115, 172, 255]},
    20: {"number": "14b", "name": "Left posterolateral", "color": [149, 198, 108, 255]},
    21: {"number": "15", "name": "Posterior descending", "color": [101, 118, 205, 255]},
    22: {"number": "16", "name": "Posterolateral branch from RCA", "color": [212, 109, 94, 255]},
    23: {"number": "16a", "name": "Posterolateral branch from RCA", "color": [87, 219, 142, 255]},
    24: {"number": "16b", "name": "Posterolateral branch from RCA", "color": [184, 80, 226, 255]},
    25: {"number": "16c", "name": "Posterolateral branch from RCA", "color": [234, 234, 72, 255]},
    26: {"number": "XX", "name": "Plaque", "color": [118, 167, 188, 255]},
    27: {"number": "UNKN", "name": "Vessel", "color": [195, 111, 146, 255]}
}

def unpack_mask(mask, shape=(512, 512)):
    """Unpack segmentation mask sent in HTTP request.

    Args:
        mask (bytes): Packed segmentation mask.

    Returns:
        np.array: Numpy array containing segmentation mask.
    """
    mask = base64.b64decode(mask)
    mask = zlib.decompress(mask)
    mask = list(mask)
    mask = np.array(mask, dtype=np.uint8)
    # pylint:disable=too-many-function-args
    mask = mask.reshape(-1, *shape)
    mask = mask.squeeze()
    return mask
