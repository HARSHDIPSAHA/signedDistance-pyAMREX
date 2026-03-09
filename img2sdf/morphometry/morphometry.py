"""Morphometric shape analysis per segmented phase.

Adapted from uSCMAN Modules/Analysis/Morphometry/Morphometry.py.
Logic is unchanged; imports updated for img2sdf package structure.
"""
from __future__ import annotations
import numpy as np
try:
    import cv2 
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from skimage.measure import label, regionprops, regionprops_table
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# pixel_value → phase name mapping (mirrors uSCMAN)
_FEATURE_MAP = {0: "defect", 128: "crystal", 255: "binder"}

def _computeScale(clusteredImg, Inputs):
    scale = Inputs["Image Properties"]["pixel scale"]
    DX = [scale, scale]
    if scale == 0:
        row, col = np.shape(clusteredImg)
        physical_width = Inputs["Image Properties"]["image width"]
        physical_height = Inputs["Image Properties"]["image height"]
        dx = float(physical_width) / float(col)
        dy = float(physical_height) / float(row)
        DX = [dx, dy]
    return DX

def _createDefectArray(mask: np.ndarray, prop, labels):
    defect_dict = {}
    for props in prop:
        # Get feature number
        idx = props.label

        # Store defect in matrix
        row_length = (props.slice[0].stop - props.slice[0].start) + 1
        col_length = (props.slice[1].stop - props.slice[1].start) + 1
        pad_row = row_length // 4
        pad_col = col_length // 4
        if pad_row < 5:
            pad_row = 5
        if pad_col < 5:
            pad_col = 5

        # Find indices of defect
        defect_mask = mask[props.slice].copy()
        defect_mask = defect_mask.astype(np.uint8)
        defect_mask[defect_mask==0] = 255
        defect_mask[defect_mask==1] = 0

        # Find where labels are not equal if multiple defects
        False_array_index = np.where(labels[props.slice] != idx)
        defect_mask[False_array_index] = 255

        # Pad
        defect_mask = np.pad(defect_mask, ((pad_row, pad_row),(pad_col, pad_col)), 'constant', constant_values=(255,255))

        # Save dictionary
        defect_dict[f'Defect{idx}'] = defect_mask

    return defect_dict


def _get_feature_data(ImgName: str, featureName: str, dictionary: dict, percent: float, scale: list):
    # Get the scale
    dx, dy = scale

    # Update percent
    dictionary['percent'] = percent

    # Update the perimeter and major/minor axis length if 0
    dictionary['perimeter_crofton'][dictionary['perimeter_crofton']==0] = 1
    dictionary['axis_major_length'][dictionary['axis_major_length']== 0] = 1
    dictionary['axis_minor_length'][dictionary['axis_minor_length']== 0] = 1

    # Update area and diamater for physical scale
    dictionary['area'] = dictionary['area'] * dx * dy
    dictionary['equivalent_diameter_area'] = dictionary['equivalent_diameter_area'] * min(dx,dy)
    dictionary['perimeter_crofton'] = dictionary['perimeter_crofton'] * dx

    # Compute the circularity
    circularity = np.sqrt(4*np.pi*dictionary['area'] / dictionary['perimeter_crofton'])
    dictionary['circularity'] = circularity
    dictionary['circularity'][dictionary['circularity'] > 1] = 1
    dictionary['circularity'][dictionary['circularity'] < 0] = 0

    # Get the ellipse properties
    dictionary['orientation'] = dictionary['orientation'] * 180 / np.pi + 90
    dictionary['AR'] = dictionary['axis_major_length'] / dictionary['axis_minor_length']

    # Get bounding box properties
    AR = []
    for i, coord in enumerate(dictionary['coords']):
        min_rect = cv2.minAreaRect(coord)
        center, size, orientation = min_rect

        # Identify major and minor axis length
        maj_len, min_len = np.max(size)/2.0, np.min(size)/2.0

        # Specify the minimum axis lengths
        if maj_len == 0:
            maj_len = 1.0
        if min_len == 0:
            min_len = 1.0

        # Calculate the aspect ratio of the bounding rectangles
        rect_AR = maj_len / min_len
        AR.append(rect_AR)   
    dictionary['bbox_AR'] = AR


    # Delete unnecessary keys
    keys_to_delete = ['axis_major_length',
                        'axis_minor_length',
                        'coords',
                        'perimeter_crofton'
    ]
    for key in keys_to_delete:
        del dictionary[key]
        
    # Reorder dictionary
    key_order = [
                'percent',
                'area', 
                'equivalent_diameter_area', 
                'bbox_AR',
                'orientation',
                'AR',
                'circularity',
                'solidity'
    ]
    dictionary = {k: dictionary[k] for k in key_order}
 
    return dictionary



def computeMorphometry(image_dict: dict, Inputs: dict, clusteredImg: np.ndarray, morph_flag, defect_flag):
    # Get the image name
    ImgName = image_dict['name']
    base = image_dict['base material']

    # Compute scale
    scale = _computeScale(clusteredImg, Inputs)

    # Determine the features based on the number of pixel values of the image
    unique_values = np.unique(clusteredImg)
    min_val = np.amin(unique_values)
    max_val = np.amax(unique_values)

    # Create a feature map dictionary to loop through to generate masks for analysis
    if len(unique_values) == 2:
        feature_map = {min_val: 'defect', max_val: 'crystal'}
    elif len(unique_values) == 3:
        middle_val = unique_values[1]
        feature_map = {min_val: 'defect', middle_val: 'crystal', max_val: 'binder'}

    # Define the properties necessary for analysis
    properties = ('area',
                    'axis_major_length',
                    'axis_minor_length',
                    'coords',
                    'equivalent_diameter_area',
                    'orientation',
                    'perimeter_crofton',
                    'solidity'
    )

    # Define the dictionary to store all stats and defect images in
    morph_dict = {}
    defect_dict = {}

    # Loop through each of the regions and analyze
    for pixel_value, name in feature_map.items():
        # Create the mask and determine feature percent
        mask = clusteredImg == pixel_value
        total = mask.size
        percent = np.sum(mask) / total * 100

        # Cycle through other features if not defect and not computing morphometry
        if not morph_flag:
            if name != 'defect':
                continue

        # Get the region props dictionary
        label_array = label(mask)
        dictionary = regionprops_table(label_array, properties=properties)

        # Extract defect
        if defect_flag and name == 'defect':
            # Get the properties 
            props = regionprops(label_array)

            # Write defects
            defect_dict = _createDefectArray(mask, props, label_array)     

        # If computing morphometry, create and save dictionary
        if morph_flag:
            stat_dict = _get_feature_data(ImgName, name, dictionary, percent, scale)
            morph_dict[name] = stat_dict
            morph_dict[name]['units'] = Inputs['Morphometry']['units']
            if Inputs["Morphometry"]["plot regions"]:
                morph_dict[name]['mask'] = mask
                morph_dict[name]['labels'] = label_array
                morph_dict[name]['scale'] = scale

    return morph_dict, defect_dict