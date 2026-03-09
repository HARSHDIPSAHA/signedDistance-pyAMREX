"""Image preprocessing: bilateral filter, Otsu threshold, morphological ops.

Adapted from uSCMAN Modules/Analysis/Preprocessing/Preprocess.py.
All logic is identical; imports are updated for the img2sdf package structure.
"""
from __future__ import annotations
import numpy as np
import os
try:
    from sklearn.metrics import root_mean_squared_error
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import cv2 
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    from skimage.measure import label, regionprops_table
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False

from ..morphometry.morphometry import _computeScale


# ---------------------------------------------------------------------------
# Private helpers (directly from uSCMAN — unchanged)
# ---------------------------------------------------------------------------
def loadImage(image_obj, Inputs):
    # Load grayscale image (2D)
    if not _HAS_CV2:
        raise ImportError("opencv-python is required for preprocessing. "
                          "Install it with: pip install opencv-python")
    gray = cv2.imread(image_obj['path'],0)

    # Scale if max grid length is above zero
    max_length = Inputs["Preprocessing Properties"]["max grid length"]
    if max_length > 0:
        # Get amount of pixels along each dimension
        row, col = np.shape(gray)

        # Determine the maximum number of pixels along the 2 dimensions
        max_dimension = max(row, col)

        # Ensure the max grid size is less than or equal to maximum number of pixels along dimension
        if max_length < max_dimension:
	    # Get the scaling factor
            scale_factor = float(max_length) / float(max_dimension)

            # Scale
            gray = cv2.resize(gray, (0,0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

            # Update image properties in input file when computing scales
            row_scaled, col_scaled = np.shape(gray)
            pixel_scale = Inputs["Image Properties"]["pixel scale"]
            if pixel_scale > 0:
                physical_length = pixel_scale * row
                pixel_scale = physical_length / row_scaled
                Inputs["Image Properties"]["pixel scale"] = pixel_scale

    return gray

def _computeWeightedMean(image, a, b, T1, T2, T):
    # Compute subintervals R1 and R2
    R1 = np.logical_and(image >= a, image <= T1)
    if np.any(R1):
        # Compute means
        mean1 = np.mean(image[R1])
        # Assign mean threshold values to subranges
        image[R1] = mean1
        # Append threshold values
        T.append(mean1)
    else:
        return
        
    R2 = np.logical_and(image >= T2, image <= b)
    if np.any(R2):
        # Compute means
        mean2 = np.mean(image[R2])
        # Assign mean threshold values to subranges
        image[R2] = mean2
        # Append threshold values
        T.append(mean2)
    else:
        return


def _computePSNR(image, thresh):
    # Assert images are same size
    assert np.shape(image) == np.shape(thresh)
    
    # Compute the RMSE
    RMSE = root_mean_squared_error(image, thresh)
    
    # Compute PSNR
    PSNR = 20 * np.log10(255.0 / RMSE) 
    
    return PSNR


def _MultiLevelOtsu(image, astart=0, bstart=255, k1=1.0, k2=1.0, tol=0.1):
    # Define parameters
    n = 2
    difPSNR = 100
    PSNR_old = 0
    
    
    # While loop
    while difPSNR > tol and n <= 20:
        a = astart
        b = bstart
        T = []
        thresh = image.copy() 
        
        # Repeat steps 2-6 n//2 - 1 times
        repeat = n//2 - 1
        for i in range(repeat):
            # Step 2: compute range R
            R = np.logical_and(thresh >= a, thresh <= b)
            
            # Step 3: Compute mean and std
            mean = np.mean(thresh[R])
            sigma = np.std(thresh[R])
            
            # Step 4: Compute subranges
            T1 = mean - k1*sigma
            T2 = mean + k2*sigma
            
            # Step 5: Compute weighted means
            _computeWeightedMean(thresh, a, b, T1, T2, T)
            
            # Step 6: Update a/b
            a = T1 + 1
            b = T2 - 1

            if a > b:
                break
            
        # Repeat step 5 with T1 = mean and T2 = mean+1
        R = np.logical_and(thresh >= min(a,b), thresh <= max(a,b))
        if np.any(R):
            mean = np.mean(thresh[R])
            _computeWeightedMean(thresh, min(a,b), max(a,b), mean, mean+1, T)
        
            # Compute PSNR and compare to old
            PSNR = _computePSNR(image, thresh)
            difPSNR = abs(PSNR - PSNR_old)
            PSNR_old = PSNR
        else:
            difPSNR = 0
        
        # Sort thresholds
        T = np.unique(np.sort(T)).astype(np.uint8)
           
        # Increase iteration count
        n += 2
        
    return (T, thresh)
    


def _convertHEDS(image):
    # Defect pixels have value of 255 and binder pixels are 0
    # Want to switch to maintain consistency
    defect = image == 255
    binder = image == 0
    image[defect] = 0
    image[binder] = 255
    return image


def _RemoveSmallObjects(mask: bool, dx: float, minval=0.10):
    # Create label array
    label_array = label(mask)
    
    # Get region property table
    props = regionprops_table(
        label_array,
        properties=('axis_major_length', 
                    'axis_minor_length', 
                    'equivalent_diameter_area',
                    'feret_diameter_max',
                    'label'
                    ),
    )
    
    # Pull array values
    major = props['axis_major_length'] * dx
    minor = props['axis_minor_length'] * dx
    diameter = props['equivalent_diameter_area'] * dx
    feret = props['feret_diameter_max'] * dx
    labelNumber = props['label']
    
    # Compare to minval ad create bool of values to remove
    major_idx = major < minval
    minor_idx = minor < minval
    diameter_idx = diameter < minval
    feret_idx = feret < minval
    idx_to_remove = np.logical_or(major_idx, minor_idx, diameter_idx)
    idx_to_remove = np.logical_or(idx_to_remove,feret_idx)
    labels_to_remove = labelNumber[idx_to_remove]
    
    # Remove label numbers from mask
    mask_to_remove = np.isin(label_array,labels_to_remove,kind='table')
    mask[mask_to_remove] = False
        
    return mask


def _IdentifyRegion(mask, pixelval, dx, clustered, Inputs):
    # Identify feature
    if pixelval==0:
        feature = 'defects'
    else:
        feature = 'binder regions'
    
    if Inputs["Preprocessing Properties"]["Perform morphological operations"]:
        # Define kernel for morphological operations
        size = Inputs["Preprocessing Properties"]["kernel size"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    
        # Perform morphological operations to smooth image
        itr = Inputs["Preprocessing Properties"]["iterations"]
        mask = cv2.dilate(mask,kernel,iterations = itr)
        mask = cv2.erode(mask,kernel,iterations = itr)
    
    # Remove small defects
    mask = mask == 1 # Boolean
    
    print(f'Removing small {feature}')
    mask = _RemoveSmallObjects(mask, dx)
    
    # Identify damage in clustered array
    clustered[mask] = pixelval
    
    return mask, clustered



# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def preprocessImage(image_obj, Inputs):
    # Get image properties
    name = image_obj['name']

    # Load image
    gray = loadImage(image_obj, Inputs)
    clustered = gray.copy() # Clustered array with 2 or 3 pixel values for regions

    # Get dx for _RemoveSmallObjects function
    DX = _computeScale(gray, Inputs)
    ddx, ddy = DX
    dx = min(ddx,ddy)

    # Filter and blur grayscale image using bilateral filter
    sigma = Inputs["Preprocessing Properties"]["sigma values"]
    filter_diameter = Inputs["Preprocessing Properties"]["filter diameter"]
    blurred = cv2.bilateralFilter(gray,filter_diameter,sigma,sigma)

    # Identify defect regions
    print(f'Identifying defects for {name}')
    _, defect = cv2.threshold(blurred,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    defect, clustered = _IdentifyRegion(defect, 0, dx, clustered, Inputs)
    print(f'Identified defects for {name}')

    # If binary image, label crystal in clustered array
    method = Inputs["Image Properties"]["method"]
    if method == 'binary':
        print(f'Identifying crystals for {name}')
        crystal = np.logical_not(defect)
        clustered[crystal] = 128
        print(f'Identified crystals for {name}')
    else: # Multiphase
        # Threshold to find binder
        # Identify non-binder regions and get mean pixel value for algorithm
        print(f'Identifying binder regions for {name} ')
        nondefect = np.logical_not(defect)
        mean = np.mean(blurred[nondefect])

        # Run multilevel thresholding
        #print('Multilevel Otsu thresholding for ',image.name)
        k1 = Inputs["Preprocessing Properties"]["k1"]
        k2 = Inputs["Preprocessing Properties"]["k2"]
        tol = Inputs["Preprocessing Properties"]["PSNR tolerance"]
        T, _ = _MultiLevelOtsu(blurred,astart=mean,k1=k1,k2=k2,tol=tol)
        print(f'Completed multilevel Otsu thresholding for {name}')

        # Identify binder
        binder = blurred >= T[-2]
        clustered[binder] = 255
        print(f'Identified binder regions for {name}')
        
        # Identify crystal
        print(f'Identifying crystals for {name}')
        crystal = np.logical_not(np.logical_or(defect,binder))
        clustered[crystal] = 128
        print(f'Identified crystals for {name}')

    # Switch values if image is generated via HEDS
    switch_values = Inputs["Preprocessing Properties"]["Threshold HEDS"]
    if switch_values:
        clustered = _convertHEDS(clustered)

    return clustered, gray, Inputs