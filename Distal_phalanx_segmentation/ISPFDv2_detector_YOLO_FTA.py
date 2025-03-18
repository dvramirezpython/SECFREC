import cv2
import numpy as np
from constants import CONFIG
from ultralytics import YOLO  
from skimage.exposure import rescale_intensity
from skimage.util import invert
import time
import torch
import torchvision.transforms as transforms
from FingerNet_pytorch.src.modelDeploy import MainNet
from FingerNet_pytorch.src.utils import *
import os

def increase_contrast(img, min_percent=2, max_percent=98):
    v_min, v_max = np.percentile(img, (min_percent, max_percent))
    img = np.clip(img, v_min, v_max)  # Avoid extreme outliers
    better_contrast = rescale_intensity(img, in_range=(v_min, v_max))
    return better_contrast / better_contrast.max()  # Normalize

def process_image(model, img, device):
    return model(source=img,
                 device=device, 
                 save=False,
                 conf=0.1, 
                 show=False,
                 verbose = False,
                 iou = 0.1 # Intersection over union as low as 0.1 because we want to avoid duplicated fingerprints
                           # but, we have to consider close fingers with overlapped bouding boxes
                 )  # Predict on the image without saving or displaying

def resize_with_pad(image, new_shape, padding_color = (0,0,0)):
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_h, original_w = image.shape[:2]
    ratio = min(new_shape[1] / original_h, new_shape[0] / original_w)
    new_size = (int(original_w * ratio), int(original_h * ratio))
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    pad_w = (new_shape[0] - new_size[0]) // 2
    pad_h = (new_shape[1] - new_size[1]) // 2
    return cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=padding_color)

def preprocess_image_4feature(image, m0=0.0, var0=1.0):
    """
    Prepares a grayscale image for input into the MainNet model.
    Args:
        image (numpy array): Grayscale image (H, W).
        m0 (float): Mean for normalization.
        var0 (float): Variance for normalization.
    Returns:
        torch.Tensor: Preprocessed image tensor (1, 1, H, W).
    """
    # Ensure the image is grayscale (H, W)
    if image.ndim == 3 and image.shape[2] == 3:  
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image.shape
    h, w = round(h / 8) * 8, round(w / 8) * 8
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[m0], std=[var0])
    ])
    return transform(image).unsqueeze(0)

def compute_fp_quality(minutiae, mask):
    """
    Computes the quality of a fingerprint image based on the number of minutiae and the orientation field.
    Args:
        minutiae (list): List of minutiae with format (x, y, o).
        orientation_field (ndarray): 2D numpy array of orientation angles in radians.
        mask (ndarray): 2D binary numpy array indicating the valid region of the image.
    Returns:
        float: Quality score in the range [0, 1].
    """
    # Compute the number of minutiae
    num_minutiae = len(minutiae)
    # Compute the normalized orientation field support
    orientation_support = np.sum(mask) / mask.size
    # Compute the quality score
    quality = (min((num_minutiae / 100), 1.0) + orientation_support) / 2
    return quality

def extract_features(fp_image):
    """
    
    """
    with torch.no_grad():
        enh_img_real, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = feature_model(fp_image.to(device).double())
    # transforming torch tensors to numply to reuse fingernet original util functions
    seg_out = seg_out.permute(0,2,3,1).detach().cpu().numpy()
    mnt_o_out = mnt_o_out.permute(0,2,3,1).detach().cpu().numpy()
    mnt_s_out = mnt_s_out.permute(0,2,3,1).detach().cpu().numpy()
    mnt_w_out = mnt_w_out.permute(0,2,3,1).detach().cpu().numpy()
    mnt_h_out = mnt_h_out.permute(0,2,3,1).detach().cpu().numpy()
    enh_img_real = enh_img_real.permute(0,2,3,1).detach().cpu().numpy()
    # post processing segmentation mask
    round_seg = np.round(np.squeeze(seg_out))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    seg_out = cv2.morphologyEx(round_seg, cv2.MORPH_OPEN, kernel)
    # post processing minutia labels
    mnt = label2mnt(np.squeeze(mnt_s_out)*np.round(np.squeeze(seg_out)), mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5)
    mnt_nms = nms(mnt)
    # post processing orientation field
    ori = torch_ori_highest_peak(ori_out_1)
    ori = ((torch.argmax(ori, axis = 1)*2-90)/180.*np.pi).detach().cpu().numpy()
    return ori, mnt_nms, seg_out

def estimate_contrast(img):
    """
    Params: input image in BGR or Grayscale format
    Output: max, min, anc contrast levels
    Estimates the contrast level of the image using the Michelson method
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_val, max_val = np.percentile(img, [5, 95])  # Avoid extreme noise
    contrast = (max_val - min_val) / (max_val + min_val + 1e-5)  # Avoid division by zero
    return min_val, max_val, contrast

def preprocess_fingerprint(box, mask, source_img, resized_img):
    """
    # # Iterate over each detection's bounding box and mask
    # # Compute Image quality for each detected bounding box and keep the highest one
    # # Process the one with the highest quality, assuming is the first one
    """
    x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int).flatten()
    if resized_img is not None:
        # Ensure the coordinates are within the image boundaries
        x1 = int((x1 * source_img.shape[1]) / resized_img.shape[1])
        y1 = int((y1 * source_img.shape[0]) / resized_img.shape[0])
        x2 = int((x2 * source_img.shape[1]) / resized_img.shape[1])
        y2 = int((y2 * source_img.shape[0]) / resized_img.shape[0])
    # Extract the mask for the current detection
    # mask_array = selected_mask.data.cpu().numpy()[0]
    mask_array = mask.data.cpu().numpy()[0]
    # Resize the mask to match the source image size
    mask_resized = cv2.resize(mask_array, 
                              (source_img.shape[1], source_img.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    # Convert the resized mask to a binary mask (0 or 255)
    _, mask_binary = cv2.threshold(mask_resized, 0.5, 255, cv2.THRESH_BINARY)
    # Apply the binary mask to the entire grayscale source image using bitwise AND
    masked_img = cv2.bitwise_and(source_img, 
                                 source_img, 
                                 mask=mask_binary.astype(np.uint8))
    return enhance_fingerprint(masked_img, x1, x2, y1, y2)

def enhance_fingerprint(masked_img, x1, x2, y1, y2):
    """
    Input: masked_img is the whole image
            x1, x2, y1, y1 are the coordinates for obtaining the reduced img
    Output: Image obtained from the bounding box enhanced according to its contrast level
    It uses the Mechelson method to estimate the contrast
    It choose the CLAHE parameters based on the contrast level.
    """
    # Crop the region of the masked image defined by the bounding box
    cropped_img = masked_img[y1:y2, x1:x2]
    resized = resize_with_pad(cropped_img, tuple(CONFIG['image']['fingernet_size']))
        
    # Malhotra method ********
    # Step 1: Convert to grayscale
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # Step 3: Histogram equalization is applied to mitigate the effect of illumination variation.
    hist_equalized = cv2.equalizeHist(gray_img)
    # Tile grid size is the number of cells. We want sizes of 8x8. So, it is 640/8 = 80
    # It is too much, so we will try with the half
    # clipLimits default values is 40
    _, _, contrast = estimate_contrast(hist_equalized)
    # if contrast < 0.2:
    #     cplim = 40.0
    #     grid_size = (80,80)
    #     median_blur = 7
    # elif contrast > 0.8:
    #     cplim = 20.0
    #     grid_size = (16,16)
    #     median_blur = 3        
    # else:
    #     cplim = 10.0
    #     grid_size = (32,32)
    #     median_blur = 5
    # max(1, int(contrast * 5))  # Scales blur based on contrast
    if contrast < 0.2:
        cplim = 30.0
        grid_size = (40,40)
        median_blur = 5
    elif contrast > 0.8:
        cplim = 20.0
        grid_size = (10,10)
        median_blur = 3
    else:
        cplim = 10.0
        grid_size = (20,20)
        median_blur = 1
    # Step 2: Speckle noise is removed using median filtering.
    # median_filtered = cv2.medianBlur(gray_img, median_blur)
    # Gaussian blur to reduce noise
    gaussian_blurred = cv2.GaussianBlur(hist_equalized, (3, 3), sigmaX=2, sigmaY=2)

    clahe = cv2.createCLAHE(clipLimit=cplim, tileGridSize=grid_size)
    hist_equalized = clahe.apply(gaussian_blurred)
    return hist_equalized


# Load the custom variables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = CONFIG['output']['path']
sample_set_array = CONFIG['image']['sample_set_array']
min_fp_quality = CONFIG['image']['min_fp_quality']
max_steps = CONFIG['image']['max_steps']

# Load YOLO model
phalange_seg_model = YOLO(CONFIG['yolo_models']['fingerprint']['path'])
# Load and prepare the FingerNet model
feature_model = MainNet().double().to(device)
feature_model.load_state_dict(torch.load(CONFIG['fingernet']['path'], weights_only=CONFIG['fingernet']['weights_only']))
feature_model.eval()

for sample_set in sample_set_array:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_file = open(f'{output_dir}/FTA_{sample_set}.txt', 'w')
        # Measurements for ISPFDv2 database
    SOURCE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    SOURCE_FOLDER = os.path.join(SOURCE_FOLDER, f'{CONFIG['image']['path_to_database']}/{sample_set}')
    img_filename_list = list(os.scandir(SOURCE_FOLDER))
    results_file.write(f'FTA for the set {sample_set}.\n')
    fp_quality = 0.0
    attempt_dict = {i: 0 for i in range(max_steps)} 
    
    for img_idx, img_filename in enumerate(img_filename_list):
        attempt = 0
        source_img = cv2.imread(img_filename)
        img_name = img_filename.name.split('.')[0]
        sample_splitted = img_name.split('_')
        sample_id = f'{sample_splitted[0]}_{sample_splitted[1]}_{sample_splitted[2]}'

        if source_img is not None:
            if source_img.shape[1] != tuple(CONFIG['image']['fingernet_size'])[0]:
                resized_img = cv2.resize(source_img, tuple(CONFIG['image']['yolo_size']), interpolation=cv2.INTER_CUBIC)
            
            start_time = time.time()
            # Perform prediction using the YOLO model
            results = process_image(phalange_seg_model, resized_img, device)

            # Modify the contrast max_steps times to improve the fingerprint detection
            while ((len(results[0].boxes) == 0) and (attempt < max_steps)):
                # Rotate both resized and source image for a correct posprocessing
                print(f"No detections found at attempt {attempt}")
                results_file.write(f'No detections found at attempt {attempt} for image {img_filename.name}\n')
                attempt_dict[attempt] += 1
                attempt += 1
                resized_img = increase_contrast(resized_img, 2, 98)
                source_img = increase_contrast(source_img, 2, 98)
                if attempt > (max_steps // 2):
                    resized_img = cv2.rotate(resized_img, cv2.ROTATE_90_CLOCKWISE)
                    source_img = cv2.rotate(source_img, cv2.ROTATE_90_CLOCKWISE)
                results = process_image(phalange_seg_model, resized_img, device)

            # No detection at all
            if len(results[0].boxes) == 0:
                print(f'No fingerprint detected on sample {img_filename.name}')
                results_file.write(f'No fingerprint detected on sample {img_filename.name}\n')
            # One or more fingerprint detected
            else:
                # If detected one or more than one bounding
                processed_img = preprocess_fingerprint(results[0].boxes[0], results[0].masks[0], source_img, resized_img)
                # Invert images to make ridges black and valleys white for FingerNet
                processed_img = invert(processed_img)
                fp_image = preprocess_image_4feature(processed_img)
                ori_array, mnt_array, mask_array = extract_features(fp_image)
                fp_quality = compute_fp_quality(mnt_array, mask_array)
                idx = 1
                while (idx < len(results[0].boxes)) and (fp_quality <= min_fp_quality):
                    processed_img = preprocess_fingerprint(results[0].boxes[idx], results[0].masks[idx], 
                                                           source_img, resized_img)
                    # Invert images to make ridges black and valleys white for FingerNet
                    processed_img = invert(processed_img)
                    fp_image = preprocess_image_4feature(processed_img)
                    ori_array, mnt_array, mask_array = extract_features(fp_image)
                    fp_quality = compute_fp_quality(mnt_array, mask_array)
                    idx += 1 

                # Uncomment to save enhanced fingerprints with orientation field and minutiae
                # draw_minutiae(processed_img, mnt_array[:,:3], "%s/%s_mnt.png"%(output_dir, img_name))
                # draw_ori_on_img(processed_img, ori_array, mask_array, "%s/%s_ori.png"%(output_dir, img_name))        
            
                if fp_quality > min_fp_quality:
                    time_taken = time.time() - start_time
                    # Uncomment to print the results in console
                    # print(f'Post processing finished with success for image {img_name} with quality {fp_quality} in {time_taken * 1000} miliseconds')
                    results_file.write(f'Post processing finished with success for image {img_name} with quality {fp_quality} in {time_taken * 1000} miliseconds\n')
                else:
                    attempt = max_steps-1 if attempt > max_steps else attempt
                    attempt_dict[attempt] += 1
                    # Uncomment to print the results in console
                    # print(f'Quality {fp_quality} too low for image {img_name} ')
                    results_file.write(f'Quality {fp_quality} too low for image {img_name}\n')

    # Computing performance metrics
    # Uncomment to print the final results summary in console
    # print(f'Total fingerprints processed: {len(img_filename_list)}')
    results_file.write(f'Total fingerprints processed: {len(img_filename_list)}\n')
    for att in attempt_dict:
        # Uncomment to print the final results summary in console
        # print(f'FTA in attempt {att} = {attempt_dict[att]}')
        results_file.write(f'FTA in attempt {att} = {attempt_dict[att]}\n')

    results_file.close()
    cv2.destroyAllWindows()