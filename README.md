MST-AI: Monk Skin Lesion AI

Author: VK

Date: 2025-08-20

Version: 1.0

Description:

    This script implements the MST-AI pipeline for skin lesion analysis.
    It includes functions to load and process Monk ORB images, extract
    lesions, frames, skin, and inliers, estimate PDFs using Gaussian Mixture
    Models, and compute membership scores based on KL and L1 distances.

 """
  Class to perform all the steps of the MST-AI pipeline.
  Attributes:
      msts_idir: String, directory containing ORB images.
      msts: List of numpy arrays, ORB samples.
      msts_pdfs: List of fitted Gaussian Mixture Models for ORBs.
      transforms: torchvision.transforms.Compose, image transformations.
      model: Loaded pre-trained model for lesion extraction.
  Methods:
      make_transforms: Create image transformations.
      load_model: Load a pre-trained model from a file.
      get_lesion: Extract lesion pixels from the image.
      get_frame: Extract frame pixels from the image.
      get_skin: Extract skin pixels from the image.
      get_inliers: Extract inlier pixels from the image.
      get_monk_pixels: Load and process ORB images.
      get_pdf: Estimate the PDF of a sample using GMM.
      get_pdf_vals: Get PDF values for a range of points in 3D space.
      get_kl_distances: Compute KL distances between ORB PDFs and image PDF.
      get_l1_distances: Compute L1 distances between ORB PDFs and image PDF.
      get_membership_score: Compute membership scores from distances.
  """


You need to copy the model file (not the directory but the file itself) into the model directory.
At the first step, simply run it with
>> python mst_ai.py

If there are import errors, use pip or conda to install the required libraries. You can also check the "import" section.

For a better understanding, I included a “test_ISIC” function that can clarify all the steps.

Please Note:
If the image is from the Kaggle skin color dataset, you need to use the following functions in order:
1. get_lesion
2. get_frame
3. get_skin
4. get_inliers
5. get_monk_pixels
6. get_pdf
7. get_pdf_vals
8. kl_divergence
9. get_kl_distances (or get_l1_distances)
10. get_membership_score

If the image is from another source (for example, may not have a lesion or frame), then:
4. get_inliers
5. get_monk_pixels
6. get_pdf
7. get_pdf_vals
8. kl_divergence
9. get_kl_distances (or get_l1_distances)
10. get_membership_score

I mean, if an image is not from the skin cancer sample (benign or malignant), for example, a human face, then using lesion, frame, and skin extractions are not necessary.
The output of get_membership_score is a list of 10 probabilities.

