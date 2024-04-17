# Learning the effects of undersampling in image reconstruction using Deep Learning Frequency Domain - Diffuse Optical Tomography
## Background
Near Infrared (NIR) light is being extensively researched as an alternative source for medical imaging due to its non-ionizing nature and ability to penetrate deep into tissues. The wavelength range for NIR typically spans from about 700 nm to 2500 nm, with the 700-900 nm window being particularly relevant for medical imaging because it optimally balances tissue penetration depth with minimal absorption by water and hemoglobin.

NIR technology is utilized in various medical fields, including functional brain imaging, muscle oxygenation studies, and cancer detection, showcasing its versatility. In terms of imaging techniques, NIR can be utilized in two different ways: either by sending a continuous wave (CW) of NIR light into the tissue or by employing frequency-modulated imaging, where the frequency of the NIR light alternates between two or three frequencies. Empirically, the frequency-modulated approach tends to yield higher quality images due to its ability to provide detailed information about tissue properties and depth, albeit at a higher cost compared to CW-NIR imaging.

The mechanism behind NIR imaging involves strategically placing a series of light sources and detectors on the part of the human body being scanned. When NIR light is emitted into the tissues, it interacts with the tissues and is either absorbed or scattered. The scattered light is then captured by the detectors. Two key aspects of the detected light are analyzed: the intensity of the light and the phase shift (i.e., this can cause both a delay in time (time lag) and a shift in the phase of the light wave relative to the original light wave). These characteristics are crucial as they help differentiate between various types of tissues based on their optical properties, including differences that allow for imaging functional aspects of the tissue, such as blood flow or oxygenation levels. With this information, it is possible to reconstruct an image of the underlying tissues, providing valuable insights for medical diagnosis and treatment planning.

The imaging technique used in this project is Diffuse Optical Tomogrpahy(DOT) which consists of lights sources (usually lasers or LEDs that emit near infrared light) and light detectors (like photodiodes or CCD cameras). These are placed in contact with or close to the skin surface around the area of interes and the design of the DOT system used in this project is frequency domain (i.e., frequency modulated NIR is used).  Light is emitted into the tissue from multiple source points on the surface sequencially. As the near-infrared light propagates through the tissue, it gets scattered and absorbed. Detectors placed on the surface capture the light that eventually exits the tissue. The intensity of the detected light and the phase shift of the light are measured. This data is influenced by the optical properties of the tissue (such as absorption and scattering coefficients), providing indirect information about the tissue's internal structure and composition. In a traditional set-up, using models of light propagation through tissue, the detected signals are analyzed to infer the distribution of optical properties within the tissue. This step involves solving an inverse problem, which is computationally intensive and requires sophisticated algorithms. The outcome is a spatial map of optical properties, which can be related to the tissue's structure, composition, and, in some cases, functional characteristics (like blood oxygenation levels). In recent times the same is being attempted to acheive using deep learning models, especially based on convolutional neural networks (CNNs) and other advanced architectures, offering promising approaches to tackle the challenge of the inverse problem. These models attempt to directly map the measured data (light intensities and phase shifts) to the optical properties or the internal structures of interest. 

## About the project
The objective of this project is to explore the application of undersampling methods in refining DL models for enhanced image reconstruction. This initiative employs datasets comprising intensity and phase shift information, derived from simulating the FD-DOT technique on a breast phantom embedded with anomalies. The complete simulated dataset is utilized to train a CNN model. The performance of this model is then evaluated in comparison to that achieved when the model is trained on various subsets of undersampled data. Hypothesis to be tested:
#### Hypothesis 1: 
Integrating phase and amplitude data significantly improves image reconstruction quality by minimizing localization errors, compared to using amplitude data alone.
#### Hypothesis 2:
Although training takes longer, high-resolution model training markedly enhances reconstruction results.
#### Hypothesis 3: 
It's both possible and beneficial to strike a balance between shorter data collection durations and maintaining high-quality image reconstruction.

This study explores spatially sparse sampling, an undersampling technique in imaging that targets specific regions for measurement to enhance efficiency and reduce data volume, which can accelerate the acquisition process and make DOT systems more portable. Additionally, the research investigates the impact of using either amplitude or phase measurements alone on the quality of image reconstruction, aiming to understand their individual contributions to capturing optical properties. Finally, the experiment assesses the effects of reduced input measurements on the reconstruction of images at smaller output resolutions, exploring the balance between data collection efficiency and image quality.

## Model: FDNet
### Model Selection:
This model modifies the architectures developed by Yoo et al. and Deng et al. (FDU-Net), particularly by omitting the U-Net component while retaining the fully connected and encoder-decoder sections, resulting in sharper image reconstructions. The aim isn't to compare different models but to evaluate the effect of undersampling on image reconstruction quality.

### Model Description:

#### Loss Function: 
A weighted loss function is employed, focusing on regions of interest (ROIs) to better capture minor perturbations, enhancing the detection of small contrasts within the images.
#### Fully Connected Layers: 
The model uses two fully connected layers and they include an input layer and an intermediary hidden layer, both utilizing a hyperbolic tangent (tanh) activation function. A dropout rate of 0.2 is applied to reduce overfitting. The output of these layers is reshaped to match the dimensions of the simulated target data, preparing it for further convolutional processing.
#### Convolutional Encoder-Decoder: 
##### Initial Processing: 
The reshaped output from fully connected layers enters the convolutional encoder-decoder network, which uses padding to preserve spatial dimensions from input to output.
##### Convolutional Layers: 
Begins with three convolutional layers, each equipped with 64 filters of size 5x5x5, applying a stride of 1. A ReLU activation function is used to ensure outputs remain non-negative, representing optical properties accurately.
##### Max-Pooling Operation: 
A 2x2x2 max-pooling step follows the first convolutional layer, halving the data size in each dimension, which acts as an effective downsampling method to concentrate the features.
##### Transpose Convolution: 
After the second convolution layer, a transpose convolution with a 2x2x2 kernel and a stride of 2 is employed to up-sample the spatial dimensions, countering the earlier downsampling effect.
##### Final Layer Configuration: 
The network concludes with a layer utilizing 2 filters of size 1x1x1 to adjust the depth of the feature map to align with the target data dimensions, without altering the spatial size.
##### Detailed 3D Volume Reconstruction: 
This setup ensures a precise mapping from the high-dimensional feature space back to the spatial domain, which is crucial for accurately reconstructing and interpreting the detailed 3D volume reflective of the analyzed attributes.
#### Optimizer:
Adamax was chosen for its robustness against large gradient updates, showing good generalization to new datasets while performing comparably to RMSprop and Adam in terms of training stability and time.

### Training
The model used 9,000 training examples and 1,000 for validation, with training capped at 1,000 epochs, batch size of 64, and early stopping triggered if validation loss didn't improve for 60 epochs. Early stopping, which also saves the best model version, prevents overfitting and conserves computational resources. The fully connected layer employs a 0.2 dropout rate, deactivating 20% of neurons to promote a more distributed learning and robustness, with a learning rate of 0.001. 
Computationally, the system relied on the NVIDIA Tesla P100 used in Kaggle Kernels with
the following specifications:
CUDA Cores: 3,584
Base Clock: Approximately 1.3 GHz
Memory: 16 GB
Memory Bandwidth: Up to 732 GB/s
Compute Capability: 6.0
FP32 (Single Precision) Performance: Up to 9.3 TFLOPs
FP64 (Double Precision) Performance: Up to 4.7 TFLOPs
NVLink: Supports NVIDIA's NVLink
The time taken for training was 5 to 10 hours, varying by input and output resolution. Future training on V100 or A100 GPUs could reduce time significantly. 3D results visualization utilized Plotly's graphic_objects module.

### Observations
Training with z-score standardization on datasets of log-transformed amplitude and phase measurements showed superior results compared to min-max standardization, establishing it as the standard method for all subsampling strategies in this study. The models were trained across various configurations with consistent preprocessing steps:

**Model 1:** Input size 288, Amp+Phase, Output 32x32x16, on NVIDIA Tesla P100 16GB.

**Model 2:** Input size 288, Amp+Phase, Output 16x16x8, on NVIDIA Tesla P100 16GB.

**Model 3:** Input size 288, Amp+Phase, Output 8x8x4, on NVIDIA Tesla P100 16GB.

**Model 4:** Input size 1152, Amp+Phase, Output 32x32x16, on NVIDIA Tesla P100 16GB.

**Model 5:** Input size 1152, Amp+Phase, Output 16x16x8, on NVIDIA Tesla P100 16GB.

**Model 6:** Input size 1152, Amp+Phase, Output 8x8x4, on NVIDIA Tesla P100 16GB.

**Model 7:** Input size 576, Amp, Output 32x32x16, on NVIDIA Tesla P100 16GB.

**Model 8:** Input size 576, Amp, Output 16x16x8, on NVIDIA Tesla P100 16GB.

**Model 9:** Input size 576, Phase, Output 32x32x16, on NVIDIA Tesla P100 16GB.

**Model 10:** Input size 576, Phase, Output 16x16x8, on NVIDIA Tesla P100 16GB.

Models targeting higher output resolutions displayed better performance, attributed to the increased ability to capture finer details, leading to a focus on the 32x32x16 resolution for further discussions on the effects of using amplitude-only, phase-only data, and undersampling strategies.

### Model Evaluation
The performance metrics adapted from White & Culver 2010 and Doulgerakis 2019 for evaluating the reconstructions include:

**Localisation Error (LOCA):** Measures the linear distance between the centroids of predicted and actual lesions, assessing the spatial accuracy of lesion detection.

**Full Width at Half Maximum (FWHM):** Indicates the spatial dispersion of the lesion's reconstruction by measuring the maximum distance between points at half of the reconstruction's peak value, providing insight into the spatial resolution.

**Effective Resolution (ERES):** Calculates twice the maximum distance between the ground truth centroid and any point within the predicted ROI at or above half of the peak intensity, evaluating the spatial distribution of the predicted lesion.

**Structural Similarity Index (SSIM):** Assesses similarity between two images in terms of luminance, contrast, and structure, where a high SSIM score indicates similar brightness, contrast, and structural information between the compared images.








