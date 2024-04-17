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
The model uses two fully connected layers with a dropout rate of 0.2 to approximate the inversion operator, transforming measurement values into spatial layouts of absorbers.
#### Convolutional Encoder-Decoder: 
This part detects inclusion attributes via a 3D convolutional network with padding, employing 64 filters initially, followed by max-pooling and transpose convolution operations for accurate depth and spatial dimension mapping.
#### Optimizer:
Adamax was chosen for its robustness against large gradient updates, showing good generalization to new datasets while performing comparably to RMSprop and Adam in terms of training stability and time.








