This is a **Cross-Modal Multi-Organ Registration** system. The system is developed using the Python programming language, with the interface created using PyQt5, and the deep learning algorithms implemented using PyTorch.

The main functions implemented by the system are as follows:
- Normalize and standardize the uploaded images, and process the uploaded labels.
- Complete the registration of multiple organs, and output the registered images as well as the deformation fields.
- utput and compare the label images of the reference, pre-registration, and post-registration images, and provide the Dice, Mutual Information (MI), and Average Symmetric Surface Distance (ASD) metrics after registration.
