This dataset encompasses a comprehensive collection of data from a biomechanical tactile sensor array, systematically gathered using a pneumatic device. The dataset is uniquely characterized by its inclusion of static, dynamic, and fatigue testing conditions, providing a multifaceted view of sensor performance and behavior under varied stimuli.

The data is saved in Google's cloud storage. Please click the following link to download the complete dataset: https://drive.google.com/drive/folders/1guMXFfzL7g3ctiylD6ZpBfzZCMRVt9WI?usp=share_link

The tactile sensors, based on capacitive multi-channel pressure sensor technology, offer high-resolution data capturing subtle interactions between the biomedical device and its environment. This feature is critical for advancing research in fields such as prosthetics, robotic surgery, and tactile diagnostics, where nuanced pressure information is paramount.

The dynamic portion captures the sensor's performance under changing conditions, emulating real-world biomedical applications. The fatigue data, collected over repeated cycles of stress, provides insight into the sensor's durability and long-term stability.

By employing a pneumatic device for data collection, the process ensures consistent, controlled, and repeatable pressure application, which is essential for the reliability of tactile sensing data. 

To augment the utility and accessibility of this dataset, accompanying source code has been provided to facilitate the data analysis process. The code includes comprehensive scripts for preprocessing the raw sensor data, ensuring that researchers can readily calibrate, clean, and normalize the data as per their analytical requirements.

The programs in the root directory are for preprocessing the dataset and can be used as needed. The function names correspond to the file names. For example, reshapeArr represents the program for reshaping arrays. The check65535 program is used to exclude potential outliers from sensor data because the sensor's output capacitance value peaks at 2 to the power of 16 (which is 65535).
