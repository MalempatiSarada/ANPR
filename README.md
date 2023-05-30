# ANPR
- This project uses the concept of transfer learning to detect number plates and a cnn model to read the plate.
- This uses pretrained model ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.
- For training the object detection model we use the dataset: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
- For training the model to read from the number plate it uses dataset: https://www.kaggle.com/datasets/nainikagaur/dataset-characters
- After training the weights, models are stored and uploaded. Download the files and run the file. 
- For Executing the file replace the paths in the Anprsys.ipynb.
- To run effeciently it requires Tensorflow, Keras, OpenCV, EasyOCR. Additionally If you have CUDA the model performs even better.
