# streamlit-example
YOLO model deployment using Streamlit

☕ Read code explanations on medium blog [here](https://medium.com/@ikhsanadi300/how-to-deploy-yolo-detection-and-segmentation-model-on-streamlit-cloud-545167733188).

This project use official YOLO pretrained weights from COCO dataset, change pretrained weights with your created model based on your dataset.

Tested on:
- yolov7 detection ✅
- yolov7 segmentation ✅
  
Future test (Should be ok because it's use same approach):
- yolov5 detection ☑️
- yolov5 segmentation ☑️
- yolov8 detection ☑️
- yolov8 segmentation ☑️

⚠️ Just use what you need!  
- Object detection: `app.py` and `/runs/detect/yolov7.pt`  
- Object segmentation: `app-segment.py` and `/runs/sement/yolov7-seg.pt`

*delete unused files and pretrained model before uploading source folder to your github repo
  
⚠️ YOLO versions other than 5 (Ultralytics), 7 (WongKinLiu), 8 (Ultralytics) or those using a different approach require additional changes. 
