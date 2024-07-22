from ultralytics import YOLO


model = YOLO("SUDCN.engine")

results = model(r"E:\data_set\infraredSmallObject\arrangedData_testLabel\val\images", iou=0.5, conf=0.3,save =True)

# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk

#yolo export model=runs/detect/train77/weights/SUDCN.pt  format=engine half=True simplify opset=13 workspace=16
#trtexec --onnx=runs/detect/train99/weights/best.onnx --saveEngine=SUDCN.engine --fp16

#yolo export model=runs/detect/train103/weights/SUDCN.pt  format=engine half=True simplify opset=13 workspace=16


#=============tensorrt ==================

#yolo export model=runs/detect/train99/weights/SUDCN.pt format=engine half=True simplify opset=13 workspace=16




