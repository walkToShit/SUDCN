from ultralytics import YOLO

# Load a model

model = YOLO("SUDCN.engine") #last  best


if __name__ == '__main__':
    #validation_results = model.val( imgsz=640, batch=16, conf=0.25, iou=0.6, device="0") # r"E:\data_set\visDroneYOLO\data.yaml"  , conf=0.25
    model.val( data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml")