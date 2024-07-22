from ultralytics import YOLO

# Load a model
#mybackbone_denseConcat.yaml
#mybackbonen.yaml
#ultralytics/cfg/models/v8/yolov8.yaml
#mybackbone_denseConcat.yaml
#mybackbone_denseConcat_PFP.yaml   runs/detect/train9/weights/SUDCN.pt
#mybackbone_denseConcat_PFP_lightDetect.yaml
#mybackbone_denseConcat_PFP_lightDetectWithAttention.yaml
#mybackbone_denseConcat_PFP_lightDetect_dySample.yaml
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
#model = YOLO("runs/detect/train9/weights/SUDCN.pt")  # load a pretrained model (recommended for training)


# Use the model
#E:\data_set\infraredSmallObject\arrangedData
#E:\data_set\infraredSmallObject\arrangedData_testLabel
#E:\data_set\infraredSmallObject\arrangeData_withemptylabels
#model.train(data=r"E:\data_set\infraredSmallObject\arrangeData_withemptylabels\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
#results = model(r"D:\allWorkspace\pycharmWorkspace\futbol-players-7\valid\images\1-fps-2_00007_jpeg_jpg.rf.a3d1f8280daeb1829020234f994c1375.jpg")  # predict on an image
#path = model.export(format="torchscript")  # export the model to ONNX format  onnx
# #mybackbone_denseConcat.yaml
# model = YOLO("mybackbone_denseConcat.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangeData_withemptylabels\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
# #mybackbone_denseConcat_PFP.yaml
# model = YOLO("mybackbone_denseConcat_PFP.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangeData_withemptylabels\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
# #mybackbone_denseConcat_PFP_lightDetect.yaml
# model = YOLO("mybackbone_denseConcat_PFP_lightDetect.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangeData_withemptylabels\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
# #mybackbone_denseConcat_PFP_lightDetectWithAttention.yaml
# model = YOLO("mybackbone_denseConcat_PFP_lightDetectWithAttention.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangeData_withemptylabels\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
# #mybackbone_denseConcat_PFP_lightDetect_dySample.yaml
# model = YOLO("mybackbone_denseConcat_PFP_lightDetect_dySample.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangeData_withemptylabels\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)


# #mybackbone_denseConcat_PFP_lightDetect_dySample.yaml
# model = YOLO("mybackbone_denseConcat_PFP_lightDetect_dySample.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
# #mybackbone_denseConcat_PFP_lightDetect_dySample_withoutC2f.yaml
# model = YOLO("mybackbone_denseConcat_PFP_lightDetect_dySample_withoutC2f.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)

# #mybackbone_denseConcat_PFP_lightDetectwithoutAttention_withoutAllC2f.yaml
# model = YOLO("mybackbone_denseConcat_PFP_lightDetectwithoutAttention_withoutAllC2f.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)

# #mybackbone_denseConcat_PFP_lightDetectwithoutAttention_withoutAllC2f_APFPSPPF.yaml
# model = YOLO("mybackbone_denseConcat_PFP_lightDetectwithoutAttention_withoutAllC2f_APFPSPPF.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)


# #mybackbone_denseConcat_ResPFP_lightDetectwithoutAttention_withoutAllC2f_SPPF.yaml
# model = YOLO("mybackbone_denseConcat_ResPFP_lightDetectwithoutAttention_withoutAllC2f_SPPF.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
#
#
# #mybackbone_denseConcat_ResPFP_lightDetectwithoutAttention_withoutAllC2f_newAPFPSPPF.yaml
# model = YOLO("mybackbone_denseConcat_ResPFP_lightDetectwithoutAttention_withoutAllC2f_newAPFPSPPF.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)


# #mybackbone_denseConcat_PFP_lightDetectwithoutAttention_withoutAllC2f_APFPSPPF.yaml
# model = YOLO("mybackbone_denseConcat_PFP_lightDetectwithoutAttention_withoutAllC2f_APFPSPPF.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)


# #SUDCN.yaml
# model = YOLO("SUDCN.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
# #mybackbone_denseConcat_ShufflePFP_lightDetectwithoutAttention_withoutAllC2f_newAPFPSPPF.yaml
# model = YOLO("mybackbone_denseConcat_ShufflePFP_lightDetectwithoutAttention_withoutAllC2f_newAPFPSPPF.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
#
# #mybackbone_denseConcat_ShufflePFPAndShuffleDC_lightDetectwithoutAttention_withoutAllC2f_newAPFPSPPF.yaml
# model = YOLO("mybackbone_denseConcat_ShufflePFPAndShuffleDC_lightDetectwithoutAttention_withoutAllC2f_newAPFPSPPF.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
#
# #mybackbone_denseConcat_ShuffleResPFP_lightDetectwithoutAttention_withoutAllC2f_newAPFPSPPF.yaml
# model = YOLO("mybackbone_denseConcat_ShuffleResPFP_lightDetectwithoutAttention_withoutAllC2f_newAPFPSPPF.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)

#mybackbone_denseConcat_PartialResPFPWithPWGC_lightDetectwithoutAttention_withoutAllC2f_APFPSPPF.yaml
#model = YOLO("mybackbone_denseConcat_PartialResPFPWithPWGC_lightDetectwithoutAttention_withoutAllC2f_APFPSPPF.yaml")
#model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)

#mybackbone_denseConcat_PFP.yaml
# model = YOLO("mybackbone_denseConcat_PFP.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
#==========================ablation attention======================
#attentionAblationEMA.yaml
# model = YOLO("attentionAblationEMA.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
#
# #LSKA
# model = YOLO("attentionAblationLSKA.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
#
# #OD
# model = YOLO("attentionAblationOD.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
# #SE
# model = YOLO("attentionAblationSE.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
# #CA
# model = YOLO("attentionAblationCA.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)
#
# #NoAttention.yaml
# model = YOLO("NoAttention.yaml")
# model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)

#lossWIOU.yaml
#model = YOLO("lossPIOU.yaml")
#model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)

#lossWIOU_PCDownV2.yaml
#lossWIOU_APCDownV2.yaml
#lossWIOU_SCDown.yaml
#runs/detect/train68/weights/last.pt   接着跑
#mybackbone_denseConcat_PartialResPFP_lightDetectwithoutAttention_withoutAllC2f_ABFFSPPF
model = YOLO(r"SUDCN.yaml") #ABFFSPPF
model.train(data=r"E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml", epochs=300,workers = 0,imgsz=640,batch=16)


#==========================ablation attention======================
#数据集
#E:\data_set\infraredSmallObject\arrangedData_testLabel\data.yaml
#E:\data_set\infraredSmallObject\arrangeData_withemptylabels\data.yaml

# Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     result.show()  # display to screen
#     result.save(filename='result.jpg')  # save to disk

#yolo detect train workers=0 data=E:/data_set/infraredSmallObject/arrangedData_testLabel/data.yaml model=SUDCN.yaml epochs=300 batch=16 imgsz=640 device=0