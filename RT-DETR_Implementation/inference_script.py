from ultralytics import RTDETR
#Load a COCO-pretrained RT-DETR-L model
model = RTDETR('rtdetr-l.pt')
# Display model information (optional)
model.info()
# Run inference with the RT-DETR-L model on the 'bus.jpg" image
results = model("inference/images/Test.jpg", show=True, save=True)
