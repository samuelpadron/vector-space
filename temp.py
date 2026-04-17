from ultralytics import YOLO 

# Load pretrained YOLO model
model = YOLO("yolo26n.pt")

# Train model for 3 epochs
results = model.train(data="coco8.yaml", epochs=3)

results = model.val()

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")

