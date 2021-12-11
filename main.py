import cv2

conf_scale = 0.6
nms_scale = 0.4
colors = [(0, 0, 255), (255, 255, 255)]
class_names = ["maskeli", "maskesiz"]
cap = cv2.VideoCapture(0)
net = cv2.dnn.readNet("pretrained_model/yolov4-custom.cfg","pretrained_model/yolov4.weights")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
while 1:
    (_, frame) = cap.read()
    classes, scores, boxes = model.detect(frame, conf_scale, nms_scale)
    for (classid, score, box) in zip(classes, scores, boxes):
        color_id = int(classid) % len(colors)
        color = colors[color_id]
        label = "%s -> %%%.2f" % (class_names[classid[0]], score * 100)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    cv2.imshow("detections", frame)

    if cv2.waitKey(1) > 1:
        break