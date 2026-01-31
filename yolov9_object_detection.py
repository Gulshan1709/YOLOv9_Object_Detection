import cv2
import torch
from ultralytics import YOLO

def main():
    # Step 1: Select device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 2: Load YOLOv9 model
    model = YOLO("yolov9c.pt")
    model.to(device)

    # Step 3: Initialize video capture (0 = default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return

    print("Press 'q' to exit")

    # Step 4: Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Run YOLOv9 inference
        results = model(frame)

        # Step 5: Draw bounding boxes and labels
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                label_id = int(box.cls[0].item())
                confidence = box.conf[0].item()

                class_label = model.names[label_id]
                label_text = f"{class_label}: {confidence:.2f}"

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                # Draw label
                cv2.putText(
                    frame,
                    label_text,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

        # Step 6: Display the result
        cv2.imshow("Real-Time Object Detection (YOLOv9)", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Step 7: Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
