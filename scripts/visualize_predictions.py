import cv2
import numpy as np


def visualize_predictions(images, predictions, labels, num_samples=5):
    images = images.cpu().numpy()
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    errors_indices = [i for i, (pred, label) in enumerate(zip(predictions, labels)) if pred != label]
    errors_to_display = min(num_samples, len(errors_indices))

    for i in range(errors_to_display):
        error_index = errors_indices[i]
        img = images[error_index].transpose(1, 2, 0)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        pred_label = predictions[error_index]
        true_label = labels[error_index]

        cv2.putText(img, 'Pred: {}'.format(pred_label), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(img, 'True: {}'.format(true_label), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        window_name = 'Error Sample {}'.format(i)
        cv2.imshow(window_name, img)
        cv2.waitKey(1000)  # 等待1毫秒

    cv2.destroyAllWindows()  # 关闭所有窗口
