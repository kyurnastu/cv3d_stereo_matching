import numpy as np

def show_results(left_image, result_disparity, gt_disparity):
    result_disparity = np.array(result_disparity, dtype=np.float32)
    gt_disparity = np.array(gt_disparity, dtype=np.float32)

    mask_gt_exists = gt_disparity > 0
    mask_correct = ((abs(result_disparity - gt_disparity) < 3) * mask_gt_exists).astype(np.float32)
    mask_incorrect = (1 - mask_correct) * mask_gt_exists

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(result_disparity / np.max(result_disparity))
    plt.subplot(1, 2, 2)
    plt.imshow(left_image // 2)
    plt.imshow(np.stack([mask_incorrect, mask_correct, np.zeros(mask_correct.shape)], axis=2), alpha=0.7)
    plt.show()
