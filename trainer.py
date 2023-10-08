import os
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import matplotlib.pyplot as plt
from segmentation.data import preview_data, preview_dataset, get_datasets, get_dataloaders, train_id_to_colour
from segmentation.model import PSPNet, PSPNetLoss
from segmentation.utils import meanIoU, plot_training_results, evaluate_model, train_validate_model, visualize_predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%d-%m-%Y %H:%M',
                        filename='trainer.log',
                        filemode='w')
                        
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format without the date time for console 
    formatter = logging.Formatter('%(name)-12s %(levelname)-4s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    data_logger = logging.getLogger('PSPNET datalogger')
    logger1 = logging.getLogger('PSPNet Main')
    logger2 = logging.getLogger('PSPNet Trainer')

    dataset = os.path.join(os.getcwd(), "dataset")
    raw_images = os.path.join(dataset, "image_180_320.npy")
    raw_labels = os.path.join(dataset, "label_180_320.npy")
    images = np.load(raw_images)
    labels = np.load(raw_labels)

    idx = 202
    # preview_data(images, labels, idx)

    train_set, val_set, test_set = get_datasets(images, labels)
    logger1.info(f"There are {len(train_set)} train images, {len(val_set)} validation images, {len(test_set)} test images")

    sample_image, sample_label = train_set[0]
    logger1.info(f"Input shape = {sample_image.shape}, Output shape = {sample_label.shape}")

    # preview_dataset(train_set)

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_set, val_set, test_set)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    N_EPOCHS = 6
    NUM_CLASSES = 3
    MAX_LR = 3e-4
    MODEL_NAME = "PSPNET_resnet50_aux"

    # aux_weight refers to the alpha hyperparam in the paper
    criterion = PSPNetLoss(num_classes=NUM_CLASSES, aux_weight=0.4)

    model = PSPNet(in_channels=3, num_classes=NUM_CLASSES, use_aux=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=N_EPOCHS,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.3,
        div_factor=10,
        anneal_strategy="cos"
    )

    # Train loop
    output_path = os.path.join(os.getcwd(), "artifacts")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logger1.info("Train model ...")
    results = train_validate_model(model, N_EPOCHS, MODEL_NAME, criterion, optimizer, device, train_dataloader, val_dataloader, meanIoU, 'meanIoU', NUM_CLASSES, lr_scheduler=scheduler, output_path=output_path, savefig=True, logger=logger2)

    logger1.info("Evaluating model on test set...")
    model.load_state_dict(torch.load(f"{output_path}/{MODEL_NAME}.pt", map_location=device))

    _, test_metric = evaluate_model(model, test_dataloader, criterion, meanIoU, NUM_CLASSES, device)
    logger1.info(f"Model has {test_metric} mean IoU in test set")

    num_test_samples = 2
    _, axes = plt.subplots(num_test_samples, 3, figsize=(3 * 6, num_test_samples * 4))

    visualize_predictions(model, test_set, axes, device, numTestSamples=num_test_samples, id_to_color=train_id_to_colour(), savefig=True)