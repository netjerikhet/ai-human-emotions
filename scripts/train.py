from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from model.full_model import GeneratorFullModel, DiscriminatorFullModel
from torch.optim.lr_scheduler import MultiStepLR
from batchnorm_sync import DataParallelWithCallback
from frames_dataset import DatasetRepeater
from face_cropper import FaceCropper
import imageio
import os
import cv2
from skimage.transform import resize
from output import load_checkpoints, make_animation
import wandb
import numpy as np

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):
    wandb_obj = wandb.init(project="training emotion model", entity="ai-human-emotion")
    train_params = config['train_params']
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector, optimizer_generator, 
        optimizer_discriminator, Non if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
    else:
        start_epoch = 0

    optimizer_generator.param_groups[0]['lr'] = train_params['lr_generator']
    optimizer_discriminator.param_groups[0]['lr'] = train_params['lr_discriminator']
    optimizer_kp_detector.param_groups[0]['lr'] = train_params['lr_kp_detector']
        

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
    last_epoch=start_epoch -1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
    last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
    last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=1, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)
    if train_params['freeze_layers'] == True:
        # generator_full.kp_extractor.requires_grad_(False)
        generator_full.generator.dense_motion_network.requires_grad_(False)
        # generator_full.generator.up_blocks.requires_grad_(False)
        # generator_full.discriminator.requires_grad_(False)
        generator_full.vgg.requires_grad_(False)
        # generator_full.vgg.slice2.requires_grad_(False)
        # generator_full.vgg.slice3.requires_grad_(False)

        # discriminator_full.kp_extractor.requires_grad_(False)
        # discriminator_full.generator.dense_motion_network.requires_grad_(False)
        # discriminator_full.generator.first.requires_grad_(False)
        # discriminator_full.generator.down_blocks.requires_grad_(False)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)
    
    with Logger(log_dir=log_dir, wandb_object=wandb_obj, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            print("Current epoch generator learning rate:", get_lr(optimizer_generator))
            for x in dataloader:
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            logger.log_epoch(epoch, {'generator': generator,
                                        'discriminator': discriminator,
                                        'kp_detector': kp_detector,
                                        'optimizer_generator': optimizer_generator,
                                        'optimizer_discriminator': optimizer_discriminator,
                                        'optimizer_kp_detector': optimizer_kp_detector}, inp=x, out=generated)