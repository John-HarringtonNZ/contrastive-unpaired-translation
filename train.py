import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import wandb

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    # Initialize Wandb
    wandb.init('CUT_model')
    wandb.define_metric("training_step")
    wandb.define_metric("train/epoch")
    wandb.define_metric("train/gan_loss", step_metric="training_step")
    wandb.define_metric("train/d_real_loss", step_metric="training_step")
    wandb.define_metric("train/d_fake_loss", step_metric="training_step")
    wandb.define_metric("train/g_loss", step_metric="training_step")
    wandb.define_metric("train/nce_loss", step_metric="training_step")
    
    wandb.define_metric("train/mIOU", step_metric="train/epoch")
    wandb.define_metric("train/fid", step_metric="train/epoch")

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            model.compute_metrics_on_batch()
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                # WANDB
                losses = model.get_current_losses()
                wandb.log({
                    'train/gan_loss': losses['G_GAN'],
                    'train/d_real_loss': losses['D_real'],
                    'train/d_fake_loss': losses['D_fake'],
                    'train/g_loss': losses['G'],
                    'train/nce_loss': losses['NCE'],
                    'training_step' : i})

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        # Compute applicable metrics
        metrics = model.get_metrics()
        if 'fid' in metrics:
            print(f'Epoch {epoch}; FID Score: {metrics["fid"]}')

            # WANDB
            images = model.get_current_visuals()
            wandb.log({f"training/realA, Epoch {epoch}": [wandb.Image(images['real_A'])]})
            wandb.log({f"training/fakeB, Epoch {epoch}": [wandb.Image(images['fake_B'])]})
            wandb.log({f"training/realB, Epoch {epoch}": [wandb.Image(images['real_B'])]})
            wandb.log({
                'train/epoch': epoch,
                'train/fid': metrics["fid"]})

        # TODO add FID plot to visualizer as well
        visualizer.plot_current_metrics(epoch, metrics)
        model.reset_metrics()

        # step lambda
        if opt.segmentation_loss:
            model.step_segmentation_lambda()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
