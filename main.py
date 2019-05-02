from datetime import datetime
from time import sleep
import time
import copy
import os
import shutil
import math
import numpy as np
import itertools

import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from tensorboardX import SummaryWriter

import tqdm

import my_losses

from utils import get_train_val_loader, get_eval_loader, end_point_error, metrics
from PlyFile import PlyFile
from config import Config
import platform

from models_voxelnet import SiameseModel3D_1M_no_second_1x1conv as SiameseModel3D
from models_voxelnet import SiameseVoxelNet as SiameseVoxelNet
from models_pointnet import SiamesePointNet as SiamesePointNet

MODELS = {
    'SiameseModel3D': SiameseModel3D,
    'SiameseVoxelNet': SiameseVoxelNet,
    'SiamesePointNet': SiamesePointNet
}


def train(cfg):
    ## Get current time
    date = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

    learning_rates = [cfg.learning_rate]
    weight_decays = [0]
    use_batchnorms = [cfg.use_batchnorm]
    use_locals = [False]
    use_normals = [True]
    loss_types = ['MySmoothL1LossSparse']

    for use_batchnorm, learning_rate, weight_decay, use_local, use_normal, loss_type \
            in itertools.product(use_batchnorms, learning_rates, weight_decays,
                                 use_locals, use_normals, loss_types):

        ## Get train and val loader
        train_loader, val_loader = get_train_val_loader(dataset_dir=cfg.data_dir[cfg.data_type_3d],
                                                        data_split="TRAIN",
                                                        data_type=cfg.data_type_3d,
                                                        use_local=use_local,
                                                        use_normal=use_normal,
                                                        sequences_to_train=cfg.sequences_to_train,
                                                        batch_size_train=cfg.batch_sizes["train"],
                                                        batch_size_val=cfg.batch_sizes["val"],
                                                        validation_percentage=0.1)

        criterion_epe = my_losses.EPELoss(size_average=cfg.size_average)

        if loss_type == 'MySmoothL1LossSparse':
            criterion = my_losses.MySmoothL1LossSparse(size_average=cfg.size_average)
        elif loss_type == 'EPELoss':
            criterion = my_losses.EPELoss(size_average=cfg.size_average)

        ## Load model
        if cfg.model_name == "SiameseModel3D":
            model_raw = MODELS[cfg.model_name](verbose=False,
                                           use_bn=use_batchnorm,
                                           use_dropout=cfg.use_dropout)
        else:
            model_raw = MODELS[cfg.model_name](verbose=False,
                                       use_bn=use_batchnorm,
                                       use_dropout=cfg.use_dropout,
                                       use_local_features=use_local,
                                       use_normals=use_normal,
                                       nfeat=cfg.nfeat)



        # model = nn.DataParallel(model_raw, device_ids=[0, 1])
        # device = torch.device("cuda:2")
        #model.to(device)
        model = model_raw
        model.cuda()


        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of params in model", cfg.model_name, "is", pytorch_total_params)

        if cfg.model_name == "SiameseModel3D":
            info = loss_type + \
                "_lr" + str(learning_rate) + \
                "_lrDecay" + str(cfg.use_lr_decay) + \
                "_dropout" + str(cfg.use_dropout) + \
                "_batchNorm" + str(use_batchnorm) + \
                "_wDecay" + str(weight_decay) + \
                "_bs" + str(cfg.batch_sizes["train"]) + \
                "_e" + str(cfg.epochs) + \
                "_" + cfg.np_datasets_base_path + \
                "-" + cfg.sequences_to_train_str
        else:
            info = loss_type + \
                "_local" + str(use_local) + \
                "_normal" + str(use_normal) + \
                "_lr" + str(learning_rate) + \
                "_lrDecay" + str(cfg.use_lr_decay) + \
                "_dropout" + str(cfg.use_dropout) + \
                "_batchNorm" + str(use_batchnorm) + \
                "_wDecay" + str(weight_decay) + \
                "_bs" + str(cfg.batch_sizes["train"]) + \
                "_e" + str(cfg.epochs) + \
                "_" + cfg.np_datasets_base_path + \
                "-" + cfg.sequences_to_train_str

        info_time = date + "_" + info
        data_source = cfg.data_source[cfg.data_type_3d]

        log_dir = cfg.log_dir_base[cfg.data_type_3d] + "/" + model_raw.__class__.__name__ + "/" + info_time
        save_path_best = log_dir + "/" + "BEST.net"
        save_path_last = log_dir + "/" + "LAST.net"

        #############################
        #############################
        trained = False
        try:
            ## Initialise summary writer

            writer = SummaryWriter(log_dir=log_dir)

            best_model_loss = cfg.best_model_loss
            best_model_epe = cfg.best_model_loss
            best_model = ""

            if cfg.optimizer_type is "adam":
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            else:
                raise Exception("Set an optimizer type")

            hyperparams = "Training on dataset: " + cfg.dataset + "_" + str(cfg.n_voxels) \
                          + "\nLoss type: " + str(loss_type) \
                          + "\nLearning rate:" + str(learning_rate) \
                          + "\nOptimizer: " + str(cfg.optimizer_type) \
                          + "\nBatch size: " + str(cfg.batch_sizes["train"]) \
                          + "\nEpochs: " + str(cfg.epochs) \
                          + "\nNumber of voxels: " + str(cfg.n_voxels)

            writer.add_text(data_source + "/hyperparams", hyperparams)
            print(hyperparams, "\n")

            iteration = -1
            # Loop over the dataset multiple times
            #for epoch in tqdm.trange(cfg.epochs, desc='\nEpoch'):
            for epoch in range(cfg.epochs):

                if cfg.use_lr_decay and epoch >= cfg.start_lr_decay and epoch % cfg.lr_decay_step_size == 0:
                    #lr = cfg.learning_rate * (cfg.lr_gamma ** (epoch // (cfg.start_lr_decay + cfg.lr_decay_step_size)))
                    learning_rate = learning_rate * cfg.lr_gamma
                    print(learning_rate)
                    if learning_rate > 1e-5:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate
                            print("Epoch", epoch, "- Learning rate:", param_group['lr'])
                    else:
                        cfg.use_lr_decay = False

                # Loop over the different mini-batches
                #t = tqdm.tqdm(iter(train_loader), leave=True, total=len(train_loader), desc='----Batch')
                for batch_train, sample in enumerate(train_loader):
                    iteration += 1
                    ####################
                    ##### TRAINING #####
                    ####################
                    model.train()

                    optimizer.zero_grad()

                    pcl_data_t0, pcl_data_t1, sf_data_gt, samples_name = sample

                    if cfg.model_name == "SiameseModel3D":
                        voxel_coords_t0 = pcl_data_t0
                        voxel_coords_t1 = pcl_data_t1
                        sf_gt = sf_data_gt
                        voxel_coords_t0 = torch.from_numpy(voxel_coords_t0).type(torch.cuda.LongTensor)
                        voxel_coords_t1 = torch.from_numpy(voxel_coords_t1).type(torch.cuda.LongTensor)
                        sf_gt = torch.cuda.FloatTensor(sf_gt)
                        sceneflow_pred = model(False, "train",
                                               voxel_coords_t0, voxel_coords_t1)

                    elif cfg.model_name == "SiamesePointNet":
                        voxel_features_t0, voxel_coords_t0 = pcl_data_t0
                        voxel_features_t1, voxel_coords_t1 = pcl_data_t1
                        sf_gt = sf_data_gt

                        ## Put our inputs into cuda
                        voxel_features_t0 = torch.cuda.FloatTensor(voxel_features_t0)
                        voxel_coords_t0 = torch.from_numpy(voxel_coords_t0).type(torch.cuda.LongTensor)

                        voxel_features_t1 = torch.cuda.FloatTensor(voxel_features_t1)
                        voxel_coords_t1 = torch.from_numpy(voxel_coords_t1).type(torch.cuda.LongTensor)

                        sf_gt = torch.cuda.FloatTensor(sf_gt)

                        sceneflow_pred = model(False, "train",
                                               (voxel_features_t0, voxel_coords_t0),
                                               (voxel_features_t1, voxel_coords_t1))

                    elif cfg.model_name == "SiameseVoxelNet":
                        _, voxel_features_t0, voxel_coords_t0, _ = pcl_data_t0
                        _, voxel_features_t1, voxel_coords_t1, _ = pcl_data_t1
                        _, sf_gt = sf_data_gt

                        ## Put our inputs into cuda
                        voxel_features_t0 = torch.cuda.FloatTensor(voxel_features_t0)
                        voxel_coords_t0 = torch.from_numpy(voxel_coords_t0).type(torch.cuda.LongTensor)

                        voxel_features_t1 = torch.cuda.FloatTensor(voxel_features_t1)
                        voxel_coords_t1 = torch.from_numpy(voxel_coords_t1).type(torch.cuda.LongTensor)

                        sf_gt = torch.cuda.FloatTensor(sf_gt)

                        sceneflow_pred = model(False, "train",
                                               (voxel_features_t0, voxel_coords_t0),
                                               (voxel_features_t1, voxel_coords_t1))


                    loss = criterion(sceneflow_pred, sf_gt)
                    epe = criterion_epe(sceneflow_pred, sf_gt)
                    if math.isnan(loss.item()):
                        raise Exception("Loss is NAN in ", samples_name)
                    loss.backward()
                    optimizer.step()

                    # cache loss and epe from training
                    loss_train = loss.item()
                    epe_loss = epe.item()

                    ####################
                    #### Statistics ####
                    ####################
                    if batch_train == 0 or (batch_train > 0 and batch_train % cfg.val_every == 0):
                        with torch.no_grad():
                            ## Save training loss in loss_dict, later to be used in writer
                            loss_dict = {'training_loss': loss_train, 'training_epe': epe_loss}

                            ####################
                            #### VALIDATION ####
                            ####################
                            if cfg.do_validation and len(val_loader) != 0:
                                running_val_loss = 0
                                running_val_epe = 0
                                validated_batches = 0

                                for batch_val, sample_val in enumerate(val_loader):

                                    if validated_batches < cfg.batches_to_val:
                                        pcl_data_t0, pcl_data_t1, sf_data_gt, samples_name = sample_val

                                        model.eval()

                                        if cfg.model_name == "SiameseModel3D":
                                            voxel_coords_t0 = pcl_data_t0
                                            voxel_coords_t1 = pcl_data_t1
                                            sf_gt = sf_data_gt
                                            voxel_coords_t0 = torch.from_numpy(voxel_coords_t0).type(
                                                torch.cuda.LongTensor)
                                            voxel_coords_t1 = torch.from_numpy(voxel_coords_t1).type(
                                                torch.cuda.LongTensor)
                                            sf_gt = torch.cuda.FloatTensor(sf_gt)
                                            sceneflow_pred = model(True, "val",
                                                                   voxel_coords_t0, voxel_coords_t1)

                                        elif cfg.model_name == "SiamesePointNet":
                                            voxel_features_t0, voxel_coords_t0 = pcl_data_t0
                                            voxel_features_t1, voxel_coords_t1 = pcl_data_t1
                                            sf_gt = sf_data_gt

                                            ## Put our inputs into cuda
                                            voxel_features_t0 = torch.cuda.FloatTensor(voxel_features_t0)
                                            voxel_coords_t0 = torch.from_numpy(voxel_coords_t0).type(
                                                torch.cuda.LongTensor)

                                            voxel_features_t1 = torch.cuda.FloatTensor(voxel_features_t1)
                                            voxel_coords_t1 = torch.from_numpy(voxel_coords_t1).type(
                                                torch.cuda.LongTensor)

                                            sf_gt = torch.cuda.FloatTensor(sf_gt)

                                            sceneflow_pred = model(True, "val",
                                                                   (voxel_features_t0, voxel_coords_t0),
                                                                   (voxel_features_t1, voxel_coords_t1))


                                        elif cfg.model_name == "SiameseVoxelNet":
                                            voxel_features_t0, voxel_coords_t0 = pcl_data_t0
                                            voxel_features_t1, voxel_coords_t1 = pcl_data_t1
                                            sf_gt = sf_data_gt

                                            ## Put our inputs into cuda
                                            voxel_features_t0 = torch.cuda.FloatTensor(voxel_features_t0)
                                            voxel_coords_t0 = torch.from_numpy(voxel_coords_t0).type(
                                                torch.cuda.LongTensor)

                                            voxel_features_t1 = torch.cuda.FloatTensor(voxel_features_t1)
                                            voxel_coords_t1 = torch.from_numpy(voxel_coords_t1).type(
                                                torch.cuda.LongTensor)

                                            sf_gt = torch.cuda.FloatTensor(sf_gt)

                                            sceneflow_pred = model(True, "val",
                                                                   (voxel_features_t0, voxel_coords_t0),
                                                                   (voxel_features_t1, voxel_coords_t1))

                                        loss = criterion(sceneflow_pred, sf_gt)
                                        running_val_loss += loss.item()

                                        epe = criterion_epe(sceneflow_pred, sf_gt)
                                        running_val_epe += epe.item()

                                        validated_batches += 1
                                    else:
                                        break

                                print("Validated batches", validated_batches)
                                average_val_loss = running_val_loss / validated_batches
                                average_val_epe = running_val_epe / validated_batches

                                print('\nEpoch %d - Batch %d - Iteration: %d' % (epoch, batch_train, iteration),
                                      ', Training Loss: %0.6f' % (loss_train),
                                      ', Training EPE: %0.6f' % (epe_loss),
                                      ', Average Validation Loss: %0.6f' % average_val_loss,
                                      ', Average Validation EPE: %0.6f' % average_val_epe,
                                      ', LR %.14f' % optimizer.param_groups[0]['lr'])

                                ## Add validation loss to loss_dict for later printing in Tensorboard
                                loss_dict['validation_loss'] = average_val_loss
                                loss_dict['validation_epe'] = average_val_epe

                                ## Save the model
                                if average_val_epe < best_model_epe:
                                    best_model_epe = average_val_epe
                                    best_model = copy.deepcopy(model)
                                    torch.save(best_model, save_path_best)

                                if average_val_loss < best_model_loss:
                                    best_model_loss = average_val_loss

                            else:
                                print('\nEpoch %d - Batch %d - Iteration: %d' % (epoch, batch_train, iteration),
                                      ', Training Loss: %0.6f' % (loss_train))

                            writer.add_scalars(data_source, loss_dict,
                                               batch_train + len(train_loader) * epoch)


            if platform.system() == "Linux":
                best_model_epe_str = '%.5f' % best_model_epe
                info_time = best_model_epe_str + "_" + date + "_" + info
                log_dir_old = log_dir
                log_dir = cfg.log_dir_base[cfg.data_type_3d] + "/" + model_raw.__class__.__name__ + "/" + info_time
                save_path_best_old = save_path_best
                save_path_best = log_dir + "/" + "BEST.net"
                save_path_last = log_dir + "/" + "LAST.net"

                if cfg.do_validation:
                    os.remove(save_path_best_old)
                os.rename(log_dir_old, log_dir)
                if cfg.do_validation:
                    torch.save(best_model, save_path_best)

            with open(log_dir + "/training_results.txt", 'w') as f:
                f.write("Best model val loss " + str(best_model_loss) + "\n")
                f.write("Best model val epe " + str(best_model_epe) + "\n")

            ## Save last model and best model
            torch.save(model, save_path_last)
            trained = True

            print("\nTraining Done!")

            ####################
            ##### evalING ######
            ####################
            if cfg.do_eval:
                model_eval = best_model if cfg.model_quality_to_use_at_eval == "BEST" else model
                eval(cfg, criterion, use_local, use_normal, model_eval, model_raw.__class__.__name__,
                     log_dir, info_time)

        except KeyboardInterrupt:
            print("Bye!")
            writer.close()
            exit(-1)

        writer.close()
        sleep(2)

        print("Finished with model")

    print("Completely Finished.")


def eval(cfg, criterion, use_local, use_normal, model=None, model_name=None, logdir=None, info_time=None):
    ## Get eval loader
    eval_loader = get_eval_loader(dataset_dir=cfg.data_dir[cfg.data_type_3d],
                                  data_split="TEST",
                                  data_type=cfg.data_type_3d,
                                  use_local=use_local,
                                  use_normal=use_normal,
                                  sequences_to_eval=cfg.sequences_to_eval,
                                  batch_size=cfg.batch_sizes["eval"])

    ## Load model
    if model is None:
        if not cfg.do_eval:
            ## Just playing around with a new raw model
            model = MODELS[cfg.model_name](verbose=False,
                                           use_bn=cfg.use_batchnorm,
                                           use_dropout=cfg.use_dropout)
            logdir = cfg.log_dir_base[cfg.data_type_3d] + "/" + model_name + "/quickevals"
        else:
            ## Load existing model
            if cfg.model_dir_to_use_at_eval is not None:
                logdir = os.path.join(cfg.log_dir_base[cfg.data_type_3d], MODELS[cfg.model_name].__name__,
                                      cfg.model_dir_to_use_at_eval)
                model_path = logdir + "/" + cfg.model_quality_to_use_at_eval + ".net"
                model = torch.load(model_path)
                print("\nLoaded model from ", model_path)

            else:
                raise Exception("Define a model for evaling!")

    with torch.no_grad():
        model.eval()
        running_loss_eval = 0
        # end-point-error: average L2 distance between the estimated
        # flow vector to the ground truth flow vector
        running_EPE_eval_vg = 0
        running_EPE_eval_pts = 0
        # accuracy: measures the portion of estimated flow vectors that are
        # below a specified end point error, among all the points
        running_accuracy_1_vg = 0
        running_accuracy_2_vg = 0
        running_accuracy_1_pts = 0
        running_accuracy_2_pts = 0
        num_samples = 0

        # Loop over the different mini-batches
        t = tqdm.tqdm(iter(eval_loader), leave=False, total=len(eval_loader))
        for batch_eval, sample_test in enumerate(t):
            model.cuda()

            pcl_data_t0, pcl_data_t1, sf_data_gt, samples_name = sample_test

            if cfg.model_name == "SiameseModel3D":
                pcl_t0, voxel_coords_t0, inv_ind_t0 = pcl_data_t0
                pcl_t1, voxel_coords_t1, _ = pcl_data_t1
                sf_gt_pts, sf_gt = sf_data_gt
                voxel_coords_t0 = torch.from_numpy(voxel_coords_t0).type(
                    torch.cuda.LongTensor)
                voxel_coords_t1 = torch.from_numpy(voxel_coords_t1).type(
                    torch.cuda.LongTensor)
                sf_gt = torch.cuda.FloatTensor(sf_gt)
                sceneflow_pred = model(True, "eval",
                                       voxel_coords_t0, voxel_coords_t1)

            elif cfg.model_name == "SiamesePointNet":
                pcl_t0, voxel_features_t0, voxel_coords_t0, inv_ind_t0 = pcl_data_t0
                pcl_t1, voxel_features_t1, voxel_coords_t1, _ = pcl_data_t1
                sf_gt_pts, sf_gt = sf_data_gt

                ## Put our inputs into cuda
                voxel_features_t0 = torch.cuda.FloatTensor(voxel_features_t0)
                voxel_coords_t0 = torch.from_numpy(voxel_coords_t0).type(torch.cuda.LongTensor)

                voxel_features_t1 = torch.cuda.FloatTensor(voxel_features_t1)
                voxel_coords_t1 = torch.from_numpy(voxel_coords_t1).type(torch.cuda.LongTensor)

                sf_gt = torch.cuda.FloatTensor(sf_gt)

                sceneflow_pred = model(True, "eval",
                                       (voxel_features_t0, voxel_coords_t0),
                                       (voxel_features_t1, voxel_coords_t1))

            elif cfg.model_name == "SiameseVoxelNet":
                pcl_t0, voxel_features_t0, voxel_coords_t0, inv_ind_t0 = pcl_data_t0
                pcl_t1, voxel_features_t1, voxel_coords_t1, _ = pcl_data_t1
                sf_gt_pts, sf_gt = sf_data_gt

                ## Put our inputs into cuda
                voxel_features_t0 = torch.cuda.FloatTensor(voxel_features_t0)
                voxel_coords_t0 = torch.from_numpy(voxel_coords_t0).type(torch.cuda.LongTensor)

                voxel_features_t1 = torch.cuda.FloatTensor(voxel_features_t1)
                voxel_coords_t1 = torch.from_numpy(voxel_coords_t1).type(torch.cuda.LongTensor)

                sf_gt = torch.cuda.FloatTensor(sf_gt)

                sceneflow_pred = model(True, "eval",
                                       (voxel_features_t0, voxel_coords_t0),
                                       (voxel_features_t1, voxel_coords_t1))

            sample_name = samples_name[0]
            # print(sample_name)
            loss = criterion(sceneflow_pred, sf_gt)
            # print("----loss:", loss.item())
            running_loss_eval += loss.item()

            ############
            ## VOXELS ##
            ############
            ## METRICS AND DRAWING PLYs
            ## Move to CPU and convert to numpy
            sceneflow_pred_np = sceneflow_pred.cpu().numpy()
            sceneflow_gt_np = sf_gt.cpu().numpy()
            ## Compute End-Point-Error and Accuracies
            EPE_vg, acc1_vg, acc2_vg = metrics(sceneflow_pred_np, sceneflow_gt_np)
            running_EPE_eval_vg += EPE_vg
            running_accuracy_1_vg += acc1_vg
            running_accuracy_2_vg += acc2_vg

            if any(seq in sample_name for seq in cfg.sequences_to_draw):

                ################## Draw PLY file #######################
                voxel_coords_t0 = voxel_coords_t0.cpu().numpy()
                coords = voxel_coords_t0[:, 1:]
                ## Get our arrow
                arrow_vertices, arrow_faces = PlyFile.read_ply("ply_examples/awesome_arrow.ply")
                # ## Generate a PlyFile object and initialize it with our points from the voxelgrid
                plyfile_vg = PlyFile(coords, color=[255, 0, 0])
                plyfile_vg.draw_arrows_for_sceneflow(coords, sceneflow_gt_np, arrow_vertices,
                                                     arrow_faces, color=[0, 255, 0]) # Green for prediction
                plyfile_vg.draw_arrows_for_sceneflow(coords, sceneflow_pred_np, arrow_vertices,
                                                     arrow_faces, color=[255, 0, 0]) # Red for prediction
                plyfile_vg.write_ply(logdir + "/plys-" + cfg.model_quality_to_use_at_eval + "/" +
                                     sample_name + "-EPE_vg" + str(EPE_vg) + ".ply")

            ## ALL POINTS ##
            ## METRICS AND DRAWING PLYs
            ## Move to CPU and convert to numpy
            sceneflow_pred = sceneflow_pred.cpu().numpy()
            sceneflow_pred_np = sceneflow_pred[inv_ind_t0]
            sceneflow_gt_np = sf_gt_pts
            ## Compute End-Point-Error and Accuracies
            EPE_pts, acc1_pts, acc2_pts = metrics(sceneflow_pred_np, sceneflow_gt_np)
            running_EPE_eval_pts += EPE_pts
            running_accuracy_1_pts += acc1_pts
            running_accuracy_2_pts += acc2_pts

            if any(seq in sample_name for seq in cfg.sequences_to_draw):
                ################## Align Pointcloud ####################
                plyfile = PlyFile(pcl_t0, color=[255, 255, 255])  # White for Pointcloud at t0
                plyfile.append_points(pcl_t1, color=[0, 255, 0])  # Green for Pointcloud at t1
                flowed_points = pcl_t0 + sceneflow_pred_np
                plyfile.append_points(flowed_points, color=[0, 0, 0])  # Black for predicted flowed points
                plyfile.write_ply(logdir + "/plys-" + cfg.model_quality_to_use_at_eval + "/" +
                                  sample_name + "-EPE_pts" + str(EPE_pts) + ".ply")

            num_samples += 1

        print(num_samples, len(eval_loader))

        average_eval_loss = running_loss_eval / num_samples
        average_eval_EPE_vg = running_EPE_eval_vg / num_samples
        average_eval_EPE_pts = running_EPE_eval_pts / num_samples

        ## Accuracy
        average_acc1_vg = round(running_accuracy_1_vg / num_samples * 100, 3)
        average_acc1_pts = round(running_accuracy_1_pts / num_samples * 100, 3)
        average_acc2_vg = round(running_accuracy_2_vg / num_samples * 100, 3)
        average_acc2_pts = round(running_accuracy_2_pts / num_samples * 100, 3)


        with open(logdir + "/eval_results.txt", 'w') as f:
            f.write("Number of batches evaled: " + str(len(eval_loader)) + "\n")
            f.write("Average eval loss " + str(average_eval_loss) + "\n")
            f.write("Average eval EPE vg " + str(average_eval_EPE_vg) + "\n")
            f.write("Average eval EPE pts" + str(average_eval_EPE_pts) + "\n\n")

            f.write("Average eval acc 1 vg " + str(average_acc1_vg) + "\n")
            f.write("Average eval acc 1 pts " + str(average_acc1_pts) + "\n")
            f.write("Average eval acc 2 vg " + str(average_acc2_vg) + "\n")
            f.write("Average eval acc 2 pts " + str(average_acc2_pts) + "\n")

        print("Number of batches evaled:", len(eval_loader))
        print("Average eval loss", average_eval_loss)
        print("Average eval EPE vg", average_eval_EPE_vg)
        print("Average eval EPE pts", average_eval_EPE_pts)

        print("Average eval acc 1 vg", average_acc1_vg)
        print("Average eval acc 1 pts", average_acc1_pts)
        print("Average eval acc 2 vg", average_acc2_vg)
        print("Average eval acc 2 pts", average_acc2_pts)


        ## Rename folder by prepending average_eval_EPE_pts

        if (info_time is not None):
            # We have just trained a model, and we want to prepend
            # the average_eval_EPE_pts to its folder name
            average_eval_EPE_pts_str = '%.5f' % average_eval_EPE_pts
            info_time = str(average_eval_EPE_pts_str) + "-" + info_time
            new_logdir = cfg.log_dir_base[cfg.data_type_3d] + "/" + model_name + "/" + info_time
            os.rename(logdir, new_logdir)

        print("\nEvaluating model Done!")


def run():
    print("Welcome! Happy training :)")

    ##################################################################################
    ## Setting GPU
    if not torch.cuda.is_available():
        raise Exception("CUDA not available!")
    print(torch.cuda.get_device_name(0))
    ##################################################################################

    ##################################################################################
    ## Get config parameters
    system = platform.system()
    cfg = Config(system=system)
    ##################################################################################

    if cfg.do_train:
        print("\nTRAINING!!!\n")
        train(cfg)
    else:
        print("\nOnly EVALUATING!!!\n")

        ## must match the params used for training the model
        loss_type = "EPELoss"
        use_local = True
        use_normal = True

        if loss_type == 'MySmoothL1LossSparse':
            criterion = my_losses.MySmoothL1LossSparse(size_average=cfg.size_average)
        elif loss_type == 'EPELoss':
            criterion = my_losses.EPELoss(size_average=cfg.size_average)

        eval(cfg, criterion, use_local, use_normal)


if __name__ == '__main__':
    run()

