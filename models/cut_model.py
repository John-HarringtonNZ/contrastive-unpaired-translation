import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from torchmetrics.image.fid import FrechetInceptionDistance
from mmseg.apis import inference_segmentor_remap, init_segmentor
import os

from collections import defaultdict
import wandb


PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]]

PALETTE_SUM = [sum(color) for color in PALETTE]

INV_PALETTE = defaultdict(lambda: -1)
for i, x in enumerate(PALETTE):
    INV_PALETTE[tuple(x)] = i


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")


        # Segmentation Loss Addition
        parser.add_argument('--seg_loss_lambda', type=float, default=1.0, help='weight for segmentation loss')
        parser.add_argument('--seg_loss_lambda_step', type=float, default=0.2, help='step increase per epoch')
        parser.add_argument('--seg_loss_lambda_cap', type=float, default=5.0, help='weight cap')
        parser.add_argument('--segmentation_loss', type=util.str2bool, nargs='?', const=True, default=False, help='use segmentation loss')


        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        if opt.segmentation_loss:
            self.loss_names.append('segmentation')
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if opt.segmentation_loss:
            self.visual_names.extend(['real_A_seg_viz', 'fake_B_seg_viz', ])
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # Define class members for computing FID per epoch
        # Using same feature size as default in pytorch-fid, used for CUT results
        # https://github.com/mseitzer/pytorch-fid/blob/3d604a25516746c3a4a5548c8610e99010b2c819/src/pytorch_fid/fid_score.py#L62
        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False)
        self.fid.to(device='cuda')
        self.fid_real_features_updated = False


        ## Add segmentation loss
        self.segmentation_loss = opt.segmentation_loss
        if opt.segmentation_loss:

            
            # Import Segmentation Model Class
            dir_path = os.path.dirname(os.path.realpath(__file__))

            # Instantiate Segmentation Model
            checkpoint_file = dir_path + "/../../mmsegmentation/checkpoints/pspnet_r18-d8_512x1024_80k_cityscapes_20201225_021458-09ffa746.pth"
            config_file = dir_path + "/../../mmsegmentation/configs/pspnet/pspnet_r18-d8_512x1024_80k_cityscapes.py"
            print("IMPORTING SEGMENTATION")
            self.seg_model = init_segmentor(config_file, checkpoint_file, device=self.device)

            # Set Segmentation model to Eval mode
            self.seg_model.eval()

            # Define segmentation loss
            self.segmentation_criterion = torch.nn.BCELoss()

            # Define segmentation loss hyperparameter
            self.seg_loss_lambda = opt.seg_loss_lambda
            self.seg_loss_lambda_cap = opt.seg_loss_lambda_cap
            self.seg_loss_lambda_step = opt.seg_loss_lambda_step

    
    def step_segmentation_lambda(self):
        self.seg_loss_lambda = min(self.seg_loss_lambda+self.seg_loss_lambda_step, self.seg_loss_lambda_cap)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A_seg = input['A_seg'].to(self.device)
        self.real_A_seg_viz = input['A_seg_viz'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both

        # SEGMENTATION LOSS
        if self.segmentation_loss:

            # Run segmentation model on fake_B
            fake_b_np = ((self.fake_B.cpu().detach().squeeze().numpy() + 1) * 127.5).astype(np.uint8)
            seg_fake_B = inference_segmentor_remap(self.seg_model, np.transpose(fake_b_np, (1, 2, 0)))

            self.fake_B_seg_viz = seg_fake_B

            label_seg = 19 * torch.ones((seg_fake_B.shape[1], seg_fake_B.shape[2]), dtype=torch.int64, device=self.device)            
            real_A_seg_ids = self.real_A_seg.squeeze().sum(dim=0)            

            # k = 1
            # for i in range(real_A_seg_ids.shape[0]):
            #     for j in range(real_A_seg_ids.shape[1]):
            #         if real_A_seg_ids[i,j].item() not in PALETTE_SUM and real_A_seg_ids[i,j].item() != 0:
            #             print("Count: ", k)
            #             print("loc: ", i, j)
            #             k += 1
            #             print("Not in sum: ", real_A_seg_ids[i,j].item())
            #             print("original: ", self.real_A_seg[0,:,i,j])
            #             pass

            for i, PAL in enumerate(PALETTE_SUM):
                label_seg[torch.where(real_A_seg_ids == torch.tensor(PAL, dtype=torch.int16, device=self.device))] = i

            real_label_one_hot = torch.nn.functional.one_hot(label_seg, num_classes=seg_fake_B.shape[0] + 1).to(torch.float)

            # self.fake_B_seg_viz = real_label_one_hot.permute(2, 0, 1)

            # Run segmentation loss b/w fake_B and GT
            self.loss_segmentation = self.segmentation_criterion(seg_fake_B.unsqueeze(dim=0), real_label_one_hot.permute(2, 0, 1)[:19].unsqueeze(dim=0))

            # Add seg loss to G loss
            self.loss_G += self.seg_loss_lambda * self.loss_segmentation

        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def compute_metrics_on_batch(self):
        # Convert from [-1, 1] to [0, 255] normalization
        self.fid.update(torch.round((self.fake_B+1)*127.5).to(torch.uint8), real=False)
        if not self.fid_real_features_updated:
            self.fid.update(torch.round((self.real_B+1)*127.5).to(torch.uint8), real=True)

    def reset_metrics(self):
        # Assume that once we call reset_metrics for the first time, all real features have been loaded
        self.fid_real_features_updated = True
        self.fid.reset()

    def get_metrics(self):
        return {'fid': self.fid.compute().item()}
