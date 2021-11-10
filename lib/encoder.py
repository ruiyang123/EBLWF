import copy
import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from linear_nets import MLP,fc_layer,vgg16
from lib.exemplars import ExemplarHandler
from lib.continual_learner import ContinualLearner
from lib.replayer import Replayer
import torchvision.models as models
import utils
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torch import optim
from collections import OrderedDict
from opt_fromp import opt_fromp
from torch.utils.data import ConcatDataset, DataLoader
class PostScalingLayer(nn.Module):
    def __init__(self,classes):
        super(PostScalingLayer, self).__init__()
        self.class_count = Parameter(torch.zeros(classes),requires_grad=False)
    def forward(self,x):
        if self.training:
            return x
        else:
            pts = 1/len(self.class_count)
            ptr = self.class_count/torch.sum(self.class_count)
            x = x + torch.log(pts/ptr)
            return x
    def update_ps(self,targets):
        for i in targets:
            self.class_count[i] +=1
    def reset_ps(self):
        nn.init.zeros_(self.class_count)




def meta_train_a_batch(model, x, y,batch_idx,memory=None, task=1,s=2,previous_model=None,mode="mix"):


        loss = None
        predL = None

        precision=None
        model.train()
        if task == 1:
            y_hat = model(x)
            predL = F.cross_entropy(input=y_hat,target=y,reduction="mean")
            loss_cur = predL
            model.optimizer.zero_grad()
            loss_cur.backward()
            model.optimizer.step()
            loss_replay=None
            loss = loss_cur
        elif mode == "individual":
            if memory is not None:
                batch_size_to_use = model.memory_batch
                memory_loader = iter(utils.get_data_loader(ConcatDataset(memory),
                                                                  batch_size_to_use, cuda=True, drop_last=True))
                org_weights = copy.deepcopy((OrderedDict(model.named_parameters())))
                x_m, y_m = next(memory_loader)
                x_m = x_m.cuda()
                y_m = y_m.cuda()
                for i in range(len(x_m)):
                    y_m_hat = model(x_m[i].unsqueeze(dim=0))
                    model.optimizer.zero_grad()
                    loss_replay = F.cross_entropy(input=y_m_hat, target=y_m[i].unsqueeze(dim=0), reduction="mean")
                    loss_replay.backward()
                    model.optimizer.step()
                for name, param in model.named_parameters():
                    param.data = param.data - 0.5 * (param.data.detach() - org_weights[name].data.detach())


            if previous_model is not None:
                y_l = previous_model(x)
                org_weights = copy.deepcopy((OrderedDict(model.named_parameters())))
                for i in range(len(x)):
                    y_l_hat = previous_model(x[i].unsqueeze(dim=0))
                    y_hat = model(x[i].unsqueeze(dim=0))
                    model.optimizer.zero_grad()
                    loss_lwf = utils.adjust_loss_fn_kd(y_hat, y_l_hat)
                    loss_lwf.backward()
                    model.optimizer.step()
                for name, param in model.named_parameters():
                    param.data = param.data - 0.5 * (param.data.detach() - org_weights[name].data.detach())

            org_weights = copy.deepcopy((OrderedDict(model.named_parameters())))


            for i in range(len(x)):
                y_hat = model(x[i].unsqueeze(dim=0))
                loss_cur = F.cross_entropy(input=y_hat, target=y[i].unsqueeze(dim=0), reduction="mean")
                model.optimizer.zero_grad()
                loss_cur.backward()
                model.optimizer.step()
                loss = loss_cur
            for name, param in model.named_parameters():
                param.data = param.data - 0.5 * (param.data.detach() - org_weights[name].data.detach())


        elif mode == "mix":
            org_weights = copy.deepcopy((OrderedDict(model.named_parameters())))
            batch_size_to_use = model.memory_batch
            memory_loader = iter(utils.get_data_loader(ConcatDataset(memory),
                                                              batch_size_to_use, cuda=True, drop_last=True))
            x_m,y_m = next(memory_loader)
            x_m,y_m = x_m.cuda(),y_m.cuda()
            b_x = torch.cat((x,x_m),0)
            b_y = torch.cat((y,y_m),0)
            l_order = list(range(len(b_y)))
            np.random.shuffle(l_order)

            for i in l_order:
                b_y_hat = model(b_x[i].unsqueeze(dim=0))
                predL_r = F.cross_entropy(input=b_y_hat, target=b_y[i].unsqueeze(dim=0), reduction="mean")
                loss_cur = predL_r
                model.optimizer.zero_grad()
                loss_cur.backward()
                model.optimizer.step()
            loss_replay = None
            loss = loss_cur
            for name, param in model.named_parameters():
                param.data = param.data - 0.7 * (param.data.detach() - org_weights[name].data.detach())
        elif mode =="batch":
            batch_size_to_use = model.memory_batch
            memory_loader = iter(utils.get_data_loader(ConcatDataset(memory),
                                                              batch_size_to_use, cuda=True, drop_last=True))
            x_m,y_m = next(memory_loader)
            x_m,y_m = x_m.cuda(),y_m.cuda()

            y_hat = model(x)
            loss_cur = F.cross_entropy(input=y_hat, target=y, reduction="mean")
            model.optimizer.zero_grad()
            loss_cur.backward()
            model.optimizer.step()

            y_m_hat = model(x_m)
            loss_replay = F.cross_entropy(input=y_m_hat, target=y_m, reduction="mean")
            model.optimizer.zero_grad()
            loss_replay.backward()
            model.optimizer.step()

            loss = loss_replay + loss_cur
        elif mode == "lwf_batch":
            loss_re = 0
            pre_weights = copy.deepcopy((OrderedDict(previous_model.named_parameters())))
            for name, param in model.named_parameters():
                loss_re = 0.01 * torch.mean((param - pre_weights[name])**2)


            batch_size_to_use = model.memory_batch
            memory_loader = iter(utils.get_data_loader(ConcatDataset(memory),
                                                              batch_size_to_use, cuda=True, drop_last=True))
            x_m,y_m = next(memory_loader)
            x_m,y_m = x_m.cuda(),y_m.cuda()

            y_hat = model(x)
            loss_cur = F.cross_entropy(input=y_hat, target=y, reduction="mean") + loss_re
            model.optimizer.zero_grad()
            loss_cur.backward()
            model.optimizer.step()

            y_m_hat = model(x_m)
            y_m_target = previous_model(x_m)
            loss_replay = utils.loss_fn_kd(y_m_hat, y_m_target.detach())
            model.optimizer.zero_grad()
            loss_replay.backward()
            model.optimizer.step()

            # auto lwf

            y_p_target = previous_model(x)
            y_p_hat = model(x)
            loss_lwf = utils.adjust_loss_fn_kd(y_p_hat,y_p_target)
            model.optimizer.zero_grad()
            loss_lwf.backward()
            model.optimizer.step()

            ## regularization term



            loss = loss_replay + loss_cur

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': 0,
            'distil_r': 0,
            'ewc': 0, 'si_loss': 0,
            'precision': precision if precision is not None else 0.1,
        }



class Classifier(ContinualLearner, Replayer, ExemplarHandler):
    '''Model for classifying images, "enriched" as "ContinualLearner"-, Replayer- and ExemplarHandler-object.'''

    def __init__(self, image_size, image_channels, classes,
                 fc_layers=3, fc_units=1000, fc_drop=0, fc_bn=True, fc_nl="relu", gated=False,
                 bias=True, excitability=False, excit_buffer=False, binaryCE=False, binaryCE_distill=False,AGEM=False,fromp=False,select_method=None,ps=None,auto_lwf=None, contrasive=None):

        # configurations
        super().__init__()
        self.classes = classes
        self.label = "Classifier"
        self.fc_layers = fc_layers
        self.AGEM = AGEM

        self.auto_lwf = auto_lwf
        self.contrasive_loss = contrasive
        self.meta = 0

        # training regime

        self.lr = None
        self.gammalr=None
        self.schedular = None
        self.optim_type = None
        self.use_schedular = 0
        self.re_init = 0
        self.drop_p = 0
        self.memory_batch = 4

        # meta update
        self.meta_mode = None


        # settings for fromp
        self.fromp = fromp
        self.memorable_points =None
        self.memorableloader=None
        self.select_method = select_method

        # settings for training
        self.binaryCE = binaryCE
        self.binaryCE_distill = binaryCE_distill

        # check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")

        # Get inter feature maps
        self.handlers = []
        self.list_attentions = []


        ######------SPECIFY MODEL------######

        # flatten image to 2D-tensor
        #self.flatten = utils.Flatten()

        # fully connected hidden layers
        # self.fcE = MLP(input_size=image_channels*image_size**2, output_size=fc_units, layers=fc_layers-1,
        #                hid_size=fc_units, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
        #                excitability=excitability, excit_buffer=excit_buffer, gated=gated)
        # mlp_output_size = fc_units if fc_layers>1 else image_channels*image_size**2
        # print('*************num of classes in encoder: '+str(classes))
        model = models.resnet18(pretrained=False)
        self.feature_extractor_layer = nn.Sequential(*list(model.children())[:-1])
        self.out_layers = nn.Linear(512,classes)

        #self.vgg.fc=nn.Linear(512,classes)
        #self.vgg = models.resnet18(num_classes=classes)

        self.ps = ps
        if self.ps:
            self.ps_layer = PostScalingLayer(classes)




        #self.vgg = vgg16(classes)
        # classifier
        #self.classifier = fc_layer(mlp_output_size, classes, excit_buffer=True, nl='none', drop=fc_drop)


    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        #return "{}_c{}".format(self.fcE.name, self.classes)
        return "resnet18"


    def remove_hooks(self):
        for handle in self.handlers:
            handle.remove()

    def register_hook(self):
        def get_inter_feature(module,fea_in,fea_out):
            self.list_attentions.append(fea_out)
        for name,module in self.feature_extractor_layer.named_modules():
            if isinstance(module,(nn.Conv2d,nn.ReLU,nn.BatchNorm2d)):
                self.handlers.append(module.register_forward_hook(get_inter_feature))



    def forward(self, x):
        if self.ps:
            x = self.feature_extractor_layer(x)
            x = torch.flatten(x, 1)

            if self.drop_p!=0:
                x = F.dropout(x,self.drop_p,self.training)
            x = self.out_layers(x)
            x = self.ps_layer(x)
            return x
        else:
            x = self.feature_extractor_layer(x)
            x = torch.flatten(x, 1)

            if self.drop_p!=0:
                x = F.dropout(x,self.drop_p,self.training)
            x = self.out_layers(x)
            return x

    def feature_extractor(self, images):
        return self.feature_extractor_layer(images)
    def get_feature(self,x):
        x = self.feature_extractor_layer(x)
        x = torch.flatten(x, 1)
        x = self.out_layers(x)
        return x

    def re_init_optim(self):
        self.optim_list = [{'params': filter(lambda p: p.requires_grad,self.parameters()), 'lr': self.lr}]
        if self.optim_type in ("adam", "adam_reset"):
            self.optimizer = optim.Adam(self.optim_list, betas=(0.9, 0.999))
        elif self.optim_type == "sgd":
            self.optimizer = optim.SGD(self.optim_list)
        if self.use_schedular == 1:
            self.schedular = ExponentialLR(self.optimizer,gamma=self.gammalr)


    def train_a_batch(self, x, y, scores=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None, task=1,list_attentions_previous=None):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_/scores_]).

        [x]               <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]               <tensor> batch of corresponding labels
        [scores]          None or <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x]

        [x_]              None or (<list> of) <tensor> batch of replayed inputs
        [y_]              None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [scores_]         None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]             <number> in [0,1], relative importance of new task
        [active_classes]  None or (<list> of) <list> with "active" classes
        [task]            <int>, for setting task-specific mask'''

        # Set model to training-mode
        self.train()

        # Post-scaling

        if self.ps:
            self.ps_layer.update_ps(y)
            print(y)
            print(self.ps_layer.class_count)




        # Reset optimizer
        self.optimizer.zero_grad()

        gradient_per_task = True if ((self.mask_dict is not None) and (x_ is not None)) else False

        ##--(1)-- REPLAYED DATA --##

        if x_ is not None:
            # In the Task-IL scenario, [y_] or [scores_] is a list and [x_] needs to be evaluated on each of them
            # (in case of 'exact' or 'exemplar' replay, [x_] is also a list!
            TaskIL = (type(y_) == list) if (y_ is not None) else (type(scores_) == list)
            if not TaskIL:
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else len(scores_)

            # Prepare lists to store losses for each replay
            loss_replay = [None] * n_replays
            predL_r = [None] * n_replays
            distilL_r = [None] * n_replays


            if not self.contrasive_loss:
                # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)
                if (not type(x_) == list) and (self.mask_dict is None):
                    y_hat_all = self(x_)

                # Loop to evalute predictions on replay according to each previous task
                for replay_id in range(n_replays):

                    # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                    if (type(x_) == list) or (self.mask_dict is not None):
                        x_temp_ = x_[replay_id] if type(x_) == list else x_
                        if self.mask_dict is not None:
                            self.apply_XdGmask(task=replay_id + 1)
                        y_hat_all = self(x_temp_)

                    # -if needed (e.g., Task-IL or Class-IL scenario), remove predictions for classes not in replayed task
                    y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]

                    # Calculate losses
                    if (y_ is not None) and (y_[replay_id] is not None):
                        if self.binaryCE:
                            binary_targets_ = utils.to_one_hot(y_[replay_id].cpu(), y_hat.size(1)).to(y_[replay_id].device)
                            predL_r[replay_id] = F.binary_cross_entropy_with_logits(
                                input=y_hat, target=binary_targets_, reduction='none'
                            ).sum(dim=1).mean()  # --> sum over classes, then average over batch
                        else:
                            predL_r[replay_id] = F.cross_entropy(y_hat, y_[replay_id], reduction='mean')
                    if (scores_ is not None) and (scores_[replay_id] is not None):
                        # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                        n_classes_to_consider = y_hat.size(1)  # --> zeros will be added to [scores] to make it this size!
                        kd_fn = utils.loss_fn_kd_binary if self.binaryCE else utils.loss_fn_kd
                        if self.auto_lwf:
                            kd_fn = utils.adjust_loss_fn_kd



                        distilL_r[replay_id] = kd_fn(scores=y_hat[:, :n_classes_to_consider],
                                                     target_scores=scores_[replay_id], T=self.KD_temp)
                    # Weigh losses
                    if self.replay_targets == "hard":
                        loss_replay[replay_id] = predL_r[replay_id]
                    elif self.replay_targets == "soft":
                        loss_replay[replay_id] = distilL_r[replay_id]

                    # If needed, perform backward pass before next task-mask (gradients of all tasks will be accumulated)
                    if gradient_per_task:
                        weight = 1 if self.AGEM else (1 - rnt)
                        weighted_replay_loss_this_task = weight * loss_replay[replay_id] / n_replays
                        weighted_replay_loss_this_task.backward()
            else:
                f1 = self.feature_extractor(x)
                f2 = self.feature_extractor(x_)
                cos_loss = utils.cos_sim_loss(f1,f2,y,y_)


        # Calculate total replay loss
        if not self.contrasive_loss:
            loss_replay = None if (x_ is None) else sum(loss_replay) / n_replays
        else:
            loss_replay = None if (x_ is None) else cos_loss

        # If using A-GEM, calculate and store averaged gradient of replayed data
        if self.AGEM and x_ is not None:
            # Perform backward pass to calculate gradient of replayed batch (if not yet done)
            if not gradient_per_task:
                loss_replay.backward()
            # Reorganize the gradient of the replayed batch as a single vector
            grad_rep = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_rep.append(p.grad.view(-1))
            grad_rep = torch.cat(grad_rep)
            # Reset gradients (with A-GEM, gradients of replayed batch should only be used as inequality constraint)
            self.optimizer.zero_grad()

        ##--(2)-- CURRENT DATA --##

        if x is not None :
            # If requested, apply correct task-specific mask
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            # Run model
            self.list_attentions = []
            y_hat = self(x)
            # -if needed, remove predictions for classes not in current task
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
                y_hat = y_hat[:, class_entries]

            # Calculate prediction loss
            if self.binaryCE:
                # -binary prediction loss
                binary_targets = utils.to_one_hot(y.cpu(), y_hat.size(1)).to(y.device)
                if self.binaryCE_distill and (scores is not None):
                    classes_per_task = int(y_hat.size(1) / task)
                    binary_targets = binary_targets[:, -(classes_per_task):]
                    binary_targets = torch.cat([torch.sigmoid(scores / self.KD_temp), binary_targets], dim=1)
                predL = None if y is None else F.binary_cross_entropy_with_logits(
                    input=y_hat, target=binary_targets, reduction='none'
                ).sum(dim=1).mean()  # --> sum over classes, then average over batch
            else:
                # -multiclass prediction loss
                predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='mean')

            # Weigh losses
            loss_cur = predL

            # Calculate training-precision
            precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

            # If backward passes are performed per task (e.g., XdG combined with replay), perform backward pass
            if gradient_per_task:
                weighted_current_loss = rnt * loss_cur
                weighted_current_loss.backward()
        if self.fromp:
            if isinstance(self.optimizer, opt_fromp):
                def closure():
                    self.optimizer.zero_grad()
                    y_hat = self(x)
                    loss_cur = F.cross_entropy(input=y_hat, target=y, reduction='mean')
                    return loss_cur, y_hat
                def closure_memorable_points(task):
                    memorable_points_t = self.memorable_points[task-1][0]
                    memorable_points_t = memorable_points_t.cuda()
                    self.optimizer.zero_grad()
                    y_hat = self(memorable_points_t)
                    return y_hat

        else:
            precision = predL = None
            # -> it's possible there is only "replay" [e.g., for offline with task-incremental learning]

        # Combine loss from current and replayed batch
        if x_ is None or self.AGEM :
            loss_total = loss_cur
        else:
            loss_total = loss_replay if (x is None) else rnt * loss_cur + (1 - rnt) * loss_replay


        ##--(3)-- ALLOCATION LOSSES --##

        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss()
        if self.si_c > 0:
            loss_total += self.si_c * surrogate_loss

        # Add EWC-loss
        ewc_loss = self.ewc_loss()
        if self.ewc_lambda > 0:
            loss_total += self.ewc_lambda * ewc_loss

        # Add Attention-loss
        if list_attentions_previous:
            attention_loss = utils.pod(self.list_attentions,list_attentions_previous)
            loss_total = 0.5 * loss_total + 1 * attention_loss


        # Backpropagate errors (if not yet done)
        if not gradient_per_task and not self.fromp:
            loss_total.backward()

        # If using A-GEM, potentially change gradient:
        if self.AGEM and x_ is not None:
            # -reorganize gradient (of current batch) as single vector
            grad_cur = []
            for p in self.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur * grad_rep).sum()
            if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_rep * grad_rep).sum()
                grad_proj = grad_cur - (angle / length_rep) * grad_rep
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index + n_param].view_as(p))
                        index += n_param

        # Take optimization-step
        if self.fromp:
            loss_total,_ = self.optimizer.step(closure, closure_memorable_points, task-1)
            loss_cur = loss_total
        else:
            self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': sum(predL_r).item() / n_replays if (x_ is not None and predL_r[0] is not None) else 0,
            'distil_r': sum(distilL_r).item() / n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
            'ewc': ewc_loss.item(), 'si_loss': surrogate_loss.item(),
            'precision': precision if precision is not None else 0.,
        }
